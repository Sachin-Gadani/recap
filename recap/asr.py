"""Local speech-to-text backends: faster-whisper (default) and whisper.cpp."""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path

from . import audio as audio_mod
from .config import ASRConfig
from .transcript import TRANSCRIPT_SUFFIXES, Segment, Transcript, merge_short_segments
from .util import fmt_ts, human_duration

ProgressFn = Callable[[float, float, str], None]
"""Called as (seconds_done, seconds_total, latest_text)."""


class ASRError(RuntimeError):
    pass


def transcribe(
    audio_path: str | Path,
    config: ASRConfig,
    work_dir: Path,
    on_progress: ProgressFn | None = None,
    audio_sha: str | None = None,
) -> Transcript:
    _reject_transcripts(audio_path)
    backend = config.backend.strip().lower()
    if backend in {"faster-whisper", "faster_whisper", "fw"}:
        transcript = _faster_whisper(audio_path, config, work_dir, on_progress)
    elif backend in {"whisper.cpp", "whisper-cpp", "whispercpp"}:
        transcript = _whisper_cpp(audio_path, config, work_dir, on_progress)
    else:
        raise ASRError(
            f"unknown asr backend {config.backend!r} (expected 'faster-whisper' or 'whisper.cpp')"
        )
    transcript.segments = merge_short_segments(transcript.segments)
    transcript.source = str(audio_path)
    transcript.audio_sha256 = audio_sha
    transcript.asr_backend = backend
    transcript.asr_model = config.model
    if not transcript.segments:
        raise ASRError(
            "no speech was transcribed. Check that the file contains audible speech, "
            "and try --asr-model small or medium."
        )
    return transcript


def _reject_transcripts(audio_path: str | Path) -> None:
    """A transcript handed to Whisper fails deep inside the audio decoder. Catch it here."""
    path = Path(audio_path)
    if path.suffix.lower() in TRANSCRIPT_SUFFIXES:
        raise ASRError(
            f"{path.name} is a transcript, not audio - there is nothing to transcribe.\n"
            f"Fix: recap summarize {path}"
        )


def _resolve_device(config: ASRConfig) -> tuple[str, str]:
    device = config.device
    compute = config.compute_type
    if device == "auto":
        device = "cpu"
        try:  # pragma: no cover - depends on the host having torch/CUDA
            import torch  # type: ignore

            if torch.cuda.is_available():
                device = "cuda"
        except Exception:
            pass
    if compute == "auto":
        compute = "float16" if device == "cuda" else "int8"
    return device, compute


def _faster_whisper(
    audio_path: str | Path,
    config: ASRConfig,
    work_dir: Path,
    on_progress: ProgressFn | None,
) -> Transcript:
    try:
        from faster_whisper import WhisperModel  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised by `recap doctor`
        raise ASRError(
            "faster-whisper is not installed.\n"
            "Fix: pip install 'recap[whisper]'  (or: pip install faster-whisper)"
        ) from exc

    device, compute_type = _resolve_device(config)
    try:
        model = WhisperModel(config.model, device=device, compute_type=compute_type)
    except Exception as exc:
        raise ASRError(
            f"could not load Whisper model {config.model!r} on {device}/{compute_type}: {exc}\n"
            "The first run downloads the model; after that it is fully offline."
        ) from exc

    try:
        segments_iter, info = model.transcribe(
            str(audio_path),
            language=config.language,
            beam_size=config.beam_size,
            vad_filter=config.vad_filter,
            initial_prompt=config.initial_prompt,
            condition_on_previous_text=False,  # avoids runaway repetition on long audio
        )
    except Exception as exc:
        # The decoder raises whatever PyAV/ffmpeg felt like; none of it is actionable.
        raise ASRError(
            f"could not decode {Path(audio_path).name} as audio ({type(exc).__name__}: {exc}).\n"
            "Check the file plays in another program. If it is already a transcript, "
            f"use: recap summarize {audio_path}"
        ) from exc
    total = float(getattr(info, "duration", 0.0) or 0.0)
    segments: list[Segment] = []
    for item in segments_iter:
        text = (item.text or "").strip()
        if not text:
            continue
        segments.append(Segment(start=float(item.start), end=float(item.end), text=text))
        if on_progress:
            on_progress(float(item.end), total, text)
    return Transcript(
        segments=segments,
        language=getattr(info, "language", None) or config.language,
        duration=total,
    )


def _whisper_cpp(
    audio_path: str | Path,
    config: ASRConfig,
    work_dir: Path,
    on_progress: ProgressFn | None,
) -> Transcript:
    binary = shutil.which(config.whisper_cpp_bin) or shutil.which("main")
    if not binary:
        raise ASRError(
            f"whisper.cpp binary {config.whisper_cpp_bin!r} not found on PATH.\n"
            "Build it from https://github.com/ggml-org/whisper.cpp, or set "
            "asr.whisper_cpp_bin to its full path."
        )
    model_path = config.whisper_cpp_model
    if not model_path or not Path(model_path).exists():
        raise ASRError(
            "whisper.cpp needs a ggml model file. Set asr.whisper_cpp_model to e.g. "
            "models/ggml-small.en.bin (download with whisper.cpp/models/download-ggml-model.sh)."
        )

    # whisper.cpp only reads 16 kHz mono PCM wav.
    info = audio_mod.probe(audio_path)
    if audio_mod.is_already_16k_mono_wav(info):
        wav_path = Path(audio_path)
    else:
        wav_path = audio_mod.to_wav16k(audio_path, work_dir / "audio-16k.wav")

    with tempfile.TemporaryDirectory(dir=work_dir) as tmp:
        prefix = Path(tmp) / "out"
        command = [
            binary, "-m", str(model_path), "-f", str(wav_path),
            "-oj", "-of", str(prefix), "-np",
        ]
        if config.language:
            command += ["-l", config.language]
        if config.whisper_cpp_threads > 0:
            command += ["-t", str(config.whisper_cpp_threads)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            tail = "\n".join(result.stderr.strip().splitlines()[-12:])
            raise ASRError(f"whisper.cpp exited {result.returncode}:\n{tail}")
        json_path = prefix.with_suffix(".json")
        if not json_path.exists():
            raise ASRError(f"whisper.cpp produced no JSON output at {json_path}")
        data = json.loads(json_path.read_text(encoding="utf-8"))

    segments = [
        Segment(
            start=float(item.get("offsets", {}).get("from", 0)) / 1000.0,
            end=float(item.get("offsets", {}).get("to", 0)) / 1000.0,
            text=str(item.get("text", "")).strip(),
        )
        for item in data.get("transcription", [])
        if str(item.get("text", "")).strip()
    ]
    if on_progress and segments:
        last = segments[-1]
        on_progress(last.end, last.end, last.text)
    return Transcript(
        segments=segments,
        language=(data.get("result") or {}).get("language") or config.language,
        duration=segments[-1].end if segments else 0.0,
    )


def format_progress(done: float, total: float, text: str, width: int = 24) -> str:
    """One-line progress string for the terminal."""
    if total > 0:
        ratio = min(1.0, done / total)
        filled = int(ratio * width)
        bar = "#" * filled + "-" * (width - filled)
        head = f"[{bar}] {ratio * 100:5.1f}%  {fmt_ts(done)}/{fmt_ts(total)}"
    else:
        head = f"transcribed {human_duration(done)}"
    snippet = " ".join(text.split())[:48]
    return f"{head}  {snippet}"
