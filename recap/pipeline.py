"""End-to-end orchestration: audio -> transcript -> notes -> executive summary."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from . import asr, render
from . import audio as audio_mod
from .config import Config, check_privacy
from .llm import BaseClient, build_client
from .summarize import Summary, summarize_transcript
from .transcript import Transcript, load_transcript
from .util import (
    human_duration,
    info,
    sha256_file,
    slugify,
    step,
    warn,
    write_atomic,
    write_json,
)

ALL_FORMATS = ("md", "txt", "json", "transcript.md", "transcript.txt", "srt", "vtt")
DEFAULT_FORMATS = ("md", "json", "transcript.md")


@dataclass
class RunResult:
    summary: Summary
    transcript: Transcript
    work_dir: Path
    outputs: dict[str, Path] = field(default_factory=dict)
    elapsed: float = 0.0


def work_dir_for(config: Config, source: Path, digest: str | None) -> Path:
    stem = slugify(source.stem, 48)
    suffix = (digest or "nodigest")[:10]
    path = Path(config.work_root).expanduser() / f"{stem}-{suffix}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def prepare_transcript(
    source: Path,
    config: Config,
    work_dir: Path,
    force: bool = False,
    quiet: bool = False,
    digest: str | None = None,
) -> Transcript:
    """Transcribe ``source``, reusing a cached transcript when the audio is unchanged."""
    cached = work_dir / "transcript.json"
    if cached.exists() and not force:
        try:
            transcript = load_transcript(cached)
            if transcript.segments:
                if not quiet:
                    info(f"reusing cached transcript ({cached})")
                return transcript
        except (ValueError, OSError) as exc:
            warn(f"ignoring unreadable cached transcript: {exc}")

    audio_info = audio_mod.probe(source)
    if not quiet:
        length = human_duration(audio_info.duration) if audio_info.duration else "unknown length"
        step(f"Transcribing {source.name} ({length}) with {config.asr.backend} / {config.asr.model}")
        if config.asr.backend.startswith("faster") and not audio_mod.have_ffmpeg():
            info("ffmpeg not found; faster-whisper will decode the file itself if it can")

    last_print = 0.0

    def on_progress(done: float, total: float, text: str) -> None:
        nonlocal last_print
        now = time.monotonic()
        if quiet or now - last_print < 0.5:
            return
        last_print = now
        print("\r\033[K    " + asr.format_progress(done, total, text), end="", flush=True)

    started = time.monotonic()
    transcript = asr.transcribe(
        source, config.asr, work_dir, on_progress=None if quiet else on_progress, audio_sha=digest
    )
    if not quiet:
        print("\r\033[K", end="", flush=True)
        info(
            f"{len(transcript.segments)} segments, {transcript.word_count():,} words "
            f"in {human_duration(time.monotonic() - started)}"
        )
    transcript.save(cached)
    return transcript


def build_llm_client(config: Config, check: bool = True) -> BaseClient:
    import os

    check_privacy(config)
    api_key = os.environ.get(config.llm.api_key_env) or None
    client = build_client(config.llm, api_key)
    if check:
        client.ensure_ready()
        if config.llm.reduce_model:
            client.with_model(config.llm.reduce_model).ensure_ready()
    return client


def summarize(
    transcript: Transcript,
    config: Config,
    work_dir: Path,
    client: BaseClient | None = None,
    use_cache: bool = True,
    quiet: bool = False,
) -> Summary:
    client = client or build_llm_client(config)
    if not quiet:
        step(f"Summarising with {client.name} / {client.config.model}")
    cache_dir = (work_dir / "cache") if use_cache else None
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)

    def on_progress(message: str) -> None:
        if not quiet:
            info(message)

    return summarize_transcript(transcript, client, config, cache_dir, on_progress)


def write_outputs(
    result_summary: Summary,
    transcript: Transcript,
    out_base: Path,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    include_notes: bool = False,
) -> dict[str, Path]:
    """Write the requested formats using ``out_base`` as the path stem."""
    out_base.parent.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    for fmt in formats:
        if fmt == "md":
            written[fmt] = write_atomic(
                out_base.with_suffix(".md"), render.render_markdown(result_summary, include_notes)
            )
        elif fmt == "txt":
            written[fmt] = write_atomic(
                out_base.with_suffix(".txt"), render.render_plain(result_summary)
            )
        elif fmt == "json":
            written[fmt] = write_json(out_base.with_suffix(".json"), result_summary.to_dict())
        elif fmt == "transcript.md":
            written[fmt] = write_atomic(
                out_base.with_name(out_base.name + ".transcript.md"),
                render.render_transcript_markdown(transcript, f"{result_summary.title} - transcript"),
            )
        elif fmt == "transcript.txt":
            written[fmt] = write_atomic(
                out_base.with_name(out_base.name + ".transcript.txt"), transcript.to_txt()
            )
        elif fmt == "srt":
            written[fmt] = write_atomic(out_base.with_suffix(".srt"), transcript.to_srt())
        elif fmt == "vtt":
            written[fmt] = write_atomic(out_base.with_suffix(".vtt"), transcript.to_vtt())
        else:
            raise ValueError(f"unknown output format {fmt!r} (choose from {', '.join(ALL_FORMATS)})")
    return written


def run(
    source: Path,
    config: Config,
    out_base: Path | None = None,
    formats: tuple[str, ...] = DEFAULT_FORMATS,
    transcript_path: Path | None = None,
    force_transcribe: bool = False,
    use_cache: bool = True,
    include_notes: bool = False,
    quiet: bool = False,
) -> RunResult:
    started = time.monotonic()

    # Fail fast on a missing model rather than after an hour of transcription.
    client = build_llm_client(config)

    if transcript_path is not None:
        transcript = load_transcript(transcript_path)
        digest = transcript.audio_sha256
        work_dir = work_dir_for(config, Path(transcript_path), digest or "given")
        if not quiet:
            step(f"Using transcript {transcript_path} ({transcript.word_count():,} words)")
    else:
        digest = sha256_file(source)
        work_dir = work_dir_for(config, source, digest)
        transcript = prepare_transcript(
            source, config, work_dir, force=force_transcribe, quiet=quiet, digest=digest
        )

    summary = summarize(transcript, config, work_dir, client=client, use_cache=use_cache, quiet=quiet)

    base = out_base or Path.cwd() / slugify(summary.title or source.stem, 60)
    outputs = write_outputs(summary, transcript, base, formats, include_notes)
    write_json(work_dir / "summary.json", summary.to_dict())

    return RunResult(
        summary=summary,
        transcript=transcript,
        work_dir=work_dir,
        outputs=outputs,
        elapsed=time.monotonic() - started,
    )
