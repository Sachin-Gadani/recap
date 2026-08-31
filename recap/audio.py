"""Audio probing and normalisation. ffmpeg is used when present, never required."""

from __future__ import annotations

import json
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

AUDIO_SUFFIXES = {
    ".wav", ".mp3", ".m4a", ".mp4", ".aac", ".flac", ".ogg", ".oga", ".opus",
    ".wma", ".aiff", ".aif", ".webm", ".mkv", ".mov", ".m4b", ".amr", ".3gp",
}


class AudioError(RuntimeError):
    pass


@dataclass
class AudioInfo:
    path: Path
    duration: float | None
    sample_rate: int | None = None
    channels: int | None = None
    codec: str | None = None


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def have_ffprobe() -> bool:
    return shutil.which("ffprobe") is not None


def probe(path: str | Path) -> AudioInfo:
    """Return duration/format info, via ffprobe when available, else the wave module."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"audio file not found: {path}")
    if have_ffprobe():
        try:
            return _probe_ffprobe(path)
        except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, ValueError):
            pass  # fall through to the stdlib reader
    if path.suffix.lower() == ".wav":
        try:
            with wave.open(str(path), "rb") as handle:
                frames = handle.getnframes()
                rate = handle.getframerate()
                return AudioInfo(
                    path=path,
                    duration=frames / rate if rate else None,
                    sample_rate=rate,
                    channels=handle.getnchannels(),
                    codec="pcm",
                )
        except wave.Error:
            pass
    return AudioInfo(path=path, duration=None)


def _probe_ffprobe(path: Path) -> AudioInfo:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", "-select_streams", "a:0", str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    data = json.loads(result.stdout)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    duration = stream.get("duration") or fmt.get("duration")
    return AudioInfo(
        path=path,
        duration=float(duration) if duration else None,
        sample_rate=int(stream["sample_rate"]) if stream.get("sample_rate") else None,
        channels=stream.get("channels"),
        codec=stream.get("codec_name"),
    )


def to_wav16k(source: str | Path, dest: str | Path, overwrite: bool = False) -> Path:
    """Decode ``source`` to 16 kHz mono 16-bit PCM - what every Whisper build wants."""
    source, dest = Path(source), Path(dest)
    if dest.exists() and not overwrite:
        return dest
    if not have_ffmpeg():
        raise AudioError(
            "ffmpeg is required to decode this file.\n"
            "  macOS:  brew install ffmpeg\n"
            "  Debian: sudo apt install ffmpeg\n"
            "  Windows: winget install Gyan.FFmpeg"
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".tmp.wav")
    command = [
        "ffmpeg", "-nostdin", "-y", "-i", str(source),
        "-vn", "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        str(tmp),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        tmp.unlink(missing_ok=True)
        tail = "\n".join(result.stderr.strip().splitlines()[-12:])
        raise AudioError(f"ffmpeg failed to decode {source.name}:\n{tail}")
    tmp.replace(dest)
    return dest


def is_already_16k_mono_wav(info: AudioInfo) -> bool:
    return (
        info.path.suffix.lower() == ".wav"
        and info.sample_rate == 16000
        and info.channels == 1
        and info.codec in {"pcm", "pcm_s16le"}
    )
