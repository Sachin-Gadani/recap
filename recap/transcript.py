"""Transcript data model plus readers/writers for json, srt, vtt and plain text."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .util import estimate_tokens, fmt_ts, fmt_ts_ms, parse_ts

SCHEMA_VERSION = 1


@dataclass
class Segment:
    start: float
    end: float
    text: str
    speaker: str | None = None

    @property
    def duration(self) -> float:
        return max(0.0, self.end - self.start)

    def label(self) -> str:
        """``[MM:SS] Speaker: text`` - the form the LLM sees."""
        prefix = f"[{fmt_ts(self.start)}]"
        if self.speaker:
            return f"{prefix} {self.speaker}: {self.text.strip()}"
        return f"{prefix} {self.text.strip()}"


@dataclass
class Transcript:
    segments: list[Segment] = field(default_factory=list)
    language: str | None = None
    duration: float = 0.0
    source: str | None = None
    asr_model: str | None = None
    asr_backend: str | None = None
    created_at: str = ""
    audio_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.created_at:
            self.created_at = datetime.now(UTC).isoformat(timespec="seconds")
        if not self.duration and self.segments:
            self.duration = self.segments[-1].end

    @property
    def text(self) -> str:
        return "\n".join(s.text.strip() for s in self.segments if s.text.strip())

    @property
    def labelled_text(self) -> str:
        return "\n".join(s.label() for s in self.segments if s.text.strip())

    @property
    def speakers(self) -> list[str]:
        seen: list[str] = []
        for segment in self.segments:
            if segment.speaker and segment.speaker not in seen:
                seen.append(segment.speaker)
        return seen

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.labelled_text)

    def word_count(self) -> int:
        return len(self.text.split())

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Transcript:
        raw_segments = data.get("segments") or []
        segments = [
            Segment(
                start=float(s.get("start", 0.0)),
                end=float(s.get("end", s.get("start", 0.0))),
                text=str(s.get("text", "")).strip(),
                speaker=s.get("speaker") or None,
            )
            for s in raw_segments
        ]
        return cls(
            segments=segments,
            language=data.get("language"),
            duration=float(data.get("duration") or 0.0),
            source=data.get("source"),
            asr_model=data.get("asr_model") or data.get("model"),
            asr_backend=data.get("asr_backend"),
            created_at=data.get("created_at", ""),
            audio_sha256=data.get("audio_sha256"),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    # -- exports ------------------------------------------------------------
    def to_txt(self) -> str:
        return self.text + "\n"

    def to_markdown(self, heading: str = "Transcript") -> str:
        lines = [f"# {heading}", ""]
        for segment in self.segments:
            if segment.text.strip():
                lines.append(f"`{fmt_ts(segment.start)}` {_speaker_prefix(segment)}{segment.text.strip()}")
                lines.append("")
        return "\n".join(lines)

    def to_srt(self) -> str:
        blocks = []
        for index, segment in enumerate(self.segments, start=1):
            blocks.append(
                f"{index}\n"
                f"{fmt_ts_ms(segment.start)} --> {fmt_ts_ms(segment.end)}\n"
                f"{_speaker_prefix(segment)}{segment.text.strip()}\n"
            )
        return "\n".join(blocks)

    def to_vtt(self) -> str:
        blocks = ["WEBVTT", ""]
        for segment in self.segments:
            blocks.append(
                f"{fmt_ts_ms(segment.start, '.')} --> {fmt_ts_ms(segment.end, '.')}\n"
                f"{_speaker_prefix(segment)}{segment.text.strip()}\n"
            )
        return "\n".join(blocks)


def _speaker_prefix(segment: Segment) -> str:
    return f"{segment.speaker}: " if segment.speaker else ""


# --- loading ---------------------------------------------------------------

_SRT_TIME_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3})"
)
_SPEAKER_RE = re.compile(r"^\s*(?:\[(?P<b>[^\]]{1,40})\]|(?P<a>[A-Z][\w .'-]{0,39}))\s*:\s+")


def load_transcript(path: str | Path) -> Transcript:
    """Load a transcript from .json (ours or whisper's), .srt, .vtt or .txt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"transcript not found: {path}")
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8", errors="replace")
    if suffix == ".json":
        transcript = _from_json(json.loads(text))
    elif suffix in {".srt", ".vtt"}:
        transcript = _from_subtitles(text)
    else:
        transcript = _from_plain_text(text)
    transcript.source = transcript.source or str(path)
    return transcript


def _from_json(data: Any) -> Transcript:
    if isinstance(data, list):
        return Transcript.from_dict({"segments": data})
    if not isinstance(data, dict):
        raise ValueError("unsupported JSON transcript: expected an object or array")
    if "segments" in data:
        return Transcript.from_dict(data)
    # whisper.cpp: {"transcription": [{"offsets": {"from": ms, "to": ms}, "text": ...}]}
    if "transcription" in data:
        segments = [
            Segment(
                start=float(item.get("offsets", {}).get("from", 0)) / 1000.0,
                end=float(item.get("offsets", {}).get("to", 0)) / 1000.0,
                text=str(item.get("text", "")).strip(),
            )
            for item in data["transcription"]
        ]
        language = (data.get("result") or {}).get("language")
        return Transcript(segments=segments, language=language, asr_backend="whisper.cpp")
    if "text" in data:
        return _from_plain_text(str(data["text"]))
    raise ValueError("unsupported JSON transcript: no 'segments', 'transcription' or 'text' key")


def _from_subtitles(text: str) -> Transcript:
    segments: list[Segment] = []
    blocks = re.split(r"\n\s*\n", text.strip())
    for block in blocks:
        match = _SRT_TIME_RE.search(block)
        if not match:
            continue
        lines = block.splitlines()
        cue_index = next(i for i, line in enumerate(lines) if _SRT_TIME_RE.search(line))
        body = " ".join(line.strip() for line in lines[cue_index + 1 :]).strip()
        if not body:
            continue
        speaker, body = _split_speaker(body)
        segments.append(
            Segment(
                start=parse_ts(match.group(1)),
                end=parse_ts(match.group(2)),
                text=body,
                speaker=speaker,
            )
        )
    if not segments:
        raise ValueError("no subtitle cues found")
    return Transcript(segments=segments)


def _from_plain_text(text: str) -> Transcript:
    """Best-effort: one segment per non-empty line, timestamps parsed when present."""
    segments: list[Segment] = []
    cursor = 0.0
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        start: float | None = None
        stamped = re.match(r"^[\[(<]?\s*((?:\d+:)?\d{1,2}:\d{2}(?:[.,]\d{1,3})?)\s*[\])>]?\s+(.*)$", line)
        if stamped:
            try:
                start = parse_ts(stamped.group(1))
                line = stamped.group(2).strip()
            except ValueError:
                start = None
        speaker, body = _split_speaker(line)
        if not body:
            continue
        if start is None:
            start = cursor
        # Assume ~2.5 words/second when the source carries no timing at all.
        end = start + max(1.0, len(body.split()) / 2.5)
        segments.append(Segment(start=start, end=end, text=body, speaker=speaker))
        cursor = end
    if not segments:
        raise ValueError("transcript file is empty")
    return Transcript(segments=segments)


def _split_speaker(line: str) -> tuple[str | None, str]:
    match = _SPEAKER_RE.match(line)
    if not match:
        return None, line.strip()
    speaker = match.group("b") or match.group("a")
    return speaker.strip(), line[match.end() :].strip()


def merge_short_segments(segments: Iterable[Segment], min_chars: int = 40) -> list[Segment]:
    """Glue tiny ASR fragments together so chunks read as sentences, not stutters."""
    merged: list[Segment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        if (
            merged
            and merged[-1].speaker == segment.speaker
            and len(merged[-1].text) < min_chars
            and not merged[-1].text.endswith((".", "!", "?"))
        ):
            merged[-1] = Segment(
                start=merged[-1].start,
                end=segment.end,
                text=f"{merged[-1].text} {text}".strip(),
                speaker=segment.speaker,
            )
        else:
            merged.append(Segment(segment.start, segment.end, text, segment.speaker))
    return merged
