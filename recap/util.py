"""Small helpers shared across the pipeline: no third-party imports allowed here."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import unicodedata
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Rough tokens-per-character ratio. Good enough for budgeting chunk sizes; we
# deliberately avoid pulling in a tokenizer dependency just to count.
_CHARS_PER_TOKEN = 3.8


def estimate_tokens(text: str) -> int:
    """Cheap upper-ish estimate of the token count of ``text``."""
    if not text:
        return 0
    return int(len(text) / _CHARS_PER_TOKEN) + 1


def fmt_ts(seconds: float, always_hours: bool = False) -> str:
    """Format ``seconds`` as ``MM:SS`` or ``H:MM:SS``."""
    if seconds is None:
        return "??:??"
    seconds = max(0.0, float(seconds))
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours or always_hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def fmt_ts_ms(seconds: float, sep: str = ",") -> str:
    """Format ``seconds`` as ``HH:MM:SS,mmm`` (SRT) or ``HH:MM:SS.mmm`` (VTT)."""
    seconds = max(0.0, float(seconds))
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{ms:03d}"


_TS_RE = re.compile(r"^(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?$")


def parse_ts(value: str) -> float:
    """Parse ``MM:SS``, ``H:MM:SS`` or ``HH:MM:SS,mmm`` into seconds."""
    match = _TS_RE.match(value.strip())
    if not match:
        raise ValueError(f"unrecognised timestamp: {value!r}")
    hours, minutes, secs, millis = match.groups()
    total = int(hours or 0) * 3600 + int(minutes) * 60 + int(secs)
    if millis:
        total += int(millis.ljust(3, "0")) / 1000
    return float(total)


def human_duration(seconds: float) -> str:
    """Human phrasing for a duration, e.g. ``1h 42m``."""
    seconds = max(0.0, float(seconds))
    hours, rem = divmod(int(round(seconds)), 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def sha256_file(path: str | Path, block_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(part.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


def write_atomic(path: str | Path, text: str) -> Path:
    """Write ``text`` to ``path`` via a temp file so partial writes never land."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    return path


def write_json(path: str | Path, payload: Any) -> Path:
    return write_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def slugify(text: str, max_length: int = 60) -> str:
    normalised = unicodedata.normalize("NFKD", text)
    ascii_text = normalised.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "-", ascii_text).strip("-")
    slug = slug[:max_length].strip("-")
    return slug or "untitled"


def dedupe_preserving_order(items: Iterable[str]) -> list[str]:
    """Drop near-duplicates (case/punctuation-insensitive), keeping first seen."""
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = re.sub(r"[^a-z0-9]+", " ", item.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def truncate(text: str, max_chars: int, suffix: str = " ...") -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - len(suffix))].rstrip() + suffix


# --- terminal output -------------------------------------------------------

_NO_COLOR = bool(os.environ.get("NO_COLOR")) or not sys.stderr.isatty()

_STYLES = {
    "dim": "\033[2m",
    "bold": "\033[1m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "cyan": "\033[36m",
}


def style(text: str, *names: str) -> str:
    if _NO_COLOR or not names:
        return text
    prefix = "".join(_STYLES.get(name, "") for name in names)
    return f"{prefix}{text}\033[0m"


def eprint(*args: Any, **kwargs: Any) -> None:
    """Status output goes to stderr so stdout stays pipeable."""
    print(*args, file=sys.stderr, **kwargs)


def step(message: str) -> None:
    eprint(style("==> ", "cyan", "bold") + message)


def info(message: str) -> None:
    eprint(style("    " + message, "dim"))


def warn(message: str) -> None:
    eprint(style("!!  " + message, "yellow"))


def error(message: str) -> None:
    eprint(style("xx  " + message, "red"))
