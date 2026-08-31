"""Split a transcript into overlapping, timestamped chunks that fit the model context."""

from __future__ import annotations

from dataclasses import dataclass, field

from .transcript import Segment, Transcript
from .util import estimate_tokens, fmt_ts


@dataclass
class Chunk:
    index: int
    segments: list[Segment] = field(default_factory=list)
    overlap_count: int = 0  # leading segments repeated from the previous chunk

    @property
    def start(self) -> float:
        return self.segments[0].start if self.segments else 0.0

    @property
    def end(self) -> float:
        return self.segments[-1].end if self.segments else 0.0

    @property
    def text(self) -> str:
        return "\n".join(s.label() for s in self.segments)

    @property
    def tokens(self) -> int:
        return estimate_tokens(self.text)

    def span(self) -> str:
        return f"{fmt_ts(self.start)}-{fmt_ts(self.end)}"


def chunk_transcript(
    transcript: Transcript,
    target_tokens: int = 2400,
    overlap_tokens: int = 200,
) -> list[Chunk]:
    """Group segments into chunks of about ``target_tokens``, never splitting a segment.

    Consecutive chunks share a short tail-to-head overlap so a decision stated
    across a boundary is not lost.
    """
    if target_tokens < 200:
        raise ValueError("target_tokens must be at least 200")
    overlap_tokens = max(0, min(overlap_tokens, target_tokens // 3))

    segments = [s for s in transcript.segments if s.text.strip()]
    if not segments:
        return []

    chunks: list[Chunk] = []
    current: list[Segment] = []
    current_tokens = 0
    overlap_count = 0

    for segment in segments:
        cost = estimate_tokens(segment.label()) + 1
        if current and current_tokens + cost > target_tokens:
            chunks.append(Chunk(index=len(chunks), segments=current, overlap_count=overlap_count))
            carry = _tail_within(current, overlap_tokens)
            overlap_count = len(carry)
            current = list(carry)
            current_tokens = sum(estimate_tokens(s.label()) + 1 for s in current)
        current.append(segment)
        current_tokens += cost

    if current:
        # Do not emit a final chunk that is nothing but repeated overlap.
        if len(current) > overlap_count:
            chunks.append(Chunk(index=len(chunks), segments=current, overlap_count=overlap_count))
        elif chunks:
            chunks[-1].segments.extend(current[overlap_count:])
    return chunks


def _tail_within(segments: list[Segment], budget_tokens: int) -> list[Segment]:
    """The longest run of trailing segments costing at most ``budget_tokens``."""
    if budget_tokens <= 0:
        return []
    tail: list[Segment] = []
    used = 0
    for segment in reversed(segments):
        cost = estimate_tokens(segment.label()) + 1
        if used + cost > budget_tokens:
            break
        tail.insert(0, segment)
        used += cost
    # Never carry the entire chunk forward - that would loop forever.
    if len(tail) >= len(segments):
        tail = tail[1:]
    return tail


def describe_plan(chunks: list[Chunk]) -> str:
    if not chunks:
        return "no chunks"
    total = sum(c.tokens for c in chunks)
    largest = max(c.tokens for c in chunks)
    return (
        f"{len(chunks)} chunk(s), ~{total} tokens total, largest ~{largest} tokens, "
        f"covering {fmt_ts(chunks[0].start)}-{fmt_ts(chunks[-1].end)}"
    )
