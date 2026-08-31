"""Map-reduce summarisation: per-chunk notes -> merged notes -> executive summary."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import prompts
from .chunking import Chunk, chunk_transcript
from .config import Config
from .llm import BaseClient, LLMError
from .transcript import Transcript
from .util import (
    dedupe_preserving_order,
    estimate_tokens,
    human_duration,
    read_json,
    sha256_text,
    truncate,
    write_json,
)

ProgressFn = Callable[[str], None]

LIST_FIELDS = {
    "topics": ("title", "detail", "timestamp"),
    "decisions": ("decision", "rationale", "timestamp"),
    "action_items": ("task", "owner", "due", "timestamp"),
    "open_questions": ("question", "timestamp"),
    "risks": ("risk", "timestamp"),
    "facts": ("fact", "timestamp"),
    "quotes": ("quote", "speaker", "timestamp"),
}
# The first key of each tuple is the identity of the item, used for de-duplication.
PRIMARY_KEY = {name: keys[0] for name, keys in LIST_FIELDS.items()}


class SummarizeError(RuntimeError):
    pass


@dataclass
class Summary:
    title: str = "Meeting summary"
    one_liner: str = ""
    executive_summary: list[str] = field(default_factory=list)
    narrative: str = ""
    themes: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, str]] = field(default_factory=list)
    action_items: list[dict[str, str]] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    next_steps: list[str] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)
    chunk_notes: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "one_liner": self.one_liner,
            "executive_summary": self.executive_summary,
            "narrative": self.narrative,
            "themes": self.themes,
            "decisions": self.decisions,
            "action_items": self.action_items,
            "open_questions": self.open_questions,
            "risks": self.risks,
            "next_steps": self.next_steps,
            "notes": self.notes,
            "chunk_notes": self.chunk_notes,
            "meta": self.meta,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Summary:
        summary = normalize_summary(data)
        summary.notes = data.get("notes") or {}
        summary.chunk_notes = data.get("chunk_notes") or []
        summary.meta = data.get("meta") or {}
        return summary


# --- normalisation ---------------------------------------------------------


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return "; ".join(_as_text(v) for v in value if _as_text(v))
    if isinstance(value, dict):
        return "; ".join(f"{k}: {_as_text(v)}" for k, v in value.items() if _as_text(v))
    return str(value)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, dict):
        value = list(value.values())
    if not isinstance(value, list):
        return [_as_text(value)]
    out = []
    for item in value:
        text = _as_text(item)
        if text:
            out.append(text)
    return dedupe_preserving_order(out)


def _as_dict_list(value: Any, keys: Sequence[str]) -> list[dict[str, str]]:
    """Coerce whatever the model returned into a list of dicts with ``keys``."""
    if value is None:
        return []
    if isinstance(value, dict):
        value = list(value.values())
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    primary = keys[0]
    out: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            record = {key: _as_text(item.get(key)) for key in keys}
            if not record[primary]:
                # Model used a different key name for the main field; take the
                # longest string value it did provide.
                candidates = [_as_text(v) for k, v in item.items() if k not in keys]
                candidates = [c for c in candidates if c]
                if candidates:
                    record[primary] = max(candidates, key=len)
        else:
            record = {key: "" for key in keys}
            record[primary] = _as_text(item)
        if record[primary]:
            out.append(record)
    return out


def normalize_notes(raw: Any) -> dict[str, Any]:
    """Force a map/fold reply into the canonical notes shape."""
    if not isinstance(raw, dict):
        raise SummarizeError(f"expected a JSON object of notes, got {type(raw).__name__}")
    notes: dict[str, Any] = {"excerpt_summary": _as_text(raw.get("excerpt_summary"))}
    for name, keys in LIST_FIELDS.items():
        notes[name] = _as_dict_list(raw.get(name), keys)
    return notes


def empty_notes() -> dict[str, Any]:
    return {"excerpt_summary": "", **{name: [] for name in LIST_FIELDS}}


def merge_notes(batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Deterministically concatenate notes, dropping exact/near-duplicate entries."""
    merged = empty_notes()
    summaries: list[str] = []
    seen: dict[str, set[str]] = {name: set() for name in LIST_FIELDS}
    for notes in batch:
        if notes.get("excerpt_summary"):
            summaries.append(notes["excerpt_summary"])
        for name in LIST_FIELDS:
            for item in notes.get(name, []):
                key = _identity(item[PRIMARY_KEY[name]])
                if not key or key in seen[name]:
                    continue
                seen[name].add(key)
                merged[name].append(item)
    merged["excerpt_summary"] = " ".join(summaries).strip()
    return merged


def _identity(text: str) -> str:
    return "".join(ch for ch in text.lower() if ch.isalnum() or ch == " ").strip()


def notes_tokens(notes: dict[str, Any]) -> int:
    return estimate_tokens(json.dumps(notes, ensure_ascii=False))


def normalize_summary(raw: Any) -> Summary:
    """Force a reduce reply into a Summary, tolerating loose model output."""
    if not isinstance(raw, dict):
        raise SummarizeError(f"expected a JSON summary object, got {type(raw).__name__}")
    themes: list[dict[str, Any]] = []
    theme_source = raw.get("themes")
    if isinstance(theme_source, dict):
        theme_source = [{"title": k, "bullets": v} for k, v in theme_source.items()]
    for item in theme_source or []:
        if isinstance(item, dict):
            title = _as_text(item.get("title") or item.get("theme") or item.get("name"))
            bullets = _as_str_list(item.get("bullets") or item.get("points") or item.get("detail"))
            timestamp = _as_text(item.get("timestamp"))
        else:
            title, bullets, timestamp = _as_text(item), [], ""
        if title or bullets:
            themes.append({"title": title or "Discussion", "bullets": bullets, "timestamp": timestamp})

    return Summary(
        title=_as_text(raw.get("title")) or "Meeting summary",
        one_liner=_as_text(raw.get("one_liner") or raw.get("tldr")),
        executive_summary=_as_str_list(raw.get("executive_summary") or raw.get("summary")),
        narrative=_as_text(raw.get("narrative")),
        themes=themes,
        decisions=_as_dict_list(raw.get("decisions"), LIST_FIELDS["decisions"]),
        action_items=_as_dict_list(raw.get("action_items") or raw.get("actions"), LIST_FIELDS["action_items"]),
        open_questions=_as_str_list(raw.get("open_questions")),
        risks=_as_str_list(raw.get("risks")),
        next_steps=_as_str_list(raw.get("next_steps")),
    )


# --- pipeline --------------------------------------------------------------


def summarize_transcript(
    transcript: Transcript,
    client: BaseClient,
    config: Config,
    cache_dir: Path | None = None,
    on_progress: ProgressFn | None = None,
) -> Summary:
    """Run map -> merge (-> fold) -> reduce over ``transcript``."""
    say = on_progress or (lambda _message: None)
    sconf = config.summary

    chunks = chunk_transcript(transcript, sconf.chunk_tokens, sconf.chunk_overlap_tokens)
    if not chunks:
        raise SummarizeError("transcript contains no usable text")

    chunk_notes: list[dict[str, Any]] = []
    for chunk in chunks:
        say(f"notes {chunk.index + 1}/{len(chunks)}  ({chunk.span()}, ~{chunk.tokens} tokens)")
        context = _context_for(chunk_notes)
        notes = _cached(
            cache_dir,
            "map",
            sha256_text(client.config.model, str(sconf.chunk_tokens), chunk.text, context or ""),
            lambda chunk=chunk, context=context: _map_chunk(chunk, len(chunks), client, config, context),
        )
        notes["_span"] = chunk.span()
        chunk_notes.append(notes)

    merged = merge_notes(chunk_notes)
    rounds = 0
    budget = max(600, int(client.config.num_ctx * 0.5))
    while notes_tokens(merged) > budget and len(chunk_notes) > 1 and rounds < 4:
        rounds += 1
        say(f"folding notes (pass {rounds}, ~{notes_tokens(merged)} tokens > {budget})")
        chunk_notes = _fold_round(chunk_notes, client, config, cache_dir, say)
        merged = merge_notes(chunk_notes)

    if notes_tokens(merged) > budget:
        say("notes still large after folding; trimming lowest-value entries")
        merged = _trim_notes(merged, budget)

    say("writing executive summary")
    summary = _reduce(merged, transcript, client, config)
    summary.notes = merged
    summary.chunk_notes = chunk_notes
    summary.meta = {
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "llm_backend": client.name,
        "llm_model": client.config.model,
        "llm_base_url": client.config.base_url,
        "asr_backend": transcript.asr_backend,
        "asr_model": transcript.asr_model,
        "source": transcript.source,
        "duration": transcript.duration,
        "duration_human": human_duration(transcript.duration),
        "word_count": transcript.word_count(),
        "chunks": len(chunks),
        "fold_rounds": rounds,
        "language": transcript.language,
        "speakers": transcript.speakers,
    }
    return summary


def _map_chunk(
    chunk: Chunk,
    total: int,
    client: BaseClient,
    config: Config,
    context: str | None,
) -> dict[str, Any]:
    messages = prompts.map_prompt(
        chunk.text, chunk.index, total, chunk.span(), config.summary, context
    )
    try:
        raw = client.chat_json(messages, retries=config.summary.max_map_retries)
    except LLMError as exc:
        # One bad chunk should not throw away an hour of transcription.
        return {
            **empty_notes(),
            "excerpt_summary": f"[notes unavailable for {chunk.span()}: {exc}]",
            "_error": str(exc),
        }
    return normalize_notes(raw)


def _context_for(previous: list[dict[str, Any]], max_chars: int = 600) -> str | None:
    if not previous:
        return None
    tail = previous[-1].get("excerpt_summary") or ""
    return truncate(tail, max_chars) or None


def _fold_round(
    batches: list[dict[str, Any]],
    client: BaseClient,
    config: Config,
    cache_dir: Path | None,
    say: ProgressFn,
) -> list[dict[str, Any]]:
    size = max(2, config.summary.fold_batch_size)
    folded: list[dict[str, Any]] = []
    groups = [batches[i : i + size] for i in range(0, len(batches), size)]
    for index, group in enumerate(groups):
        if len(group) == 1:
            folded.append(group[0])
            continue
        say(f"  fold {index + 1}/{len(groups)}")
        payload = json.dumps([_without_private(n) for n in group], ensure_ascii=False, indent=1)
        messages = prompts.fold_prompt(payload, config.summary)
        try:
            raw = client.chat_json(messages, retries=1)
            result = normalize_notes(raw)
        except (LLMError, SummarizeError):
            result = merge_notes(group)  # deterministic fallback
        result["_span"] = f"{group[0].get('_span', '')}..{group[-1].get('_span', '')}"
        folded.append(result)
    return folded


def _without_private(notes: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in notes.items() if not k.startswith("_")}


def _trim_notes(notes: dict[str, Any], budget_tokens: int) -> dict[str, Any]:
    """Last-resort shrink: drop colour (quotes, facts, topics) before substance."""
    trimmed = dict(notes)
    for name in ("quotes", "facts", "topics", "risks", "open_questions"):
        while notes_tokens(trimmed) > budget_tokens and trimmed.get(name):
            trimmed[name] = trimmed[name][:-1]
    if notes_tokens(trimmed) > budget_tokens:
        trimmed["excerpt_summary"] = truncate(trimmed.get("excerpt_summary", ""), 2000)
    return trimmed


def _reduce(
    notes: dict[str, Any],
    transcript: Transcript,
    client: BaseClient,
    config: Config,
) -> Summary:
    meta_lines = [
        f"Recording length: {human_duration(transcript.duration)}",
        f"Transcript words: {transcript.word_count():,}",
    ]
    if transcript.speakers:
        meta_lines.append(f"Speakers identified: {', '.join(transcript.speakers)}")
    if transcript.source:
        meta_lines.append(f"Source file: {Path(transcript.source).name}")
    messages = prompts.reduce_prompt(
        json.dumps(_without_private(notes), ensure_ascii=False, indent=1),
        "\n".join(meta_lines),
        config.summary,
    )
    raw = client.chat_json(messages, max_tokens=max(1024, client.config.max_output_tokens), retries=2)
    summary = normalize_summary(raw)
    if not summary.executive_summary and not summary.narrative:
        raise SummarizeError(
            "the model returned an empty summary. Try a larger model "
            "(e.g. --llm-model llama3.1:8b) or raise llm.num_ctx."
        )
    return summary


def _cached(cache_dir: Path | None, kind: str, key: str, produce: Callable[[], dict[str, Any]]) -> dict[str, Any]:
    """Memoise an LLM call on disk so an interrupted run resumes for free."""
    if cache_dir is None:
        return produce()
    path = Path(cache_dir) / f"{kind}-{key[:16]}.json"
    if path.exists():
        try:
            return read_json(path)
        except (ValueError, OSError):
            pass
    result = produce()
    if not result.get("_error"):
        with contextlib.suppress(OSError):
            write_json(path, result)
    return result
