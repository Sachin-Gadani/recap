"""Deterministic Markdown rendering of a Summary (the model returns JSON, not prose)."""

from __future__ import annotations

from typing import Any

from .summarize import Summary
from .transcript import Transcript
from .util import fmt_ts, human_duration


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip() or "-"


def _ts(value: str) -> str:
    value = (value or "").strip().strip("[]")
    return f"`{value}`" if value else "-"


def render_markdown(summary: Summary, include_notes: bool = False) -> str:
    meta = summary.meta or {}
    lines: list[str] = [f"# {summary.title}", ""]

    if summary.one_liner:
        lines += [f"**{summary.one_liner}**", ""]

    subtitle = _subtitle(meta)
    if subtitle:
        lines += [subtitle, ""]

    if summary.executive_summary:
        lines += ["## Executive summary", ""]
        lines += [f"- {bullet}" for bullet in summary.executive_summary]
        lines += [""]

    if summary.decisions:
        lines += ["## Decisions", ""]
        lines += ["| Decision | Why | When |", "| --- | --- | --- |"]
        for item in summary.decisions:
            lines.append(
                f"| {_escape_cell(item.get('decision', ''))} "
                f"| {_escape_cell(item.get('rationale', ''))} "
                f"| {_ts(item.get('timestamp', ''))} |"
            )
        lines += [""]

    if summary.action_items:
        lines += ["## Action items", ""]
        lines += ["| # | Action | Owner | Due | When |", "| --- | --- | --- | --- | --- |"]
        for index, item in enumerate(summary.action_items, start=1):
            owner = item.get("owner") or "unassigned"
            lines.append(
                f"| {index} | {_escape_cell(item.get('task', ''))} "
                f"| {_escape_cell(owner)} "
                f"| {_escape_cell(item.get('due', ''))} "
                f"| {_ts(item.get('timestamp', ''))} |"
            )
        lines += [""]

    if summary.themes:
        lines += ["## Discussion", ""]
        for theme in summary.themes:
            heading = theme.get("title", "Discussion")
            stamp = (theme.get("timestamp") or "").strip().strip("[]")
            lines.append(f"### {heading}" + (f" `{stamp}`" if stamp else ""))
            lines.append("")
            for bullet in theme.get("bullets", []):
                lines.append(f"- {bullet}")
            lines.append("")

    if summary.narrative:
        lines += ["## How the conversation went", ""]
        for paragraph in summary.narrative.split("\n\n"):
            if paragraph.strip():
                lines += [paragraph.strip(), ""]

    for heading, items in (
        ("Open questions", summary.open_questions),
        ("Risks and concerns", summary.risks),
        ("Next steps", summary.next_steps),
    ):
        if items:
            lines += [f"## {heading}", ""]
            lines += [f"- {item}" for item in items]
            lines += [""]

    if include_notes and summary.notes:
        lines += _render_notes_appendix(summary.notes)

    lines += _render_provenance(meta)
    return "\n".join(lines).rstrip() + "\n"


def _subtitle(meta: dict[str, Any]) -> str:
    bits: list[str] = []
    source = meta.get("source")
    if source:
        bits.append(f"`{source.split('/')[-1]}`")
    if meta.get("duration"):
        bits.append(human_duration(float(meta["duration"])))
    if meta.get("word_count"):
        bits.append(f"{int(meta['word_count']):,} words")
    created = meta.get("created_at", "")
    if created:
        bits.append(f"summarised {created[:10]}")
    return " · ".join(bits)


def _render_notes_appendix(notes: dict[str, Any]) -> list[str]:
    lines = ["## Appendix: extracted notes", ""]
    quotes = notes.get("quotes") or []
    if quotes:
        lines += ["### Notable quotes", ""]
        for item in quotes:
            speaker = item.get("speaker") or "Unattributed"
            stamp = (item.get("timestamp") or "").strip().strip("[]")
            suffix = f" `{stamp}`" if stamp else ""
            lines += [f"> {item.get('quote', '')}", ">", f"> - {speaker}{suffix}", ""]
    facts = notes.get("facts") or []
    if facts:
        lines += ["### Figures and specifics", ""]
        for item in facts:
            stamp = (item.get("timestamp") or "").strip().strip("[]")
            suffix = f" `{stamp}`" if stamp else ""
            lines.append(f"- {item.get('fact', '')}{suffix}")
        lines.append("")
    return lines


def _render_provenance(meta: dict[str, Any]) -> list[str]:
    if not meta:
        return []
    lines = ["---", "", "<details>", "<summary>How this summary was produced</summary>", ""]
    rows = [
        ("Source", meta.get("source")),
        ("Duration", human_duration(float(meta["duration"])) if meta.get("duration") else None),
        ("Transcription", f"{meta.get('asr_backend')} / {meta.get('asr_model')}"
            if meta.get("asr_model") else None),
        ("Summarisation", f"{meta.get('llm_backend')} / {meta.get('llm_model')} at {meta.get('llm_base_url')}"
            if meta.get("llm_model") else None),
        ("Transcript chunks", meta.get("chunks")),
        ("Fold rounds", meta.get("fold_rounds")),
        ("Detected language", meta.get("language")),
        ("Generated", meta.get("created_at")),
    ]
    for label, value in rows:
        if value not in (None, "", []):
            lines.append(f"- **{label}:** {value}")
    lines += [
        "",
        "Generated locally by [recap](https://github.com/Sachin-Gadani/recap). "
        "Timestamps refer to the source recording; treat every claim as a pointer "
        "to the transcript, not a substitute for it.",
        "",
        "</details>",
        "",
    ]
    return lines


def render_plain(summary: Summary) -> str:
    """Terminal-friendly rendering with no Markdown syntax noise."""
    out: list[str] = [summary.title.upper(), "=" * len(summary.title), ""]
    if summary.one_liner:
        out += [summary.one_liner, ""]
    if summary.executive_summary:
        out += ["EXECUTIVE SUMMARY", ""]
        out += [f"  * {bullet}" for bullet in summary.executive_summary]
        out += [""]
    if summary.decisions:
        out += ["DECISIONS", ""]
        for item in summary.decisions:
            stamp = (item.get("timestamp") or "").strip("[]")
            out.append(f"  * {item.get('decision', '')}" + (f"  ({stamp})" if stamp else ""))
        out += [""]
    if summary.action_items:
        out += ["ACTION ITEMS", ""]
        for index, item in enumerate(summary.action_items, start=1):
            owner = item.get("owner") or "unassigned"
            due = f", due {item['due']}" if item.get("due") else ""
            out.append(f"  {index}. [{owner}{due}] {item.get('task', '')}")
        out += [""]
    for heading, items in (
        ("OPEN QUESTIONS", summary.open_questions),
        ("RISKS", summary.risks),
        ("NEXT STEPS", summary.next_steps),
    ):
        if items:
            out += [heading, ""] + [f"  * {item}" for item in items] + [""]
    return "\n".join(out).rstrip() + "\n"


def render_transcript_markdown(transcript: Transcript, title: str = "Transcript") -> str:
    header = [
        f"# {title}",
        "",
        f"{human_duration(transcript.duration)} · {transcript.word_count():,} words"
        + (f" · {transcript.language}" if transcript.language else ""),
        "",
    ]
    body: list[str] = []
    for segment in transcript.segments:
        text = segment.text.strip()
        if not text:
            continue
        speaker = f"**{segment.speaker}:** " if segment.speaker else ""
        body.append(f"`{fmt_ts(segment.start)}` {speaker}{text}")
        body.append("")
    return "\n".join(header + body).rstrip() + "\n"
