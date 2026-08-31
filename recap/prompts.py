"""Prompt text for the map / fold / reduce passes.

The pipeline asks the model for JSON at every stage and renders the Markdown
itself, so the report structure is guaranteed rather than hoped for.
"""

from __future__ import annotations

from .config import SummaryConfig

GROUND_RULES = """\
Rules you must follow:
- Use only what the transcript says. Never invent names, numbers, dates or commitments.
- The transcript is machine-generated and may contain misheard words. If something is
  unclear, say so plainly rather than guessing.
- Copy timestamps verbatim from the [MM:SS] or [H:MM:SS] markers so every claim can be
  checked against the recording.
- Prefer the speaker's own concrete wording over vague corporate paraphrase.
- Leave an array empty rather than padding it with filler."""

MAP_SCHEMA = """\
{
  "excerpt_summary": "2-4 sentences on what happens in this excerpt",
  "topics":         [{"title": "", "detail": "", "timestamp": ""}],
  "decisions":      [{"decision": "", "rationale": "", "timestamp": ""}],
  "action_items":   [{"task": "", "owner": "", "due": "", "timestamp": ""}],
  "open_questions": [{"question": "", "timestamp": ""}],
  "risks":          [{"risk": "", "timestamp": ""}],
  "facts":          [{"fact": "", "timestamp": ""}],
  "quotes":         [{"quote": "", "speaker": "", "timestamp": ""}]
}"""

REDUCE_SCHEMA = """\
{
  "title": "short descriptive title for the meeting",
  "one_liner": "a single sentence capturing the outcome",
  "executive_summary": ["3 to 6 bullets, each a complete, specific sentence"],
  "narrative": "2-4 short paragraphs of prose covering how the discussion went",
  "themes":         [{"title": "", "bullets": [""], "timestamp": ""}],
  "decisions":      [{"decision": "", "rationale": "", "timestamp": ""}],
  "action_items":   [{"task": "", "owner": "", "due": "", "timestamp": ""}],
  "open_questions": [""],
  "risks":          [""],
  "next_steps":     [""]
}"""

MAP_SYSTEM = """\
You are a meticulous meeting analyst. You read one excerpt of a transcript at a time
and extract structured notes from it. You reply with a single JSON object and nothing
else - no prose, no markdown fences."""

REDUCE_SYSTEM = """\
You are an experienced chief of staff writing the executive summary of a meeting for
someone who was not there. You are given structured notes extracted from the whole
recording, in order. You reply with a single JSON object and nothing else - no prose,
no markdown fences."""

FOLD_SYSTEM = """\
You merge structured meeting notes. You reply with a single JSON object in exactly the
same schema as the input and nothing else - no prose, no markdown fences."""


def _preferences(config: SummaryConfig) -> str:
    lines = [f"- Write in {config.language}.", f"- The reader is {config.audience}."]
    if config.focus:
        lines.append(f"- Pay particular attention to: {config.focus}.")
    if config.extra_instructions:
        lines.append(f"- Additional instructions: {config.extra_instructions}")
    return "\n".join(lines)


def map_prompt(
    chunk_text: str,
    index: int,
    total: int,
    span: str,
    config: SummaryConfig,
    context: str | None = None,
) -> list[dict[str, str]]:
    """Messages for the per-chunk extraction pass."""
    preamble = f"Excerpt {index + 1} of {total}, covering {span} of the recording."
    if context:
        preamble += f"\nWhat came before (for context only - do not re-extract it):\n{context}"
    user = f"""\
{preamble}

{GROUND_RULES}
{_preferences(config)}

Return exactly this JSON shape:
{MAP_SCHEMA}

TRANSCRIPT EXCERPT
------------------
{chunk_text}
------------------

JSON notes for this excerpt only:"""
    return [
        {"role": "system", "content": MAP_SYSTEM},
        {"role": "user", "content": user},
    ]


def fold_prompt(notes_json: str, config: SummaryConfig) -> list[dict[str, str]]:
    """Messages for merging a batch of chunk notes into one set of notes."""
    user = f"""\
Merge the following sets of notes, which come from consecutive excerpts of one
recording, into a single set of notes in the same schema.

{GROUND_RULES}
- Combine duplicate or restated items into one entry, keeping the earliest timestamp.
- Keep every distinct decision, action item and open question. Do not drop detail to be brief.
- Rewrite `excerpt_summary` as a combined summary of the whole span.

Schema:
{MAP_SCHEMA}

NOTES TO MERGE
--------------
{notes_json}
--------------

Merged JSON:"""
    return [
        {"role": "system", "content": FOLD_SYSTEM},
        {"role": "user", "content": user},
    ]


def reduce_prompt(notes_json: str, meta: str, config: SummaryConfig) -> list[dict[str, str]]:
    """Messages for the final executive-summary pass."""
    user = f"""\
Write the executive summary of this recording.

{meta}

{GROUND_RULES}
{_preferences(config)}
- The executive summary bullets must be specific enough to be useful without the
  recording: name the decision, the number, the owner, the deadline.
- Group the discussion into 3-7 themes in the order they mattered, not the order
  they were mentioned.
- Every action item needs an owner. If nobody was named, use "unassigned".
- If the notes disagree with themselves, say which reading the transcript supports.

Return exactly this JSON shape:
{REDUCE_SCHEMA}

STRUCTURED NOTES FROM THE RECORDING
-----------------------------------
{notes_json}
-----------------------------------

Executive summary JSON:"""
    return [
        {"role": "system", "content": REDUCE_SYSTEM},
        {"role": "user", "content": user},
    ]
