import pytest

from recap.render import render_markdown, render_plain, render_transcript_markdown
from recap.summarize import Summary, normalize_summary


@pytest.fixture
def summary() -> Summary:
    s = normalize_summary(
        {
            "title": "Quarterly Planning Review",
            "one_liner": "The team committed to shipping the pilot.",
            "executive_summary": ["Pilot ships 30 September.", "Hiring deferred."],
            "narrative": "First paragraph.\n\nSecond paragraph.",
            "themes": [{"title": "Budget", "bullets": ["40k approved"], "timestamp": "00:30"}],
            "decisions": [{"decision": "Ship in September", "rationale": "Customer commitment", "timestamp": "05:00"}],
            "action_items": [
                {"task": "Draft the rollout plan", "owner": "Dana", "due": "Friday", "timestamp": "12:00"},
                {"task": "Confirm quote | with tax", "owner": "", "due": "", "timestamp": ""},
            ],
            "open_questions": ["Who signs the contract?"],
            "risks": ["No slack in the schedule."],
            "next_steps": ["Reconvene Monday."],
        }
    )
    s.notes = {
        "quotes": [{"quote": "We should ship it.", "speaker": "Dana", "timestamp": "07:15"}],
        "facts": [{"fact": "Budget is 40k.", "timestamp": "01:00"}],
    }
    s.meta = {
        "source": "/tmp/quarterly-review.m4a",
        "duration": 720.0,
        "word_count": 1234,
        "created_at": "2026-08-30T10:00:00+00:00",
        "asr_backend": "faster-whisper",
        "asr_model": "small.en",
        "llm_backend": "ollama",
        "llm_model": "llama3.1:8b",
        "llm_base_url": "http://127.0.0.1:11434",
        "chunks": 8,
        "language": "en",
    }
    return s


def test_markdown_has_the_expected_sections(summary):
    md = render_markdown(summary)
    for heading in (
        "# Quarterly Planning Review",
        "## Executive summary",
        "## Decisions",
        "## Action items",
        "## Discussion",
        "## Open questions",
        "## Risks and concerns",
        "## Next steps",
    ):
        assert heading in md


def test_pipes_in_content_do_not_break_the_table(summary):
    md = render_markdown(summary)
    row = next(line for line in md.splitlines() if "Confirm quote" in line)
    assert "\\|" in row  # the pipe in the content is escaped
    assert row.replace("\\|", "").count("|") == 6  # 5 columns, pipes at both ends


def test_missing_owner_is_labelled_unassigned(summary):
    assert "unassigned" in render_markdown(summary)


def test_timestamps_are_rendered_as_code_spans(summary):
    assert "`05:00`" in render_markdown(summary)


def test_provenance_block_names_both_models(summary):
    md = render_markdown(summary)
    assert "faster-whisper / small.en" in md
    assert "ollama / llama3.1:8b" in md
    assert "How this summary was produced" in md


def test_notes_appendix_is_opt_in(summary):
    assert "Appendix" not in render_markdown(summary)
    with_notes = render_markdown(summary, include_notes=True)
    assert "Notable quotes" in with_notes
    assert "> We should ship it." in with_notes
    assert "Budget is 40k." in with_notes


def test_empty_summary_still_renders(summary):
    md = render_markdown(normalize_summary({}))
    assert md.startswith("# Meeting summary")


def test_plain_text_has_no_markdown_noise(summary):
    text = render_plain(summary)
    assert "##" not in text
    assert "| ---" not in text  # no markdown tables
    assert "1. [Dana, due Friday] Draft the rollout plan" in text
    assert "QUARTERLY PLANNING REVIEW" in text


def test_transcript_markdown_lists_timestamps(transcript):
    md = render_transcript_markdown(transcript)
    assert "`00:00`" in md
    assert "**Marcus:**" in md
    assert "12m 00s" in md


def test_narrative_paragraphs_stay_separate(summary):
    md = render_markdown(summary)
    assert "First paragraph.\n\nSecond paragraph." in md


def test_markdown_ends_with_a_single_newline(summary):
    md = render_markdown(summary)
    assert md.endswith("\n") and not md.endswith("\n\n")
