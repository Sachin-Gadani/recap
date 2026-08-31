import json
import re

import pytest

from recap.config import Config
from recap.llm import build_client
from recap.summarize import (
    SummarizeError,
    Summary,
    empty_notes,
    merge_notes,
    normalize_notes,
    normalize_summary,
    notes_tokens,
    summarize_transcript,
)

# --- normalisation of sloppy model output ---------------------------------


def test_normalize_notes_fills_every_field():
    notes = normalize_notes({"excerpt_summary": "hi"})
    assert set(notes) == set(empty_notes())
    assert notes["decisions"] == []


def test_normalize_notes_accepts_bare_strings_in_lists():
    notes = normalize_notes({"decisions": ["Ship in September"]})
    assert notes["decisions"] == [{"decision": "Ship in September", "rationale": "", "timestamp": ""}]


def test_normalize_notes_recovers_from_renamed_keys():
    notes = normalize_notes({"action_items": [{"what": "Draft the plan", "owner": "Dana"}]})
    assert notes["action_items"][0]["task"] == "Draft the plan"
    assert notes["action_items"][0]["owner"] == "Dana"


def test_normalize_notes_flattens_nested_values():
    notes = normalize_notes({"facts": [{"fact": ["40k", "September"]}]})
    assert notes["facts"][0]["fact"] == "40k; September"


def test_normalize_notes_accepts_a_dict_instead_of_a_list():
    notes = normalize_notes({"risks": {"a": "Timeline is tight"}})
    assert notes["risks"][0]["risk"] == "Timeline is tight"


def test_normalize_notes_rejects_non_objects():
    with pytest.raises(SummarizeError):
        normalize_notes(["not", "an", "object"])


def test_normalize_summary_tolerates_alternate_key_names():
    summary = normalize_summary({"tldr": "One line", "summary": "A single bullet", "actions": []})
    assert summary.one_liner == "One line"
    assert summary.executive_summary == ["A single bullet"]


def test_normalize_summary_accepts_themes_as_a_mapping():
    summary = normalize_summary({"themes": {"Budget": ["40k approved"]}})
    assert summary.themes[0]["title"] == "Budget"
    assert summary.themes[0]["bullets"] == ["40k approved"]


def test_normalize_summary_defaults_the_title():
    assert normalize_summary({}).title == "Meeting summary"


# --- merging ---------------------------------------------------------------


def test_merge_notes_deduplicates_across_chunks():
    a = normalize_notes({"decisions": ["Ship in September"], "excerpt_summary": "first"})
    b = normalize_notes({"decisions": ["ship in september.", "Freeze scope"], "excerpt_summary": "second"})
    merged = merge_notes([a, b])
    assert [d["decision"] for d in merged["decisions"]] == ["Ship in September", "Freeze scope"]
    assert merged["excerpt_summary"] == "first second"


def test_merge_notes_keeps_the_first_timestamp_of_a_duplicate():
    a = normalize_notes({"decisions": [{"decision": "Ship it", "timestamp": "01:00"}]})
    b = normalize_notes({"decisions": [{"decision": "Ship it", "timestamp": "09:00"}]})
    assert merge_notes([a, b])["decisions"][0]["timestamp"] == "01:00"


def test_merge_of_nothing_is_empty():
    assert merge_notes([]) == empty_notes()


def test_notes_tokens_grows_with_content():
    assert notes_tokens(empty_notes()) < notes_tokens(normalize_notes({"facts": ["x" * 500]}))


# --- the map/reduce run ----------------------------------------------------


def _config(fake_llm, tmp_path, **summary_overrides) -> Config:
    config = Config()
    config.llm.base_url = fake_llm.base_url
    config.work_root = str(tmp_path)
    config.summary.chunk_tokens = 400
    config.summary.chunk_overlap_tokens = 60
    for key, value in summary_overrides.items():
        setattr(config.summary, key, value)
    return config


def test_summarize_transcript_end_to_end(transcript, fake_llm, tmp_path):
    config = _config(fake_llm, tmp_path)
    client = build_client(config.llm)
    summary = summarize_transcript(transcript, client, config)

    assert summary.title == "Quarterly Planning Review"
    assert len(summary.executive_summary) == 3
    assert summary.action_items[0]["owner"] == "Dana"
    assert summary.meta["chunks"] > 1
    assert summary.meta["llm_model"] == config.llm.model
    assert summary.meta["duration_human"] == "12m 00s"
    assert len(summary.chunk_notes) == summary.meta["chunks"]


def test_map_calls_carry_the_transcript_text(transcript, fake_llm, tmp_path):
    config = _config(fake_llm, tmp_path)
    summarize_transcript(transcript, build_client(config.llm), config)
    map_prompts = [
        r["messages"][-1]["content"] for r in fake_llm.requests if "TRANSCRIPT EXCERPT" in r["messages"][-1]["content"]
    ]
    assert map_prompts
    assert "forty thousand" in " ".join(map_prompts)
    assert all(re.search(r"\[\d+:\d\d\]", prompt) for prompt in map_prompts)


def test_focus_and_audience_reach_the_prompt(transcript, fake_llm, tmp_path):
    config = _config(fake_llm, tmp_path, focus="vendor risk", audience="the board")
    summarize_transcript(transcript, build_client(config.llm), config)
    joined = " ".join(r["messages"][-1]["content"] for r in fake_llm.requests)
    assert "vendor risk" in joined and "the board" in joined


def test_cache_makes_a_second_run_free(transcript, fake_llm, tmp_path):
    config = _config(fake_llm, tmp_path)
    cache = tmp_path / "cache"
    client = build_client(config.llm)
    summarize_transcript(transcript, client, config, cache_dir=cache)
    first = len(fake_llm.requests)
    summarize_transcript(transcript, client, config, cache_dir=cache)
    second = len(fake_llm.requests) - first
    # Only the (uncached) reduce call is repeated.
    assert second == 1
    assert list(cache.glob("map-*.json"))


def test_one_failing_chunk_does_not_lose_the_run(transcript, fake_llm, tmp_path, monkeypatch):
    monkeypatch.setattr("recap.llm.time.sleep", lambda _s: None)
    config = _config(fake_llm, tmp_path, max_map_retries=0)
    fake_llm.fail_next(3)
    summary = summarize_transcript(transcript, build_client(config.llm), config)
    assert summary.executive_summary  # the run still produced a report
    assert any("_error" in notes for notes in summary.chunk_notes)


def test_folding_kicks_in_when_notes_outgrow_the_context(transcript, fake_llm, tmp_path):
    config = _config(fake_llm, tmp_path, chunk_tokens=250, fold_batch_size=3)
    config.llm.num_ctx = 256  # forces the fold path
    summary = summarize_transcript(transcript, build_client(config.llm), config)
    assert summary.meta["fold_rounds"] >= 1
    assert summary.executive_summary


def test_empty_transcript_is_rejected(fake_llm, tmp_path):
    from recap.transcript import Transcript

    config = _config(fake_llm, tmp_path)
    with pytest.raises(SummarizeError):
        summarize_transcript(Transcript(segments=[]), build_client(config.llm), config)


def test_summary_json_roundtrip(transcript, fake_llm, tmp_path):
    config = _config(fake_llm, tmp_path)
    summary = summarize_transcript(transcript, build_client(config.llm), config)
    restored = Summary.from_dict(json.loads(json.dumps(summary.to_dict())))
    assert restored.title == summary.title
    assert restored.action_items == summary.action_items
    assert restored.meta["chunks"] == summary.meta["chunks"]
