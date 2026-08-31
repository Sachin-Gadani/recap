import pytest

from recap.chunking import chunk_transcript, describe_plan
from recap.transcript import Segment, Transcript
from recap.util import estimate_tokens


def test_every_segment_appears_at_least_once(transcript):
    chunks = chunk_transcript(transcript, target_tokens=400, overlap_tokens=60)
    seen = {id(s) for c in chunks for s in c.segments}
    assert len(seen) == len(transcript.segments)


def test_chunks_respect_the_token_budget(transcript):
    chunks = chunk_transcript(transcript, target_tokens=400, overlap_tokens=60)
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.tokens <= 460  # budget plus the final segment that tipped it over


def test_consecutive_chunks_overlap(transcript):
    chunks = chunk_transcript(transcript, target_tokens=400, overlap_tokens=80)
    for previous, current in zip(chunks, chunks[1:], strict=False):
        assert current.overlap_count > 0
        assert current.segments[0].start <= previous.end


def test_zero_overlap_is_honoured(transcript):
    chunks = chunk_transcript(transcript, target_tokens=400, overlap_tokens=0)
    assert all(c.overlap_count == 0 for c in chunks)
    starts = [s.start for c in chunks for s in c.segments]
    assert len(starts) == len(set(starts))


def test_short_transcript_is_one_chunk(transcript):
    chunks = chunk_transcript(transcript, target_tokens=100_000)
    assert len(chunks) == 1
    assert chunks[0].overlap_count == 0


def test_empty_transcript_yields_no_chunks():
    assert chunk_transcript(Transcript(segments=[])) == []


def test_blank_segments_are_dropped():
    t = Transcript(segments=[Segment(0, 1, "  "), Segment(1, 2, "real text here")])
    chunks = chunk_transcript(t)
    assert len(chunks[0].segments) == 1


def test_a_single_oversized_segment_still_produces_a_chunk():
    huge = "word " * 5000
    t = Transcript(segments=[Segment(0, 600, huge)])
    chunks = chunk_transcript(t, target_tokens=500)
    assert len(chunks) == 1
    assert chunks[0].tokens > 500  # we never split mid-segment


def test_no_chunk_is_pure_overlap(transcript):
    chunks = chunk_transcript(transcript, target_tokens=250, overlap_tokens=80)
    for chunk in chunks:
        assert len(chunk.segments) > chunk.overlap_count


def test_target_tokens_floor_is_enforced(transcript):
    with pytest.raises(ValueError):
        chunk_transcript(transcript, target_tokens=10)


def test_chunk_text_is_timestamped(transcript):
    chunk = chunk_transcript(transcript, target_tokens=400)[0]
    assert chunk.text.startswith("[00:00]")
    assert estimate_tokens(chunk.text) == chunk.tokens


def test_describe_plan_mentions_counts(transcript):
    plan = describe_plan(chunk_transcript(transcript, target_tokens=400))
    assert "chunk(s)" in plan and "tokens" in plan
