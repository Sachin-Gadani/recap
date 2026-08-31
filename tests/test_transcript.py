import json

import pytest

from recap.transcript import Segment, Transcript, load_transcript, merge_short_segments


def test_labelled_text_carries_timestamps_and_speakers(transcript):
    line = transcript.segments[0].label()
    assert line.startswith("[00:00]")
    assert "Marcus:" in line


def test_json_roundtrip(tmp_path, transcript):
    path = transcript.save(tmp_path / "t.json")
    loaded = load_transcript(path)
    assert loaded.word_count() == transcript.word_count()
    assert loaded.language == "en"
    assert loaded.segments[3].speaker == transcript.segments[3].speaker


def test_srt_export_and_reimport(tmp_path, transcript):
    path = tmp_path / "t.srt"
    path.write_text(transcript.to_srt())
    loaded = load_transcript(path)
    assert len(loaded.segments) == len(transcript.segments)
    assert loaded.segments[0].start == pytest.approx(transcript.segments[0].start, abs=0.01)
    assert loaded.segments[0].speaker == "Marcus"


def test_vtt_export_and_reimport(tmp_path, transcript):
    path = tmp_path / "t.vtt"
    path.write_text(transcript.to_vtt())
    loaded = load_transcript(path)
    assert len(loaded.segments) == len(transcript.segments)


def test_plain_text_with_timestamps(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("[00:10] Ana: we start now\n[01:00] Ben: and we finish\n")
    loaded = load_transcript(path)
    assert [s.speaker for s in loaded.segments] == ["Ana", "Ben"]
    assert loaded.segments[1].start == 60.0


def test_plain_text_without_timestamps_gets_synthetic_timing(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("first line of the meeting\n\nsecond line of the meeting\n")
    loaded = load_transcript(path)
    assert len(loaded.segments) == 2
    assert loaded.segments[1].start > loaded.segments[0].start


def test_whisper_cpp_json_is_understood(tmp_path):
    payload = {
        "result": {"language": "en"},
        "transcription": [
            {"offsets": {"from": 0, "to": 2500}, "text": " Hello there."},
            {"offsets": {"from": 2500, "to": 5000}, "text": " Second cue."},
        ],
    }
    path = tmp_path / "w.json"
    path.write_text(json.dumps(payload))
    loaded = load_transcript(path)
    assert loaded.language == "en"
    assert loaded.segments[1].start == 2.5
    assert loaded.segments[0].text == "Hello there."


def test_empty_transcript_file_raises(tmp_path):
    path = tmp_path / "t.txt"
    path.write_text("   \n\n")
    with pytest.raises(ValueError):
        load_transcript(path)


def test_merge_short_segments_glues_fragments():
    segments = [
        Segment(0, 1, "So", "Ana"),
        Segment(1, 2, "the budget is fixed.", "Ana"),
        Segment(2, 3, "Understood, that settles it for now.", "Ben"),
    ]
    merged = merge_short_segments(segments)
    assert len(merged) == 2
    assert merged[0].text == "So the budget is fixed."
    assert merged[0].end == 2


def test_merge_does_not_cross_speakers():
    segments = [Segment(0, 1, "So", "Ana"), Segment(1, 2, "yes", "Ben")]
    assert len(merge_short_segments(segments)) == 2


def test_duration_defaults_to_last_segment():
    t = Transcript(segments=[Segment(0, 5, "hi"), Segment(5, 9, "bye")])
    assert t.duration == 9
