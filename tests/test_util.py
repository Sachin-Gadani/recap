import pytest

from recap.util import (
    dedupe_preserving_order,
    estimate_tokens,
    fmt_ts,
    fmt_ts_ms,
    human_duration,
    parse_ts,
    slugify,
    truncate,
    write_atomic,
)


@pytest.mark.parametrize(
    "seconds,expected",
    [(0, "00:00"), (61, "01:01"), (3599, "59:59"), (3600, "1:00:00"), (7325, "2:02:05")],
)
def test_fmt_ts(seconds, expected):
    assert fmt_ts(seconds) == expected


def test_fmt_ts_ms_is_srt_shaped():
    assert fmt_ts_ms(3661.5) == "01:01:01,500"
    assert fmt_ts_ms(3661.5, ".") == "01:01:01.500"


@pytest.mark.parametrize("text", ["00:00", "12:34", "1:02:03", "01:02:03,250"])
def test_parse_ts_roundtrips(text):
    assert parse_ts(text) >= 0


def test_parse_ts_rejects_garbage():
    with pytest.raises(ValueError):
        parse_ts("banana")


def test_estimate_tokens_scales_with_length():
    assert estimate_tokens("") == 0
    assert estimate_tokens("a" * 380) > estimate_tokens("a" * 38)


def test_human_duration():
    assert human_duration(45) == "45s"
    assert human_duration(125) == "2m 05s"
    assert human_duration(3720) == "1h 02m"


def test_slugify():
    assert slugify("Q3 Planning: Budget & Hiring!") == "q3-planning-budget-hiring"
    assert slugify("///") == "untitled"


def test_dedupe_ignores_case_and_punctuation():
    assert dedupe_preserving_order(["Ship it.", "ship it", "Hold scope"]) == ["Ship it.", "Hold scope"]


def test_truncate():
    assert truncate("hello world", 50) == "hello world"
    assert truncate("hello world", 8).endswith("...")


def test_write_atomic_leaves_no_temp_file(tmp_path):
    target = tmp_path / "nested" / "out.md"
    write_atomic(target, "content")
    assert target.read_text() == "content"
    assert list(tmp_path.rglob("*.tmp")) == []
