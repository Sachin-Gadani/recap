import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fake_ollama import FakeOllama  # noqa: E402

from recap.config import Config  # noqa: E402
from recap.transcript import Segment, Transcript  # noqa: E402

SPEECH = [
    "Right, let us start with the budget for the pilot programme this quarter.",
    "Finance approved forty thousand, which is ten less than we asked for.",
    "That means we cannot fund the second engineer until January at the earliest.",
    "Dana, can you take the rollout plan and have a draft by Friday afternoon?",
    "Yes, I will circulate it Thursday night so people can read before the call.",
    "The vendor quote still has not landed and that is the piece I worry about.",
    "If it slips past the fifteenth we lose the September ship date entirely.",
    "Let us agree we ship the pilot in September and hold the scope where it is.",
    "Agreed, and anything else goes on the list for the following release.",
    "One open question is who signs the vendor contract now that Priya has left.",
]


@pytest.fixture
def transcript() -> Transcript:
    segments = []
    for index in range(60):
        start = index * 12.0
        speaker = "Dana" if index % 2 else "Marcus"
        segments.append(
            Segment(start=start, end=start + 11.5, text=SPEECH[index % len(SPEECH)], speaker=speaker)
        )
    return Transcript(
        segments=segments,
        language="en",
        duration=720.0,
        source="/tmp/quarterly-review.m4a",
        asr_model="small.en",
        asr_backend="faster-whisper",
    )


@pytest.fixture
def fake_llm():
    # Two models installed, so tests can exercise the split extraction/synthesis path.
    with FakeOllama(models=("llama3.1:8b", "qwen2.5:32b")) as server:
        yield server


@pytest.fixture
def config(fake_llm, tmp_path) -> Config:
    cfg = Config()
    cfg.llm.base_url = fake_llm.base_url
    cfg.llm.model = "llama3.1:8b"
    cfg.work_root = str(tmp_path / "work")
    cfg.summary.chunk_tokens = 400
    cfg.summary.chunk_overlap_tokens = 60
    return cfg
