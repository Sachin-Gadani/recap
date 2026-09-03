import json

import pytest

from recap import cli
from recap.transcript import load_transcript


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    """Keep tests away from the developer's real config and cache directories."""
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.chdir(tmp_path)
    for var in list(dict(**__import__("os").environ)):
        if var.startswith("RECAP_"):
            monkeypatch.delenv(var, raising=False)


@pytest.fixture
def transcript_file(tmp_path, transcript):
    return str(transcript.save(tmp_path / "meeting.json"))


def _base_args(fake_llm):
    return ["--llm-url", fake_llm.base_url, "--llm-model", "llama3.1:8b", "--chunk-tokens", "400", "-q"]


def test_summarize_writes_markdown_and_json(tmp_path, fake_llm, transcript_file, capsys):
    out = tmp_path / "exec"
    code = cli.main(["summarize", transcript_file, "-o", str(out), "--formats", "md,json,txt"] + _base_args(fake_llm))
    assert code == 0
    md = (tmp_path / "exec.md").read_text()
    assert "# Quarterly Planning Review" in md
    assert "## Action items" in md
    payload = json.loads((tmp_path / "exec.json").read_text())
    assert payload["meta"]["llm_model"] == "llama3.1:8b"
    assert (tmp_path / "exec.txt").exists()
    # Written paths go to stdout so the command composes with other tools.
    assert "exec.md" in capsys.readouterr().out


def test_summarize_defaults_the_filename_to_the_title(tmp_path, fake_llm, transcript_file):
    assert cli.main(["summarize", transcript_file, "--formats", "md"] + _base_args(fake_llm)) == 0
    assert (tmp_path / "quarterly-planning-review.md").exists()


def test_summarize_accepts_an_srt_transcript(tmp_path, fake_llm, transcript):
    srt = tmp_path / "meeting.srt"
    srt.write_text(transcript.to_srt())
    assert cli.main(["summarize", str(srt), "-o", str(tmp_path / "o"), "--formats", "md"] + _base_args(fake_llm)) == 0
    assert (tmp_path / "o.md").exists()


def test_print_flag_emits_the_summary(tmp_path, fake_llm, transcript_file, capsys):
    cli.main(["summarize", transcript_file, "-o", str(tmp_path / "o"), "--formats", "md", "--print"] + _base_args(fake_llm))
    assert "EXECUTIVE SUMMARY" in capsys.readouterr().out


def test_notes_flag_adds_the_appendix(tmp_path, fake_llm, transcript_file):
    cli.main(["summarize", transcript_file, "-o", str(tmp_path / "o"), "--formats", "md", "--notes"] + _base_args(fake_llm))
    assert "Appendix" in (tmp_path / "o.md").read_text()


def test_focus_flag_reaches_the_model(tmp_path, fake_llm, transcript_file):
    cli.main(
        ["summarize", transcript_file, "-o", str(tmp_path / "o"), "--formats", "md", "--focus", "vendor risk"]
        + _base_args(fake_llm)
    )
    assert any("vendor risk" in r["messages"][-1]["content"] for r in fake_llm.requests)


def test_run_with_an_existing_transcript_skips_asr(tmp_path, fake_llm, transcript_file):
    code = cli.main(
        ["run", "--transcript", transcript_file, "-o", str(tmp_path / "o"), "--formats", "md,transcript.md"]
        + _base_args(fake_llm)
    )
    assert code == 0
    assert (tmp_path / "o.md").exists()
    assert (tmp_path / "o.transcript.md").exists()


def test_missing_audio_file_is_a_clean_error(tmp_path, fake_llm, capsys):
    assert cli.main(["run", str(tmp_path / "nope.m4a")] + _base_args(fake_llm)) == 1
    assert "not found" in capsys.readouterr().err


def test_unknown_format_is_a_clean_error(fake_llm, transcript_file, capsys):
    assert cli.main(["summarize", transcript_file, "--formats", "pdf"] + _base_args(fake_llm)) == 1
    assert "unknown format" in capsys.readouterr().err


def test_remote_llm_is_refused_without_the_opt_in(transcript_file, capsys):
    code = cli.main(["summarize", transcript_file, "--llm-url", "https://api.example.com", "-q"])
    assert code == 1
    err = capsys.readouterr().err
    assert "not this machine" in err and "--allow-remote-llm" in err


def test_missing_model_is_refused_before_any_work(fake_llm, transcript_file, capsys):
    code = cli.main(["summarize", transcript_file, "--llm-url", fake_llm.base_url, "--llm-model", "absent:70b", "-q"])
    assert code == 1
    assert "ollama pull absent:70b" in capsys.readouterr().err


def test_transcribe_subcommand_reports_a_missing_backend(tmp_path, capsys):
    audio = tmp_path / "a.wav"
    audio.write_bytes(b"RIFF____WAVEfmt ")
    code = cli.main(["transcribe", str(audio), "--asr-backend", "whisper.cpp", "-q"])
    assert code == 1
    assert "whisper" in capsys.readouterr().err.lower()


def test_doctor_json_lists_every_check(fake_llm, capsys):
    cli.main(["doctor", "--json", "--llm-url", fake_llm.base_url])
    checks = json.loads(capsys.readouterr().out)
    names = {check["name"] for check in checks}
    assert {"python", "ffmpeg", "privacy", "work dir"} <= names
    assert next(c for c in checks if c["name"] == "privacy")["status"] == "ok"


def test_doctor_fails_when_the_llm_is_unreachable(capsys):
    assert cli.main(["doctor", "--llm-url", "http://127.0.0.1:9"]) == 1
    assert "ollama serve" in capsys.readouterr().err


def test_models_lists_what_the_server_has(fake_llm, capsys):
    assert cli.main(["models", "--llm-url", fake_llm.base_url, "--llm-model", "llama3.1:8b"]) == 0
    assert "* llama3.1:8b" in capsys.readouterr().out


def test_config_init_then_show(tmp_path, capsys):
    path = tmp_path / "recap.toml"
    assert cli.main(["config", "--init", "-c", str(path)]) == 0
    assert path.exists()
    assert cli.main(["config", "--init", "-c", str(path)]) == 1  # refuses to clobber
    assert cli.main(["config", "-c", str(path)]) == 0
    assert "llm.model" in capsys.readouterr().out


def test_config_path_prints_the_default_location(capsys):
    assert cli.main(["config", "--path"]) == 0
    assert capsys.readouterr().out.strip().endswith("recap/config.toml")


def test_env_var_configures_the_endpoint(monkeypatch, tmp_path, fake_llm, transcript_file):
    monkeypatch.setenv("RECAP_LLM_BASE_URL", fake_llm.base_url)
    assert cli.main(["summarize", transcript_file, "-o", str(tmp_path / "o"), "--formats", "md", "-q"]) == 0


def test_transcript_exports_survive_a_roundtrip(tmp_path, fake_llm, transcript_file):
    out = tmp_path / "o"
    cli.main(
        ["run", "--transcript", transcript_file, "-o", str(out), "--formats", "srt,vtt,transcript.txt"]
        + _base_args(fake_llm)
    )
    assert load_transcript(tmp_path / "o.srt").segments
    assert load_transcript(tmp_path / "o.vtt").segments
    assert (tmp_path / "o.transcript.txt").read_text().strip()


def test_reduce_model_flag_splits_the_two_passes(tmp_path, fake_llm, transcript_file):
    code = cli.main(
        ["summarize", transcript_file, "-o", str(tmp_path / "o"), "--formats", "md",
         "--reduce-model", "qwen2.5:32b"]
        + _base_args(fake_llm)
    )
    assert code == 0
    assert "(extraction)" in (tmp_path / "o.md").read_text()


def test_a_missing_reduce_model_fails_before_transcription(fake_llm, transcript_file, capsys):
    code = cli.main(
        ["summarize", transcript_file, "--reduce-model", "absent:70b"] + _base_args(fake_llm)
    )
    assert code == 1
    assert "ollama pull absent:70b" in capsys.readouterr().err


# --- a transcript handed to a command that expects audio --------------------


@pytest.mark.parametrize("suffix", [".vtt", ".srt", ".json", ".txt"])
def test_run_accepts_a_transcript_instead_of_audio(tmp_path, fake_llm, transcript, suffix, capsys):
    """`recap run notes.vtt` is the obvious thing to type; it must not reach Whisper."""
    path = tmp_path / f"meeting{suffix}"
    body = {".vtt": transcript.to_vtt(), ".srt": transcript.to_srt(), ".txt": transcript.to_txt()}
    if suffix == ".json":
        transcript.save(path)
    else:
        path.write_text(body[suffix])

    code = cli.main(["run", str(path), "-o", str(tmp_path / "o"), "--formats", "md"] + _base_args(fake_llm))
    assert code == 0
    assert (tmp_path / "o.md").exists()


def test_run_says_why_it_skipped_transcription(tmp_path, fake_llm, transcript, capsys):
    path = tmp_path / "meeting.vtt"
    path.write_text(transcript.to_vtt())
    cli.main(["run", str(path), "-o", str(tmp_path / "o"), "--formats", "md",
              "--llm-url", fake_llm.base_url, "--llm-model", "llama3.1:8b", "--chunk-tokens", "400"])
    assert "is a transcript, not audio" in capsys.readouterr().err


def test_transcribe_refuses_a_transcript_with_a_usable_fix(tmp_path, capsys):
    path = tmp_path / "meeting.vtt"
    path.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:02.000\nhello\n")
    assert cli.main(["transcribe", str(path), "-q"]) == 1
    err = capsys.readouterr().err
    assert "is a transcript, not audio" in err
    assert "recap summarize" in err


def test_a_missing_file_is_reported_before_anything_else(tmp_path, capsys):
    assert cli.main(["run", str(tmp_path / "absent.m4a"), "-q"]) == 1
    assert "file not found" in capsys.readouterr().err
