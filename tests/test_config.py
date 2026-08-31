import pytest

from recap.config import (
    Config,
    ConfigError,
    check_privacy,
    config_template,
    is_local_endpoint,
    load_config,
)


def test_defaults_are_local_and_offline_friendly():
    config = Config()
    assert config.llm.backend == "ollama"
    assert is_local_endpoint(config.llm.base_url)
    assert config.allow_remote_llm is False


@pytest.mark.parametrize(
    "url,expected",
    [
        ("http://127.0.0.1:11434", True),
        ("http://localhost:1234/v1", True),
        ("http://[::1]:11434", True),
        ("http://192.168.1.50:11434", False),
        ("https://api.example.com", False),
    ],
)
def test_is_local_endpoint(url, expected):
    assert is_local_endpoint(url) is expected


def test_check_privacy_blocks_remote_by_default():
    config = Config()
    config.llm.base_url = "https://api.example.com"
    with pytest.raises(ConfigError, match="not this machine"):
        check_privacy(config)
    config.allow_remote_llm = True
    check_privacy(config)  # explicit opt-in is allowed


def test_toml_file_overrides_defaults(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        """
        [asr]
        model = "medium.en"
        beam_size = 1
        vad_filter = false

        [llm]
        model = "qwen2.5:7b"
        num_ctx = 16384
        """
    )
    config = load_config(path, env={})
    assert config.asr.model == "medium.en"
    assert config.asr.beam_size == 1
    assert config.asr.vad_filter is False
    assert config.llm.num_ctx == 16384


def test_env_beats_file_and_cli_beats_env(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[llm]\nmodel = "from-file"\n')
    config = load_config(path, env={"RECAP_LLM_MODEL": "from-env"})
    assert config.llm.model == "from-env"
    config = load_config(path, overrides={"llm": {"model": "from-cli"}}, env={"RECAP_LLM_MODEL": "from-env"})
    assert config.llm.model == "from-cli"


def test_none_overrides_are_ignored(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[llm]\nmodel = "from-file"\n')
    config = load_config(path, overrides={"llm": {"model": None}}, env={})
    assert config.llm.model == "from-file"


def test_unknown_keys_are_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text('[llm]\nmodle = "typo"\n')
    with pytest.raises(ConfigError, match="unknown option"):
        load_config(path, env={})


def test_unknown_section_is_rejected(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("[nope]\nx = 1\n")
    with pytest.raises(ConfigError, match="unknown section"):
        load_config(path, env={})


def test_missing_explicit_config_file_raises(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "absent.toml", env={})


def test_template_is_valid_toml_and_covers_every_option():
    import tomllib

    template = config_template()
    tomllib.loads(template)  # commented-out, so it must parse as empty
    for key in ("chunk_tokens", "whisper_cpp_model", "num_ctx", "audience"):
        assert key in template
