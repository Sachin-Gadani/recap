import pytest

from recap.config import LLMConfig
from recap.llm import (
    LLMError,
    LLMUnavailable,
    OllamaClient,
    build_client,
    extract_json_object,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"a": 1}', {"a": 1}),
        ('```json\n{"a": 1}\n```', {"a": 1}),
        ('```\n{"a": 1}\n```', {"a": 1}),
        ('Here you go:\n{"a": 1}\nHope that helps!', {"a": 1}),
        ('[{"a": 1}]', {"a": 1}),
        ('{"a": "a } brace in a string"}', {"a": "a } brace in a string"}),
        ('{"a": "an escaped \\" quote"}', {"a": 'an escaped " quote'}),
        ('garbage {not json} then {"a": 1}', {"a": 1}),
    ],
)
def test_extract_json_object(raw, expected):
    assert extract_json_object(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "no json at all", "[1, 2, 3]", '"a string"'])
def test_extract_json_object_rejects_non_objects(raw):
    with pytest.raises(ValueError):
        extract_json_object(raw)


def _client(fake_llm) -> OllamaClient:
    config = LLMConfig(base_url=fake_llm.base_url, model="llama3.1:8b")
    return build_client(config)


def test_list_models_and_readiness(fake_llm):
    client = _client(fake_llm)
    assert "llama3.1:8b" in client.list_models()
    client.ensure_ready()


def test_bare_model_name_matches_a_tagged_install(fake_llm):
    config = LLMConfig(base_url=fake_llm.base_url, model="llama3.1")
    build_client(config).ensure_ready()


def test_missing_model_explains_the_fix(fake_llm):
    config = LLMConfig(base_url=fake_llm.base_url, model="nonexistent:70b")
    with pytest.raises(LLMUnavailable, match="ollama pull nonexistent:70b"):
        build_client(config).ensure_ready()


def test_unreachable_server_is_reported_clearly():
    config = LLMConfig(base_url="http://127.0.0.1:9", model="x")
    with pytest.raises(LLMUnavailable, match="cannot reach"):
        build_client(config).list_models()


def test_chat_sends_num_ctx_so_long_transcripts_are_not_truncated(fake_llm):
    config = LLMConfig(base_url=fake_llm.base_url, model="llama3.1:8b", num_ctx=12345)
    build_client(config).chat([{"role": "user", "content": "hi"}])
    assert fake_llm.requests[-1]["options"]["num_ctx"] == 12345
    assert fake_llm.requests[-1]["stream"] is False


def test_json_mode_sets_the_ollama_format_flag(fake_llm):
    _client(fake_llm).chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert fake_llm.requests[-1]["format"] == "json"


def test_transient_server_errors_are_retried(fake_llm, monkeypatch):
    monkeypatch.setattr("recap.llm.time.sleep", lambda _s: None)
    fake_llm.fail_next(2)
    completion = _client(fake_llm).chat([{"role": "user", "content": "hi"}], retries=2)
    assert completion.text


def test_retries_are_bounded(fake_llm, monkeypatch):
    monkeypatch.setattr("recap.llm.time.sleep", lambda _s: None)
    fake_llm.fail_next(5)
    with pytest.raises(LLMError):
        _client(fake_llm).chat([{"role": "user", "content": "hi"}], retries=1)


def test_chat_json_reprompts_after_malformed_output(fake_llm):
    fake_llm.send_bad_json_once()
    result = _client(fake_llm).chat_json([{"role": "user", "content": "notes please"}])
    assert "excerpt_summary" in result
    # The repair turn shows the model its own bad answer.
    assert any("not valid JSON" in m["content"] for m in fake_llm.requests[-1]["messages"])


def test_unknown_backend_is_rejected():
    with pytest.raises(LLMError, match="unknown llm backend"):
        build_client(LLMConfig(backend="telepathy"))


def test_openai_backend_targets_the_v1_path(fake_llm):
    config = LLMConfig(backend="openai", base_url=fake_llm.base_url, model="local")
    client = build_client(config)
    assert client.name == "openai"
    # The fake server only speaks the ollama API, so this must fail loudly, not silently.
    with pytest.raises(LLMError):
        client.chat([{"role": "user", "content": "hi"}], retries=0)
