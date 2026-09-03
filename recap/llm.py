"""Clients for locally hosted LLMs (Ollama, or any OpenAI-compatible server)."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from .config import LLMConfig

Message = dict[str, str]


class LLMError(RuntimeError):
    """Any failure talking to the local model server."""


class LLMUnavailable(LLMError):
    """The server is not reachable, or the requested model is not installed."""


@dataclass
class Completion:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


def _post_json(url: str, payload: dict[str, Any], timeout: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=body, method="POST")
    request.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:800]
        raise LLMError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMUnavailable(f"cannot reach {url}: {exc.reason}") from exc
    except TimeoutError as exc:
        raise LLMError(f"{url} timed out after {timeout}s") from exc


def _get_json(url: str, timeout: int, headers: dict[str, str] | None = None) -> dict[str, Any]:
    request = urllib.request.Request(url, method="GET")
    for key, value in (headers or {}).items():
        request.add_header(key, value)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        raise LLMError(f"{url} returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise LLMUnavailable(f"cannot reach {url}: {exc.reason}") from exc


class BaseClient:
    """Shared retry/JSON behaviour for the concrete backends."""

    name = "base"

    def __init__(self, config: LLMConfig, api_key: str | None = None) -> None:
        self.config = config
        self.api_key = api_key
        self.base_url = config.base_url.rstrip("/")

    def with_model(self, model: str) -> BaseClient:
        """A sibling client for the same server, talking to a different model."""
        if model == self.config.model:
            return self
        return type(self)(replace(self.config, model=model), self.api_key)

    # -- to implement in subclasses ----------------------------------------
    def _chat(self, messages: Sequence[Message], json_mode: bool, max_tokens: int | None) -> Completion:
        raise NotImplementedError

    def list_models(self) -> list[str]:
        raise NotImplementedError

    # -- public API ---------------------------------------------------------
    def chat(
        self,
        messages: Sequence[Message],
        json_mode: bool = False,
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> Completion:
        last: Exception | None = None
        for attempt in range(retries + 1):
            try:
                return self._chat(messages, json_mode, max_tokens)
            except LLMUnavailable:
                raise
            except LLMError as exc:
                last = exc
                if attempt < retries:
                    time.sleep(2 ** attempt)
        assert last is not None
        raise last

    def chat_json(
        self,
        messages: Sequence[Message],
        max_tokens: int | None = None,
        retries: int = 2,
    ) -> dict[str, Any]:
        """Chat and parse the reply as a JSON object, retrying on malformed output."""
        last: Exception | None = None
        conversation = list(messages)
        for _attempt in range(retries + 1):
            completion = self.chat(conversation, json_mode=True, max_tokens=max_tokens, retries=1)
            try:
                return extract_json_object(completion.text)
            except ValueError as exc:
                last = exc
                conversation = list(messages) + [
                    {"role": "assistant", "content": completion.text[:2000]},
                    {
                        "role": "user",
                        "content": (
                            "That was not valid JSON. Reply again with a single JSON "
                            "object only - no prose, no markdown fences."
                        ),
                    },
                ]
        raise LLMError(f"model {self.config.model} did not return valid JSON: {last}")

    def ensure_ready(self) -> None:
        """Raise a helpful error if the server is down or the model is missing."""
        models = self.list_models()
        if not self._model_installed(models):
            listed = ", ".join(models[:20]) or "(none)"
            raise LLMUnavailable(
                f"model {self.config.model!r} is not available on {self.base_url}.\n"
                f"Installed models: {listed}\n"
                f"Fix: ollama pull {self.config.model}"
            )

    def _model_installed(self, models: list[str]) -> bool:
        wanted = self.config.model
        if wanted in models:
            return True
        # Ollama reports "llama3.1:8b"; accept a bare "llama3.1" as :latest.
        bare = wanted.split(":")[0]
        return any(m == wanted or m.split(":")[0] == bare for m in models)


class OllamaClient(BaseClient):
    name = "ollama"

    def _chat(self, messages: Sequence[Message], json_mode: bool, max_tokens: int | None) -> Completion:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                # Ollama defaults num_ctx low; long transcripts need this set.
                "num_ctx": self.config.num_ctx,
                "num_predict": max_tokens or self.config.max_output_tokens,
            },
        }
        if json_mode:
            payload["format"] = "json"
        data = _post_json(f"{self.base_url}/api/chat", payload, self.config.request_timeout)
        text = (data.get("message") or {}).get("content", "")
        if not text.strip():
            raise LLMError("ollama returned an empty response")
        return Completion(
            text=text,
            model=data.get("model", self.config.model),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
        )

    def list_models(self) -> list[str]:
        data = _get_json(f"{self.base_url}/api/tags", timeout=15)
        return sorted(m.get("name", "") for m in data.get("models", []) if m.get("name"))


class OpenAICompatClient(BaseClient):
    """Works with LM Studio, llama.cpp --server, vLLM, Jan, and friends."""

    name = "openai"

    def _headers(self) -> dict[str, str]:
        if self.api_key:
            return {"Authorization": f"Bearer {self.api_key}"}
        return {}

    def _chat(self, messages: Sequence[Message], json_mode: bool, max_tokens: int | None) -> Completion:
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": list(messages),
            "temperature": self.config.temperature,
            "max_tokens": max_tokens or self.config.max_output_tokens,
            "stream": False,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        data = _post_json(
            f"{self.base_url}/v1/chat/completions",
            payload,
            self.config.request_timeout,
            self._headers(),
        )
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"no choices in response: {json.dumps(data)[:400]}")
        text = (choices[0].get("message") or {}).get("content", "")
        if not text.strip():
            raise LLMError("server returned an empty response")
        usage = data.get("usage") or {}
        return Completion(
            text=text,
            model=data.get("model", self.config.model),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
        )

    def list_models(self) -> list[str]:
        data = _get_json(f"{self.base_url}/v1/models", timeout=15, headers=self._headers())
        return sorted(m.get("id", "") for m in data.get("data", []) if m.get("id"))

    def _model_installed(self, models: list[str]) -> bool:
        # Many local servers ignore the model field entirely and serve whatever
        # is loaded, so an empty or non-matching list is not fatal here.
        return True


def build_client(config: LLMConfig, api_key: str | None = None) -> BaseClient:
    backend = config.backend.strip().lower()
    if backend == "ollama":
        return OllamaClient(config, api_key)
    if backend in {"openai", "openai-compatible", "lmstudio", "llama.cpp"}:
        return OpenAICompatClient(config, api_key)
    raise LLMError(f"unknown llm backend {config.backend!r} (expected 'ollama' or 'openai')")


_FENCE_RE = re.compile(r"^\s*```(?:json|JSON)?\s*|\s*```\s*$")


def extract_json_object(text: str) -> dict[str, Any]:
    """Pull a JSON object out of a model reply that may be fenced or padded."""
    if not text or not text.strip():
        raise ValueError("empty response")
    stripped = _FENCE_RE.sub("", text.strip())
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        parsed = _scan_for_object(stripped)
    if isinstance(parsed, list):
        # Some models wrap the object in a single-element array.
        if len(parsed) == 1 and isinstance(parsed[0], dict):
            return parsed[0]
        raise ValueError("expected a JSON object, got an array")
    if not isinstance(parsed, dict):
        raise ValueError(f"expected a JSON object, got {type(parsed).__name__}")
    return parsed


def _scan_for_object(text: str) -> Any:
    """Find the first balanced ``{...}`` run, ignoring braces inside strings."""
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escaped = False
        for index in range(start, len(text)):
            char = text[index]
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : index + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
        start = text.find("{", start + 1)
    raise ValueError("no JSON object found in response")
