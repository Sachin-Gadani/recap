"""Configuration: dataclasses, TOML file loading, env overrides, privacy checks."""

from __future__ import annotations

import ipaddress
import os
import tomllib
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

APP_NAME = "recap"


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / APP_NAME / "config.toml"


def default_work_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME")
    root = Path(base) if base else Path.home() / ".cache"
    return root / APP_NAME


@dataclass
class ASRConfig:
    """Speech-to-text settings. Everything runs on this machine."""

    backend: str = "faster-whisper"  # faster-whisper | whisper.cpp
    model: str = "small.en"
    device: str = "auto"  # auto | cpu | cuda
    compute_type: str = "auto"  # auto | int8 | int8_float16 | float16 | float32
    language: str | None = None  # None = autodetect
    beam_size: int = 5
    vad_filter: bool = True
    initial_prompt: str | None = None
    # whisper.cpp backend only
    whisper_cpp_bin: str = "whisper-cli"
    whisper_cpp_model: str | None = None
    whisper_cpp_threads: int = 0  # 0 = let whisper.cpp decide


@dataclass
class LLMConfig:
    """Local LLM settings. Defaults point at Ollama on loopback."""

    backend: str = "ollama"  # ollama | openai (any OpenAI-compatible local server)
    base_url: str = "http://127.0.0.1:11434"
    model: str = "llama3.1:8b"
    temperature: float = 0.2
    num_ctx: int = 8192
    max_output_tokens: int = 2048
    request_timeout: int = 600
    api_key_env: str = "RECAP_LLM_API_KEY"  # only used by the openai backend


@dataclass
class SummaryConfig:
    """How the transcript is carved up and what the summary should emphasise."""

    chunk_tokens: int = 2400
    chunk_overlap_tokens: int = 200
    fold_batch_size: int = 6
    audience: str = "an executive who did not attend"
    focus: str | None = None
    language: str = "English"
    extra_instructions: str | None = None
    include_transcript: bool = True
    max_map_retries: int = 2


@dataclass
class Config:
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    summary: SummaryConfig = field(default_factory=SummaryConfig)
    work_root: str = ""
    allow_remote_llm: bool = False

    def __post_init__(self) -> None:
        if not self.work_root:
            self.work_root = str(default_work_root())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ConfigError(Exception):
    """Raised for malformed config files or unknown keys."""


_SECTIONS = {"asr": ASRConfig, "llm": LLMConfig, "summary": SummaryConfig}


def _coerce(value: Any, declared: Any, where: str) -> Any:
    """Coerce a TOML/env scalar to the type declared on the dataclass field."""
    if value is None:
        return None
    text = str(declared)
    if "int" in text and not isinstance(value, bool):
        return int(value)
    if "float" in text:
        return float(value)
    if "bool" in text:
        if isinstance(value, bool):
            return value
        lowered = str(value).strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
        raise ConfigError(f"{where}: expected a boolean, got {value!r}")
    return str(value)


def _apply(section: Any, values: dict[str, Any], where: str) -> None:
    declared = {f.name: f.type for f in fields(section)}
    for key, value in values.items():
        if key not in declared:
            known = ", ".join(sorted(declared))
            raise ConfigError(f"unknown option {where}.{key!r}. Known options: {known}")
        setattr(section, key, _coerce(value, declared[key], f"{where}.{key}"))


def load_config(
    path: str | Path | None = None,
    overrides: dict[str, dict[str, Any]] | None = None,
    env: dict[str, str] | None = None,
) -> Config:
    """Build a Config from defaults <- TOML file <- environment <- CLI overrides."""
    config = Config()

    resolved = Path(path) if path else default_config_path()
    if path and not resolved.exists():
        raise ConfigError(f"config file not found: {resolved}")
    if resolved.exists():
        try:
            data = tomllib.loads(resolved.read_text(encoding="utf-8"))
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{resolved}: {exc}") from exc
        for name, values in data.items():
            if name in _SECTIONS:
                if not isinstance(values, dict):
                    raise ConfigError(f"{resolved}: [{name}] must be a table")
                _apply(getattr(config, name), values, name)
            elif name in {"work_root", "allow_remote_llm"}:
                setattr(config, name, _coerce(values, str(type(getattr(config, name))), name))
            else:
                raise ConfigError(f"{resolved}: unknown section [{name}]")

    _apply_env(config, env if env is not None else os.environ)

    for name, values in (overrides or {}).items():
        if name in _SECTIONS:
            _apply(getattr(config, name), {k: v for k, v in values.items() if v is not None}, name)
        else:
            for key, value in values.items():
                if value is not None:
                    setattr(config, key, value)
    return config


def _apply_env(config: Config, env: dict[str, str]) -> None:
    """Read RECAP_<SECTION>_<KEY> variables, e.g. RECAP_LLM_MODEL."""
    for section_name, section_type in _SECTIONS.items():
        section = getattr(config, section_name)
        for f in fields(section_type):
            var = f"RECAP_{section_name.upper()}_{f.name.upper()}"
            if var in env and env[var] != "":
                setattr(section, f.name, _coerce(env[var], f.type, var))
    if env.get("RECAP_WORK_ROOT"):
        config.work_root = env["RECAP_WORK_ROOT"]


LOOPBACK_HOSTNAMES = {"localhost", "localhost.localdomain", "ip6-localhost"}


def is_local_endpoint(url: str) -> bool:
    """True when ``url`` points at this machine (loopback address or a unix socket)."""
    parsed = urlparse(url)
    if parsed.scheme in {"unix", "file"}:
        return True
    host = (parsed.hostname or "").strip("[]")
    if not host:
        return False
    if host.lower() in LOOPBACK_HOSTNAMES:
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_privacy(config: Config) -> None:
    """Refuse to ship transcripts off-box unless the user explicitly opted in."""
    if is_local_endpoint(config.llm.base_url) or config.allow_remote_llm:
        return
    raise ConfigError(
        f"llm.base_url points at {config.llm.base_url!r}, which is not this machine.\n"
        "recap keeps recordings local by default. Pass --allow-remote-llm (or set "
        "allow_remote_llm = true) if you really intend to send transcript text there."
    )


def config_template() -> str:
    """A commented TOML file showing every option at its default value."""
    lines = [
        "# recap configuration",
        f"# Default location: {default_config_path()}",
        "# Every value below is the built-in default; delete what you do not change.",
        "",
        "# work_root = \"~/.cache/recap\"   # where transcripts and caches are kept",
        "# allow_remote_llm = false        # required to point llm.base_url off-box",
        "",
    ]
    comments = {
        "asr": {
            "backend": "faster-whisper | whisper.cpp",
            "model": "tiny(.en) base(.en) small(.en) medium(.en) large-v3 distil-large-v3",
            "device": "auto | cpu | cuda",
            "compute_type": "auto | int8 | int8_float16 | float16 | float32",
            "language": "e.g. \"en\"; leave unset to autodetect",
            "vad_filter": "skip silence with voice activity detection",
            "initial_prompt": "seed jargon/names to improve accuracy",
            "whisper_cpp_model": "path to a ggml-*.bin model file",
        },
        "llm": {
            "backend": "ollama | openai (LM Studio, llama.cpp server, vLLM ...)",
            "base_url": "must be loopback unless allow_remote_llm = true",
            "model": "any model you have pulled locally",
            "num_ctx": "context window; raise for fewer, larger chunks",
            "api_key_env": "env var holding a key, openai backend only",
        },
        "summary": {
            "chunk_tokens": "transcript tokens per map-step call",
            "fold_batch_size": "notes merged per fold pass on very long recordings",
            "audience": "who the executive summary is written for",
            "focus": "optional steer, e.g. \"budget and hiring decisions\"",
            "extra_instructions": "free-form additions to the summariser prompt",
        },
    }
    for name, section_type in _SECTIONS.items():
        lines.append(f"[{name}]")
        instance = section_type()
        for f in fields(section_type):
            value = getattr(instance, f.name)
            rendered = _toml_scalar(value)
            note = comments.get(name, {}).get(f.name)
            suffix = f"   # {note}" if note else ""
            lines.append(f"# {f.name} = {rendered}{suffix}")
        lines.append("")
    return "\n".join(lines)


def _toml_scalar(value: Any) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return '"' + str(value).replace('"', '\\"') + '"'


def describe(config: Config) -> str:
    """One-line-per-setting dump used by ``recap doctor`` and ``--show-config``."""
    out: list[str] = []
    for name in _SECTIONS:
        section = getattr(config, name)
        for f in fields(section):
            out.append(f"{name}.{f.name} = {getattr(section, f.name)!r}")
    out.append(f"work_root = {config.work_root!r}")
    out.append(f"allow_remote_llm = {config.allow_remote_llm!r}")
    return "\n".join(out)


assert is_dataclass(Config)
