"""Environment self-check: everything the pipeline needs, and how to fix it."""

from __future__ import annotations

import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from .config import Config, is_local_endpoint
from .llm import LLMError, build_client
from .util import style

OK = "ok"
WARN = "warn"
FAIL = "fail"


@dataclass
class Check:
    name: str
    status: str
    detail: str
    fix: str = ""

    def render(self) -> str:
        mark = {OK: style("[ ok ]", "green"), WARN: style("[warn]", "yellow"), FAIL: style("[fail]", "red")}[self.status]
        lines = [f"{mark} {self.name}: {self.detail}"]
        if self.fix and self.status != OK:
            for line in self.fix.splitlines():
                lines.append(f"       {style(line, 'dim')}")
        return "\n".join(lines)


def run_checks(config: Config) -> list[Check]:
    checks = [
        _check_python(),
        _check_ffmpeg(),
        _check_asr(config),
        _check_llm(config),
        _check_privacy(config),
        _check_workdir(config),
    ]
    return checks


def _check_python() -> Check:
    version = platform.python_version()
    status = OK if sys.version_info >= (3, 11) else FAIL
    return Check(
        "python",
        status,
        f"{version} on {platform.system()} {platform.machine()}",
        fix="recap needs Python 3.11 or newer (it uses the stdlib tomllib).",
    )


def _check_ffmpeg() -> Check:
    path = shutil.which("ffmpeg")
    if path:
        return Check("ffmpeg", OK, path)
    return Check(
        "ffmpeg",
        WARN,
        "not found",
        fix=(
            "Needed to read mp3/m4a/mp4 and for the whisper.cpp backend.\n"
            "macOS: brew install ffmpeg | Debian: sudo apt install ffmpeg"
        ),
    )


def _check_asr(config: Config) -> Check:
    backend = config.asr.backend.lower()
    if backend.startswith("whisper.cpp") or backend.startswith("whisper-cpp"):
        binary = shutil.which(config.asr.whisper_cpp_bin)
        model = config.asr.whisper_cpp_model
        if not binary:
            return Check(
                "asr (whisper.cpp)", FAIL, f"{config.asr.whisper_cpp_bin} not on PATH",
                fix="Build whisper.cpp, or set asr.whisper_cpp_bin to the binary path.",
            )
        if not model or not Path(model).exists():
            return Check(
                "asr (whisper.cpp)", FAIL, f"model file missing: {model or 'unset'}",
                fix="Set asr.whisper_cpp_model to a ggml-*.bin file.",
            )
        return Check("asr (whisper.cpp)", OK, f"{binary} with {model}")

    try:
        import faster_whisper  # type: ignore

        version = getattr(faster_whisper, "__version__", "installed")
        detail = f"faster-whisper {version}, model {config.asr.model}"
        try:  # pragma: no cover - hardware dependent
            import torch  # type: ignore

            if torch.cuda.is_available():
                detail += ", CUDA available"
        except Exception:
            pass
        return Check("asr (faster-whisper)", OK, detail)
    except ImportError:
        return Check(
            "asr (faster-whisper)", FAIL, "not installed",
            fix="pip install 'recap[whisper]'   (downloads the model on first run)",
        )


def _check_llm(config: Config) -> Check:
    label = f"llm ({config.llm.backend})"
    try:
        client = build_client(config.llm)
        models = client.list_models()
    except LLMError as exc:
        fix = (
            "Start the server, then pull a model:\n"
            "  ollama serve\n"
            f"  ollama pull {config.llm.model}"
            if config.llm.backend == "ollama"
            else "Start your local OpenAI-compatible server and check llm.base_url."
        )
        return Check(label, FAIL, str(exc).splitlines()[0], fix=fix)

    wanted = [config.llm.model]
    if config.llm.reduce_model:
        wanted.append(config.llm.reduce_model)
    for model in wanted:
        if not client.with_model(model)._model_installed(models):
            listed = ", ".join(models[:8]) or "(none installed)"
            return Check(
                label, FAIL, f"{model!r} not installed; available: {listed}",
                fix=f"ollama pull {model}",
            )
    serving = " + ".join(wanted)
    return Check(label, OK, f"{config.llm.base_url} serving {serving}")


def _check_privacy(config: Config) -> Check:
    if is_local_endpoint(config.llm.base_url):
        return Check("privacy", OK, f"LLM endpoint {config.llm.base_url} is on this machine")
    if config.allow_remote_llm:
        return Check(
            "privacy", WARN,
            f"transcript text will be sent to {config.llm.base_url} (allow_remote_llm is on)",
        )
    return Check(
        "privacy", FAIL, f"{config.llm.base_url} is not local and allow_remote_llm is off",
        fix="Point llm.base_url at localhost, or pass --allow-remote-llm to opt in.",
    )


def _check_workdir(config: Config) -> Check:
    path = Path(config.work_root).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        return Check("work dir", FAIL, f"{path}: {exc}", fix="Set work_root to a writable directory.")
    usage = shutil.disk_usage(path)
    free_gb = usage.free / 1e9
    status = OK if free_gb >= 2 else WARN
    return Check(
        "work dir", status, f"{path} ({free_gb:.1f} GB free)",
        fix="Whisper models need 0.1-3 GB depending on size.",
    )


def worst_status(checks: list[Check]) -> str:
    if any(check.status == FAIL for check in checks):
        return FAIL
    if any(check.status == WARN for check in checks):
        return WARN
    return OK
