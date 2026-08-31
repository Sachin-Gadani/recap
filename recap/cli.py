"""Terminal entry point for recap."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, doctor, pipeline, render
from .asr import ASRError
from .audio import AudioError
from .config import Config, ConfigError, config_template, default_config_path, describe, load_config
from .llm import LLMError
from .summarize import SummarizeError
from .transcript import load_transcript
from .util import (
    eprint,
    error,
    human_duration,
    info,
    sha256_file,
    slugify,
    step,
    style,
    write_atomic,
)

EPILOG = """\
examples:
  recap run interview.m4a                    transcribe and summarise, write ./<title>.md
  recap run standup.mp3 -o notes/monday.md   choose the output path
  recap run call.wav --focus "budget"        steer what the summary emphasises
  recap run call.wav --print                 print the summary to stdout as well
  recap transcribe long.m4a -o long.srt      transcription only
  recap summarize long.srt -o exec.md        summarise an existing transcript
  recap doctor                               check ffmpeg, whisper and the local LLM

Everything runs on this machine. recap refuses a non-loopback llm.base_url unless
you pass --allow-remote-llm.
"""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="recap",
        description="Turn longform meeting or interview audio into an executive summary, locally.",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"recap {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="audio in, executive summary out", epilog=EPILOG,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    run_parser.add_argument("audio", type=Path, nargs="?", help="path to the recording")
    run_parser.add_argument("-o", "--output", type=Path, help="output path (extension chosen per format)")
    run_parser.add_argument("--formats", default=",".join(pipeline.DEFAULT_FORMATS),
                            help=f"comma-separated: {', '.join(pipeline.ALL_FORMATS)}")
    run_parser.add_argument("--transcript", type=Path,
                            help="skip transcription and summarise this transcript instead")
    run_parser.add_argument("--retranscribe", action="store_true", help="ignore the cached transcript")
    run_parser.add_argument("--no-cache", action="store_true", help="do not reuse cached LLM calls")
    run_parser.add_argument("--notes", action="store_true", help="append quotes and figures to the report")
    run_parser.add_argument("--print", dest="print_summary", action="store_true",
                            help="also print the summary to stdout")
    _add_common(run_parser)

    transcribe_parser = subparsers.add_parser("transcribe", help="transcribe audio only")
    transcribe_parser.add_argument("audio", type=Path)
    transcribe_parser.add_argument("-o", "--output", type=Path,
                                   help="output file; .json .txt .md .srt .vtt (default: <name>.json)")
    transcribe_parser.add_argument("--retranscribe", action="store_true")
    _add_common(transcribe_parser)

    summarize_parser = subparsers.add_parser("summarize", help="summarise an existing transcript")
    summarize_parser.add_argument("transcript", type=Path, help=".json, .srt, .vtt or .txt")
    summarize_parser.add_argument("-o", "--output", type=Path)
    summarize_parser.add_argument("--formats", default="md,json")
    summarize_parser.add_argument("--no-cache", action="store_true")
    summarize_parser.add_argument("--notes", action="store_true")
    summarize_parser.add_argument("--print", dest="print_summary", action="store_true")
    _add_common(summarize_parser)

    doctor_parser = subparsers.add_parser("doctor", help="check that the local toolchain is ready")
    doctor_parser.add_argument("--json", action="store_true", help="machine-readable output")
    _add_common(doctor_parser)

    models_parser = subparsers.add_parser("models", help="list models the local LLM server has")
    _add_common(models_parser)

    config_parser = subparsers.add_parser("config", help="show or create the config file")
    config_parser.add_argument("--init", action="store_true", help="write a commented config file")
    config_parser.add_argument("--path", action="store_true", help="print the config file path")
    config_parser.add_argument("--force", action="store_true", help="overwrite an existing config file")
    _add_common(config_parser)

    return parser


def _add_common(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("configuration")
    group.add_argument("-c", "--config", type=Path, help="path to a config.toml")
    group.add_argument("--asr-backend", choices=["faster-whisper", "whisper.cpp"])
    group.add_argument("--asr-model", help="whisper model, e.g. small.en, medium, large-v3")
    group.add_argument("--language", help="force a language code, e.g. en")
    group.add_argument("--device", choices=["auto", "cpu", "cuda"])
    group.add_argument("--compute-type", choices=["auto", "int8", "int8_float16", "float16", "float32"])
    group.add_argument("--llm-backend", choices=["ollama", "openai"])
    group.add_argument("--llm-model", help="local model name, e.g. llama3.1:8b")
    group.add_argument("--llm-url", help="base URL of the local model server")
    group.add_argument("--num-ctx", type=int, help="LLM context window in tokens")
    group.add_argument("--temperature", type=float)
    group.add_argument("--chunk-tokens", type=int, help="transcript tokens per extraction call")
    group.add_argument("--audience", help="who the summary is written for")
    group.add_argument("--focus", help="what the summary should emphasise")
    group.add_argument("--instructions", help="extra instructions for the summariser")
    group.add_argument("--summary-language", help="language to write the summary in")
    group.add_argument("--work-root", help="where transcripts and caches are kept")
    group.add_argument("--allow-remote-llm", action="store_true",
                       help="permit a non-loopback llm.base_url (transcript text leaves this machine)")
    group.add_argument("-q", "--quiet", action="store_true", help="suppress progress output")


def config_from_args(args: argparse.Namespace) -> Config:
    overrides = {
        "asr": {
            "backend": getattr(args, "asr_backend", None),
            "model": getattr(args, "asr_model", None),
            "language": getattr(args, "language", None),
            "device": getattr(args, "device", None),
            "compute_type": getattr(args, "compute_type", None),
        },
        "llm": {
            "backend": getattr(args, "llm_backend", None),
            "model": getattr(args, "llm_model", None),
            "base_url": getattr(args, "llm_url", None),
            "num_ctx": getattr(args, "num_ctx", None),
            "temperature": getattr(args, "temperature", None),
        },
        "summary": {
            "chunk_tokens": getattr(args, "chunk_tokens", None),
            "audience": getattr(args, "audience", None),
            "focus": getattr(args, "focus", None),
            "extra_instructions": getattr(args, "instructions", None),
            "language": getattr(args, "summary_language", None),
        },
        "_root": {
            "work_root": getattr(args, "work_root", None),
            "allow_remote_llm": True if getattr(args, "allow_remote_llm", False) else None,
        },
    }
    return load_config(getattr(args, "config", None), overrides)


def _parse_formats(raw: str) -> tuple[str, ...]:
    formats = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = [f for f in formats if f not in pipeline.ALL_FORMATS]
    if unknown:
        raise ConfigError(
            f"unknown format(s): {', '.join(unknown)}. Choose from {', '.join(pipeline.ALL_FORMATS)}"
        )
    return formats or pipeline.DEFAULT_FORMATS


def _require_audio(path: Path | None) -> Path:
    if path is None:
        raise ConfigError("no audio file given. Try: recap run meeting.m4a")
    if not path.exists():
        raise ConfigError(f"audio file not found: {path}")
    return path


def cmd_run(args: argparse.Namespace, config: Config) -> int:
    formats = _parse_formats(args.formats)
    if args.transcript:
        source = args.transcript
    else:
        source = _require_audio(args.audio)

    result = pipeline.run(
        source=source,
        config=config,
        out_base=args.output.with_suffix("") if args.output else None,
        formats=formats,
        transcript_path=args.transcript,
        force_transcribe=args.retranscribe,
        use_cache=not args.no_cache,
        include_notes=args.notes,
        quiet=args.quiet,
    )
    _report_outputs(result.outputs, args.quiet)
    if not args.quiet:
        info(f"done in {human_duration(result.elapsed)} · work dir {result.work_dir}")
    if args.print_summary:
        print(render.render_plain(result.summary))
    return 0


def cmd_transcribe(args: argparse.Namespace, config: Config) -> int:
    source = _require_audio(args.audio)
    digest = sha256_file(source)
    work_dir = pipeline.work_dir_for(config, source, digest)
    transcript = pipeline.prepare_transcript(
        source, config, work_dir, force=args.retranscribe, quiet=args.quiet, digest=digest
    )
    output = args.output or Path.cwd() / f"{slugify(source.stem)}.json"
    suffix = output.suffix.lower()
    if suffix == ".json":
        transcript.save(output)
    elif suffix == ".srt":
        write_atomic(output, transcript.to_srt())
    elif suffix == ".vtt":
        write_atomic(output, transcript.to_vtt())
    elif suffix == ".md":
        write_atomic(output, render.render_transcript_markdown(transcript, source.stem))
    else:
        write_atomic(output, transcript.to_txt())
    _report_outputs({"transcript": output}, args.quiet)
    return 0


def cmd_summarize(args: argparse.Namespace, config: Config) -> int:
    if not args.transcript.exists():
        raise ConfigError(f"transcript not found: {args.transcript}")
    formats = _parse_formats(args.formats)
    transcript = load_transcript(args.transcript)
    if not args.quiet:
        step(f"Loaded {transcript.word_count():,} words from {args.transcript.name}")
    client = pipeline.build_llm_client(config)
    work_dir = pipeline.work_dir_for(config, args.transcript, transcript.audio_sha256 or "given")
    summary = pipeline.summarize(
        transcript, config, work_dir, client=client, use_cache=not args.no_cache, quiet=args.quiet
    )
    base = args.output.with_suffix("") if args.output else Path.cwd() / slugify(summary.title, 60)
    outputs = pipeline.write_outputs(summary, transcript, base, formats, args.notes)
    _report_outputs(outputs, args.quiet)
    if args.print_summary:
        print(render.render_plain(summary))
    return 0


def cmd_doctor(args: argparse.Namespace, config: Config) -> int:
    checks = doctor.run_checks(config)
    if args.json:
        print(json.dumps([check.__dict__ for check in checks], indent=2))
    else:
        for check in checks:
            eprint(check.render())
    status = doctor.worst_status(checks)
    if status == doctor.FAIL:
        eprint(style("\nrecap is not ready to run. Fix the [fail] items above.", "red"))
        return 1
    if not args.json:
        eprint(style("\nrecap is ready.", "green"))
    return 0


def cmd_models(args: argparse.Namespace, config: Config) -> int:
    client = pipeline.build_llm_client(config, check=False)
    models = client.list_models()
    if not models:
        eprint(f"no models installed on {config.llm.base_url}")
        eprint(f"try: ollama pull {config.llm.model}")
        return 1
    for name in models:
        marker = "*" if name == config.llm.model else " "
        print(f"{marker} {name}")
    return 0


def cmd_config(args: argparse.Namespace, config: Config) -> int:
    path = args.config or default_config_path()
    if args.path:
        print(path)
        return 0
    if args.init:
        if path.exists() and not args.force:
            error(f"{path} already exists (use --force to overwrite)")
            return 1
        write_atomic(path, config_template())
        eprint(f"wrote {path}")
        return 0
    print(describe(config))
    return 0


def _report_outputs(outputs: dict[str, Path], quiet: bool) -> None:
    for path in outputs.values():
        # Paths go to stdout so `recap run x.m4a | xargs open` works.
        print(path)
    if not quiet and outputs:
        eprint(style(f"wrote {len(outputs)} file(s)", "green"))


COMMANDS = {
    "run": cmd_run,
    "transcribe": cmd_transcribe,
    "summarize": cmd_summarize,
    "doctor": cmd_doctor,
    "models": cmd_models,
    "config": cmd_config,
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        # `config --init/--path` names the file to write, so do not try to read it.
        if args.command == "config" and (args.init or args.path):
            config = Config()
        else:
            config = config_from_args(args)
        return COMMANDS[args.command](args, config)
    except KeyboardInterrupt:
        eprint("\ninterrupted; cached work is kept, re-run to resume")
        return 130
    except (ConfigError, AudioError, ASRError, LLMError, SummarizeError, ValueError) as exc:
        error(str(exc))
        return 1
    except FileNotFoundError as exc:
        error(str(exc))
        return 1


if __name__ == "__main__":
    sys.exit(main())
