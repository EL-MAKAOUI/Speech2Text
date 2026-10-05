"""The command line.

``speech2text transcribe recording.whatever`` is the whole story; everything
else is for looking at what came out, exporting it again, or setting up the
optional cloud features.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from . import __version__, artifact, export, languages, media, summarize
from .engines import (
    DEFAULT_ENGINE,
    ENGINE_NAMES,
    LOCAL_ENGINE_NAMES,
    EngineError,
    create,
    describe_all,
)
from .engines.whisper import (
    DEFAULT_MODEL,
    MODEL_SIZES,
    _CACHE_ENV as WHISPER_CACHE_ENV,
)

#: Roughly how large each download is, so a 3 GB one is not a surprise.
#: Approximate and for guidance only; nothing depends on these numbers.
APPROXIMATE_SIZES = {
    "tiny": "~75 MB",
    "base": "~145 MB",
    "small": "~485 MB",
    "medium": "~1.5 GB",
    "large-v3": "~3.1 GB",
    "large-v3-turbo": "~1.6 GB",
    "distil-large-v3": "~1.5 GB",
}
from .pipeline import Cancelled, Progress, TranscribeOptions, transcribe_file
from .model import Transcript

DEFAULT_OUTPUT = Path("artifacts") / "desktop"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fail(message: str, code: int = 1) -> int:
    print(f"speech2text: {message}", file=sys.stderr)
    return code


def _bar(progress: Progress, width: int = 28) -> str:
    filled = int(progress.fraction * width)
    return f"[{'#' * filled}{'.' * (width - filled)}] {progress.percent:3d}%"


class _Printer:
    """One rewriting line of progress, quiet when output is piped."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled and sys.stderr.isatty()
        self._last = ""

    def __call__(self, progress: Progress) -> None:
        if not self.enabled:
            return
        line = f"\r{_bar(progress)} {progress.stage:<12} {progress.message[:44]}"
        pad = max(0, len(self._last) - len(line))
        sys.stderr.write(line + " " * pad)
        sys.stderr.flush()
        self._last = line

    def done(self) -> None:
        if self.enabled and self._last:
            sys.stderr.write("\n")
            sys.stderr.flush()


def _engine_options(args: argparse.Namespace) -> dict:
    if args.engine == "whisper":
        options = {"model_size": args.model or DEFAULT_MODEL, "device": args.device}
        if args.compute_type:
            options["compute_type"] = args.compute_type
        return options
    if args.engine == "cloud":
        options = {"provider": args.cloud_provider}
        if args.model:
            options["model"] = args.model
        if args.fallback != "none":
            options["fallback"] = create(args.fallback)
        return options
    return {}


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------


def cmd_transcribe(args: argparse.Namespace) -> int:
    try:
        wanted = languages.parse(args.language)
    except languages.UnknownLanguage as exc:
        return _fail(str(exc))

    sources = [Path(p) for p in args.inputs]
    missing = [p for p in sources if not p.exists()]
    if missing:
        return _fail(f"no such file: {', '.join(str(p) for p in missing)}")

    options = TranscribeOptions(
        engine=args.engine,
        engine_options=_engine_options(args),
        languages=wanted,
        keep_audio=args.keep_audio,
        initial_prompt=args.prompt,
    )

    output_root = Path(args.output) if args.output else DEFAULT_OUTPUT
    run_root = artifact.run_directory(output_root)
    printer = _Printer(not args.quiet)
    failures = 0

    for position, source in enumerate(sources, start=1):
        destination = (
            Path(args.output)
            if args.output and len(sources) == 1 and args.exact_output
            else artifact.allocate(run_root, source.name, position)
        )
        if not args.quiet:
            print(f"{source.name} → {destination}", file=sys.stderr)
        try:
            bundle = transcribe_file(source, destination, options, on_progress=printer)
        except Cancelled:
            printer.done()
            return _fail("stopped", 130)
        except (media.MediaError, EngineError) as exc:
            printer.done()
            failures += 1
            print(f"speech2text: {source.name}: {exc}", file=sys.stderr)
            continue
        printer.done()
        _report_bundle(bundle, args)

    return 1 if failures else 0


def _report_bundle(bundle: artifact.Bundle, args: argparse.Namespace) -> None:
    transcript = bundle.transcript
    if args.stdout:
        print(transcript.corrected_text(), end="")
    else:
        unresolved = len(transcript.unresolved_issues())
        detail = (
            f"{len(transcript.segments)} segments, {transcript.word_count()} words, "
            f"{languages.name_of(transcript.language_detected)}"
        )
        if unresolved:
            detail += f", {unresolved} to check"
        print(f"  {bundle.text_path}  ({detail})", file=sys.stderr)

    for fmt in args.format or ():
        target = bundle.directory / f"transcript.{fmt}"
        export.write(transcript, target, fmt, timestamps=args.timestamps)
        if not args.stdout:
            print(f"  {target}", file=sys.stderr)


def cmd_info(args: argparse.Namespace) -> int:
    """What a file actually is, which has nothing to do with its name."""
    status = 0
    for name in args.inputs:
        try:
            info = media.probe(name)
        except media.MediaError as exc:
            print(f"{name}: {exc}")
            status = 1
            continue
        print(f"{name}")
        print(f"  kind       {info.kind}")
        print(f"  container  {info.container}")
        print(f"  duration   {media.format_duration(info.duration)}")
        print(f"  audio      {info.audio_codec} "
              f"{info.sample_rate or '?'} Hz, {info.channels or '?'} channel(s)")
        if info.has_video:
            print(f"  video      {info.video_codec}")
        print(f"  size       {info.size_bytes / 1e6:.1f} MB")
    return status


def cmd_export(args: argparse.Namespace) -> int:
    try:
        transcript = artifact.load(args.bundle)
    except (FileNotFoundError, ValueError) as exc:
        return _fail(str(exc))
    target = Path(args.destination)
    fmt = args.format or target.suffix.lstrip(".") or "txt"
    try:
        written = export.write(transcript, target, fmt, timestamps=args.timestamps)
    except export.ExportError as exc:
        return _fail(str(exc))
    print(written)
    return 0


def cmd_summarize(args: argparse.Namespace) -> int:
    try:
        transcript = artifact.load(args.bundle)
    except (FileNotFoundError, ValueError) as exc:
        return _fail(str(exc))
    task = (
        summarize.custom_task(args.instruction)
        if args.instruction
        else summarize.task_named(args.task)
    )
    try:
        result = summarize.summarize_transcript(
            transcript,
            task,
            provider=args.provider,
            model=args.model,
            on_progress=lambda n, total: (
                print(f"  part {n}/{total}…", file=sys.stderr) if total > 1 else None
            ),
        )
    except summarize.SummaryError as exc:
        return _fail(str(exc))

    if args.output:
        path = Path(args.output)
        path.write_text(result.to_markdown(transcript.media.name), encoding="utf-8")
        print(path)
    elif Path(args.bundle).is_dir():
        path = summarize.write_result(result, args.bundle, transcript.media.name)
        print(path)
    else:
        print(result.text)
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    """List Whisper models, or fetch one so later runs need no network."""
    from .engines.whisper import WhisperEngine, downloaded_bytes, model_cache_dir

    cache = Path(args.cache).expanduser() if args.cache else model_cache_dir()

    if getattr(args, "size", None):
        engine = WhisperEngine(args.size, cache_dir=cache, device=args.device)
        usable, reason = engine.availability()
        if not usable:
            return _fail(reason)
        if engine.model_ready() and not args.force:
            on_disk = media.format_size(downloaded_bytes(args.size, cache))
            print(f"{args.size} is already in {cache} ({on_disk})")
            return 0

        # A multi-gigabyte download with no word about its size looks stalled.
        expected = APPROXIMATE_SIZES.get(args.size)
        print(
            f"fetching {args.size}"
            + (f", about {expected}" if expected else "")
            + f", into {cache}",
            file=sys.stderr,
        )
        print(
            "The download reports its own progress below. A large model takes "
            "a while; nothing else is needed while it runs.",
            file=sys.stderr,
        )
        try:
            engine.load()
        except EngineError as exc:
            return _fail(str(exc))
        on_disk = media.format_size(downloaded_bytes(args.size, cache))
        print(
            f"{args.size} is ready ({on_disk} in {cache}). "
            f"Later runs of this model need no network."
        )
        return 0

    print(f"Models are kept in {cache}")
    print(f"Set {WHISPER_CACHE_ENV} to keep them somewhere else.\n")
    print(f"{'model':<18}{'approx.':<10}{'state'}")
    for size in MODEL_SIZES:
        ready = WhisperEngine(size, cache_dir=cache).model_ready()
        state = (
            f"downloaded, {media.format_size(downloaded_bytes(size, cache))}"
            if ready
            else "not downloaded"
        )
        print(f"{size:<18}{APPROXIMATE_SIZES.get(size, '—'):<10}{state}")
    print("\nFetch one with:  speech2text models get <model>")
    return 0


def cmd_engines(args: argparse.Namespace) -> int:
    for described in describe_all():
        mark = "available" if described["available"] else "not available"
        upload = "  UPLOADS AUDIO" if described["uploads_audio"] else ""
        print(f"{described['name']:<8} {mark:<14}{upload}")
        print(f"         {described['summary']}")
        if not described["available"]:
            for line in described["reason"].splitlines():
                print(f"         {line}")
    return 0


def cmd_languages(args: argparse.Namespace) -> int:
    print(f"{'auto':<6} Detect automatically")
    for language in languages.choices():
        print(f"{language.code:<6} {language.label}")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    root = Path(args.output) if args.output else DEFAULT_OUTPUT
    found = artifact.find_previous(root)
    if not found:
        print(f"nothing transcribed under {root}")
        return 0
    for run in found:
        checks = f"{run.unresolved_reviews} to check" if run.unresolved_reviews else "—"
        print(
            f"{run.created_at[:16]:<17} {media.format_duration(run.duration):>7}  "
            f"{run.word_count:>6}w  {checks:<12} {run.name}"
        )
        print(f"{'':<17} {run.directory}")
    return 0


def cmd_keys(args: argparse.Namespace) -> int:
    from . import keys as keystore

    if args.keys_command == "list":
        found = keystore.providers_with_keys()
        if not found:
            print("no API keys found. They are only needed for the cloud features.")
            return 0
        for provider, count in sorted(found.items()):
            entries = keystore.keys_for(provider)
            print(f"{provider:<8} {count} key(s)")
            for key in entries:
                print(f"         {key.masked}  from {key.source}")
        return 0

    if args.keys_command == "add":
        import getpass

        value = args.value or getpass.getpass(f"{args.provider} API key (not echoed): ")
        if not value.strip():
            return _fail("no key given")
        path = keystore.save_key(args.provider, value, args.label)
        print(f"stored in {path}")
        return 0

    if args.keys_command == "forget":
        removed = keystore.forget_keys(args.provider)
        print(f"removed {removed} key(s) for {args.provider}")
        return 0

    return _fail("say what to do with keys: list, add, or forget")


def cmd_gui(args: argparse.Namespace) -> int:
    try:
        from .ui.app import main as gui_main
    except ImportError as exc:
        return _fail(
            f"the window needs PySide6: pip install -e '.[ui]'  ({exc})"
        )
    return gui_main(args.inputs)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speech2text",
        description="Turn any audio or video file into text, on this machine.",
    )
    parser.add_argument("--version", action="version", version=f"speech2text {__version__}")
    sub = parser.add_subparsers(dest="command")

    transcribe = sub.add_parser(
        "transcribe",
        help="transcribe one or more recordings",
        description=(
            "Transcribe recordings. The file's name and extension are ignored: "
            "what matters is what is inside it."
        ),
    )
    transcribe.add_argument("inputs", nargs="+", help="audio or video files")
    transcribe.add_argument(
        "-o", "--output", help=f"where bundles are written (default: {DEFAULT_OUTPUT})"
    )
    transcribe.add_argument(
        "--exact-output", action="store_true",
        help="write straight into --output instead of a dated subfolder",
    )
    transcribe.add_argument(
        "-l", "--language", default="auto",
        help="language code or name, or 'auto' to detect it (default: auto)",
    )
    transcribe.add_argument(
        "-e", "--engine", default=DEFAULT_ENGINE, choices=list(ENGINE_NAMES),
        help=f"which recognizer to use (default: {DEFAULT_ENGINE})",
    )
    transcribe.add_argument(
        "-m", "--model", help=f"engine model (whisper: {', '.join(MODEL_SIZES)})"
    )
    transcribe.add_argument(
        "--device", default="auto", choices=("auto", "cpu", "cuda"),
        help="where whisper runs (default: auto)",
    )
    transcribe.add_argument("--compute-type", help="whisper precision, e.g. int8 or float16")
    transcribe.add_argument(
        "--cloud-provider", default="groq", help="provider for --engine cloud",
    )
    transcribe.add_argument(
        "--fallback", default="whisper", choices=(*LOCAL_ENGINE_NAMES, "none"),
        help="local engine to use if the cloud fails (default: whisper)",
    )
    transcribe.add_argument(
        "-f", "--format", action="append", choices=list(export.FORMATS),
        help="also write this format; repeatable",
    )
    transcribe.add_argument(
        "--timestamps", action="store_true", help="include times in txt and docx"
    )
    transcribe.add_argument("--prompt", help="hint the recognizer with names or jargon")
    transcribe.add_argument(
        "--keep-audio", action="store_true", help="keep the decoded audio.wav"
    )
    transcribe.add_argument(
        "--stdout", action="store_true", help="print the text instead of a summary"
    )
    transcribe.add_argument("-q", "--quiet", action="store_true", help="no progress")
    transcribe.set_defaults(func=cmd_transcribe)

    info = sub.add_parser("info", help="say what a file actually is")
    info.add_argument("inputs", nargs="+")
    info.set_defaults(func=cmd_info)

    exporter = sub.add_parser("export", help="write a transcript in another format")
    exporter.add_argument("bundle", help="a bundle folder or its transcript.json")
    exporter.add_argument("destination")
    exporter.add_argument("-f", "--format", choices=list(export.FORMATS))
    exporter.add_argument("--timestamps", action="store_true")
    exporter.set_defaults(func=cmd_export)

    summary = sub.add_parser(
        "summarize", help="hand a transcript to a language model (optional)"
    )
    summary.add_argument("bundle")
    summary.add_argument(
        "-t", "--task", default="summary", choices=sorted(summarize.TASKS),
        help="what to do with it (default: summary)",
    )
    summary.add_argument("--instruction", help="your own instruction instead of a task")
    summary.add_argument("-p", "--provider", default="groq")
    summary.add_argument("-m", "--model")
    summary.add_argument("-o", "--output", help="write here instead of beside the bundle")
    summary.set_defaults(func=cmd_summarize)

    models = sub.add_parser(
        "models",
        help="list or download the Whisper models",
        description=(
            "Whisper models download once and then run offline. With no "
            "arguments this lists them and says which are already here."
        ),
    )
    models.add_argument("--cache", help="look in this folder instead of the default")
    models.set_defaults(func=cmd_models, size=None)
    models_sub = models.add_subparsers(dest="models_command")
    models_get = models_sub.add_parser("get", help="download a model now")
    models_get.add_argument("size", choices=list(MODEL_SIZES))
    models_get.add_argument("--cache")
    models_get.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    models_get.add_argument(
        "--force", action="store_true", help="fetch again even if it is here"
    )
    models_get.set_defaults(func=cmd_models)

    engines_command = sub.add_parser("engines", help="list recognizers and what is ready")
    engines_command.set_defaults(func=cmd_engines)

    languages_command = sub.add_parser("languages", help="list language codes")
    languages_command.set_defaults(func=cmd_languages)

    listing = sub.add_parser("list", help="list earlier transcriptions")
    listing.add_argument("-o", "--output")
    listing.set_defaults(func=cmd_list)

    keys_command = sub.add_parser("keys", help="API keys for the optional cloud features")
    keys_sub = keys_command.add_subparsers(dest="keys_command")
    keys_list = keys_sub.add_parser("list", help="show which keys are found, masked")
    keys_list.set_defaults(func=cmd_keys)
    keys_add = keys_sub.add_parser("add", help="store a key")
    keys_add.add_argument("-p", "--provider", required=True)
    keys_add.add_argument("--value", help="the key; prompted for if omitted")
    keys_add.add_argument("--label")
    keys_add.set_defaults(func=cmd_keys)
    keys_forget = keys_sub.add_parser("forget", help="remove stored keys for a provider")
    keys_forget.add_argument("-p", "--provider", required=True)
    keys_forget.set_defaults(func=cmd_keys)
    keys_command.set_defaults(func=cmd_keys, keys_command=None)

    gui = sub.add_parser("gui", help="open the window")
    gui.add_argument("inputs", nargs="*", help="files to add to the list")
    gui.set_defaults(func=cmd_gui)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    if not media.ffmpeg_available() and args.command in ("transcribe", "info"):
        return _fail(
            "ffmpeg is not installed. It is what lets Speech2Text open any "
            "audio or video file. On Ubuntu: sudo apt install ffmpeg"
        )
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return _fail("stopped", 130)
    except BrokenPipeError:  # pragma: no cover - piping into head
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
