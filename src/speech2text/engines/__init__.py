"""The recognizers, and how one is chosen by name."""

from __future__ import annotations

from typing import Any

from .base import (
    DEFAULT_REVIEW_THRESHOLD,
    EngineError,
    EngineResult,
    EngineUnavailable,
    ProgressFn,
    SpeechEngine,
    TranscriptionRequest,
    confidence_from_logprob,
    review_issues,
)

#: The order the window lists them in, and the order a fallback is chosen in.
ENGINE_NAMES = ("whisper", "sphinx", "cloud")
#: The engines that run here. Only these can serve as a cloud fallback: a
#: cloud engine falling back to a cloud engine helps nobody.
LOCAL_ENGINE_NAMES = ("whisper", "sphinx")
DEFAULT_ENGINE = "whisper"

__all__ = [
    "DEFAULT_ENGINE",
    "DEFAULT_REVIEW_THRESHOLD",
    "ENGINE_NAMES",
    "LOCAL_ENGINE_NAMES",
    "EngineError",
    "EngineResult",
    "EngineUnavailable",
    "ProgressFn",
    "SpeechEngine",
    "TranscriptionRequest",
    "confidence_from_logprob",
    "create",
    "describe_all",
    "review_issues",
]


def create(name: str = DEFAULT_ENGINE, **options: Any) -> SpeechEngine:
    """Build an engine by name.

    Importing an engine's dependencies is deferred to its own module, so
    asking for one engine never requires another's packages to be installed.
    """
    key = (name or DEFAULT_ENGINE).strip().casefold()
    if key == "whisper":
        from .whisper import WhisperEngine

        return WhisperEngine(**options)
    if key == "sphinx":
        from .sphinx import SphinxEngine

        return SphinxEngine(**options)
    if key == "cloud":
        from .cloud import CloudEngine

        return CloudEngine(**options)
    raise EngineUnavailable(
        f"unknown engine {name!r}. Choose one of: {', '.join(ENGINE_NAMES)}"
    )


def describe_all() -> list[dict[str, Any]]:
    """Every engine with whether it can run here, for the window and the CLI."""
    described = []
    for name in ENGINE_NAMES:
        try:
            engine = create(name)
            usable, reason = engine.availability()
            described.append(
                {
                    "name": name,
                    "title": engine.title,
                    "summary": engine.summary,
                    "uploads_audio": engine.uploads_audio,
                    "available": usable,
                    "reason": reason,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive
            described.append(
                {
                    "name": name,
                    "title": name,
                    "summary": "",
                    "uploads_audio": False,
                    "available": False,
                    "reason": str(exc),
                }
            )
    return described
