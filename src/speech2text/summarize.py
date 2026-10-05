"""Optional: hand a finished transcript to a language model.

This is a separate step on purpose. Transcribing is the job; summarizing is
something some people want afterwards and others never do. Nothing here runs
unless it is asked for, and a transcript is complete and exportable without
it.

A long recording does not fit in one request, so the text is split, each part
is handled, and the parts are combined in a second pass.
"""

from __future__ import annotations

import json
import textwrap
import urllib.parse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from .engines.cloud import CloudHttpError, _request
from .keys import ApiKey, first_key
from .model import Transcript

#: Roughly four characters to a token. Used only to decide where to split, so
#: an approximation is enough; the model is never told this number.
_CHARS_PER_TOKEN = 4
#: Leaves room for the instruction and the reply inside a small context window.
DEFAULT_CHUNK_CHARS = 24_000


class SummaryError(RuntimeError):
    """A rewrite could not be produced."""


@dataclass(frozen=True)
class Task:
    """One thing that can be done to a transcript."""

    name: str
    title: str
    instruction: str
    #: How the parts of a long recording are combined.
    combine: str = "Combine these notes into one result, without repeating anything."

    def describe(self) -> str:
        return f"{self.name} — {self.title}"


TASKS: dict[str, Task] = {
    "summary": Task(
        name="summary",
        title="A short summary",
        instruction=(
            "Summarize this transcript in a few clear paragraphs. Keep the "
            "speaker's meaning and any numbers, names and dates exactly as "
            "stated. Do not add anything that is not in the transcript."
        ),
        combine=(
            "These are summaries of consecutive parts of one recording. Write "
            "a single coherent summary of the whole thing, without repeating."
        ),
    ),
    "key-points": Task(
        name="key-points",
        title="The main points, as a list",
        instruction=(
            "List the main points of this transcript as concise bullet points. "
            "Only points actually made in it."
        ),
        combine="Merge these bullet lists into one, dropping duplicates.",
    ),
    "actions": Task(
        name="actions",
        title="Action items and decisions",
        instruction=(
            "From this transcript, list the decisions made and the action "
            "items, saying who owns each one where the transcript says so. If "
            "there are none, say so plainly rather than inventing any."
        ),
        combine="Merge these lists of decisions and actions into one.",
    ),
    "minutes": Task(
        name="minutes",
        title="Meeting minutes",
        instruction=(
            "Write meeting minutes from this transcript: attendees if they can "
            "be identified, topics discussed, decisions, and next steps. Use "
            "headings. Do not invent attendees or outcomes."
        ),
        combine="Combine these minutes into one document, in order, without repeating.",
    ),
    "outline": Task(
        name="outline",
        title="An outline of what was covered",
        instruction=(
            "Produce a nested outline of the topics in this transcript, in the "
            "order they were discussed."
        ),
        combine="Combine these outlines into one, preserving order.",
    ),
    "tidy": Task(
        name="tidy",
        title="The same words, tidied up",
        instruction=(
            "Rewrite this transcript as clean readable prose. Remove filler "
            "words, false starts and stutters, and fix obvious transcription "
            "slips. Keep every substantive word the speaker said, keep their "
            "voice, and do not summarize, shorten or add."
        ),
        combine="Join these cleaned sections into one continuous text, in order.",
    ),
}


def task_named(name: str) -> Task:
    try:
        return TASKS[name.strip().casefold()]
    except KeyError:
        known = ", ".join(sorted(TASKS))
        raise SummaryError(f"unknown task {name!r}. Known tasks: {known}") from None


def custom_task(instruction: str) -> Task:
    """Whatever the person typed, used as the instruction verbatim."""
    return Task(name="custom", title="Custom instruction", instruction=instruction)


# ---------------------------------------------------------------------------
# providers
# ---------------------------------------------------------------------------


class TextProvider(ABC):
    """A chat model reachable over HTTP."""

    name = "provider"
    model = ""

    @abstractmethod
    def complete(self, instruction: str, text: str, key: ApiKey) -> str:
        ...


class OpenAICompatibleText(TextProvider):
    """Anything exposing ``/v1/chat/completions``: OpenAI, Groq, and others."""

    def __init__(self, name: str, base_url: str, model: str) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.model = model

    def complete(self, instruction: str, text: str, key: ApiKey) -> str:
        payload = {
            "model": self.model,
            "temperature": 0.2,
            "messages": [
                {"role": "system", "content": instruction},
                {"role": "user", "content": text},
            ],
        }
        raw = _request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {key.value}",
                "Content-Type": "application/json",
            },
        )
        try:
            parsed = json.loads(raw.decode("utf-8", "replace"))
            return parsed["choices"][0]["message"]["content"].strip()
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            raise SummaryError("the model returned no text") from exc


class GeminiText(TextProvider):
    name = "gemini"

    def __init__(self, model: str = "gemini-2.5-flash") -> None:
        self.model = model

    def complete(self, instruction: str, text: str, key: ApiKey) -> str:
        payload = {
            "contents": [{"parts": [{"text": text}]}],
            "systemInstruction": {"parts": [{"text": instruction}]},
            "generationConfig": {"temperature": 0.2},
        }
        raw = _request(
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{self.model}:generateContent",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "x-goog-api-key": key.value},
        )
        try:
            parsed = json.loads(raw.decode("utf-8", "replace"))
            parts = parsed["candidates"][0]["content"]["parts"]
            return "".join(p.get("text", "") for p in parts).strip()
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            raise SummaryError("the model returned no text") from exc


PROVIDERS: dict[str, Callable[[str | None], TextProvider]] = {
    "groq": lambda model=None: OpenAICompatibleText(
        "groq", "https://api.groq.com/openai/v1", model or "llama-3.3-70b-versatile"
    ),
    "openai": lambda model=None: OpenAICompatibleText(
        "openai", "https://api.openai.com/v1", model or "gpt-4o-mini"
    ),
    "gemini": lambda model=None: GeminiText(model or "gemini-2.5-flash"),
}


def build_provider(name: str, model: str | None = None) -> TextProvider:
    try:
        return PROVIDERS[name.strip().casefold()](model)
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise SummaryError(f"unknown provider {name!r}. Known: {known}") from None


def available_providers() -> dict[str, int]:
    """Providers with at least one key, so the window can say what is ready."""
    from .keys import providers_with_keys

    return providers_with_keys(tuple(PROVIDERS))


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def split_text(text: str, limit: int = DEFAULT_CHUNK_CHARS) -> list[str]:
    """Split long text at paragraph, then sentence, then word boundaries.

    Splitting mid-sentence costs the model context it needs, so boundaries are
    preferred in that order and a hard split is the last resort.
    """
    text = text.strip()
    if not text:
        return []
    if len(text) <= limit:
        return [text]

    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n"):
        candidate = f"{current}\n{paragraph}" if current else paragraph
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks.append(current)
            current = ""
        if len(paragraph) <= limit:
            current = paragraph
            continue
        for piece in _split_long(paragraph, limit):
            if len(piece) <= limit:
                if current and len(current) + len(piece) + 1 <= limit:
                    current = f"{current} {piece}"
                else:
                    if current:
                        chunks.append(current)
                    current = piece
    if current:
        chunks.append(current)
    return [c.strip() for c in chunks if c.strip()]


def _split_long(paragraph: str, limit: int) -> list[str]:
    import re

    sentences = re.split(r"(?<=[.!?؟۔…])\s+", paragraph)
    pieces: list[str] = []
    for sentence in sentences:
        if len(sentence) <= limit:
            pieces.append(sentence)
        else:
            pieces.extend(textwrap.wrap(sentence, limit) or [sentence[:limit]])
    return pieces


def estimated_tokens(text: str) -> int:
    return max(1, len(text) // _CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# running a task
# ---------------------------------------------------------------------------


@dataclass
class SummaryResult:
    task: str
    text: str
    provider: str
    model: str
    parts: int
    source_words: int

    def to_markdown(self, source_name: str) -> str:
        title = TASKS.get(self.task, custom_task("")).title if self.task in TASKS else "Result"
        return (
            f"# {title}\n\n"
            f"*{source_name} · {self.source_words} words · "
            f"{self.provider} {self.model}*\n\n"
            f"{self.text.strip()}\n"
        )


def run_task(
    text: str,
    task: Task,
    *,
    provider: TextProvider | str = "groq",
    model: str | None = None,
    key: ApiKey | None = None,
    chunk_chars: int = DEFAULT_CHUNK_CHARS,
    on_progress: Callable[[int, int], None] | None = None,
) -> SummaryResult:
    """Apply a task to text, in as many passes as its length needs."""
    engine = build_provider(provider, model) if isinstance(provider, str) else provider
    api_key = key or first_key(engine.name)
    if api_key is None:
        raise SummaryError(
            f"no API key for {engine.name}. Add one with:\n"
            f"    speech2text keys add --provider {engine.name}"
        )
    body = text.strip()
    if not body:
        raise SummaryError("there is no text to work with")

    chunks = split_text(body, chunk_chars)
    outputs: list[str] = []
    for position, chunk in enumerate(chunks, start=1):
        if on_progress is not None:
            on_progress(position, len(chunks))
        try:
            outputs.append(engine.complete(task.instruction, chunk, api_key))
        except CloudHttpError as exc:
            raise SummaryError(f"the request failed: {exc}") from exc

    if len(outputs) == 1:
        combined = outputs[0]
    else:
        joined = "\n\n".join(
            f"--- part {i} of {len(outputs)} ---\n{part}"
            for i, part in enumerate(outputs, start=1)
        )
        try:
            combined = engine.complete(task.combine, joined, api_key)
        except CloudHttpError as exc:
            raise SummaryError(f"combining the parts failed: {exc}") from exc

    return SummaryResult(
        task=task.name,
        text=combined,
        provider=engine.name,
        model=engine.model,
        parts=len(chunks),
        source_words=len(body.split()),
    )


def summarize_transcript(
    transcript: Transcript,
    task: Task | str = "summary",
    **options,
) -> SummaryResult:
    """Run a task over a transcript's current text, corrections included."""
    chosen = task_named(task) if isinstance(task, str) else task
    return run_task(transcript.plain_text(), chosen, **options)


def write_result(result: SummaryResult, directory: str | Path, source_name: str) -> Path:
    """Save a result beside the bundle it came from."""
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{result.task}.md"
    path.write_text(result.to_markdown(source_name), encoding="utf-8")
    return path
