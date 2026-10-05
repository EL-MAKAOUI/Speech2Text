"""Where API keys come from, for the optional cloud features.

Nothing here is needed to transcribe: the local engines never read a key. A
key matters only for the cloud recognizer and for rewriting a finished
transcript with a language model.

Three sources, in order:

1. The environment, which wins so a one-off run can override everything.
2. ``~/.llmkit/keys.json``, the key store shared with the sibling projects.
3. ``~/.speech2text/keys.json``, this application's own store, used when
   llmkit is not installed.

A key is never written to a log, an error message, or an artifact.
"""

from __future__ import annotations

import json
import os
import stat
from dataclasses import dataclass
from pathlib import Path

#: Vendor-specific variables people already have set, read as a last resort.
_VENDOR_ENV: dict[str, tuple[str, ...]] = {
    "gemini": ("SPEECH2TEXT_GEMINI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "groq": ("SPEECH2TEXT_GROQ_API_KEY", "GROQ_API_KEY"),
    "openai": ("SPEECH2TEXT_OPENAI_API_KEY", "OPENAI_API_KEY"),
}

_LLMKIT_STORE = Path("~/.llmkit/keys.json")
_OWN_STORE = Path("~/.speech2text/keys.json")


@dataclass(frozen=True)
class ApiKey:
    provider: str
    value: str
    source: str
    label: str | None = None

    @property
    def masked(self) -> str:
        """``sk-…4f2a`` — enough to recognize a key, not enough to use it."""
        if len(self.value) <= 8:
            return "…"
        return f"{self.value[:3]}…{self.value[-4:]}"

    def __repr__(self) -> str:  # keeps keys out of tracebacks and logs
        return f"ApiKey(provider={self.provider!r}, {self.masked}, from {self.source})"


def _from_environment(provider: str) -> list[ApiKey]:
    for name in _VENDOR_ENV.get(provider, ()):
        value = os.environ.get(name, "").strip()
        if value:
            return [ApiKey(provider=provider, value=value, source=f"${name}")]
    generic = os.environ.get(f"SPEECH2TEXT_{provider.upper()}_API_KEY", "").strip()
    if generic:
        return [ApiKey(provider, generic, f"$SPEECH2TEXT_{provider.upper()}_API_KEY")]
    return []


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.expanduser().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _harvest(node: object, provider: str, source: str) -> list[ApiKey]:
    """Pull keys for one provider out of a store, tolerating its layout.

    The shared llmkit store is owned by another project, so this reads the
    shapes a key store plausibly has rather than assuming one. Anything it
    does not recognize is ignored, never guessed at.
    """
    found: list[ApiKey] = []

    def add(value: object, label: object = None) -> None:
        if isinstance(value, str) and value.strip():
            found.append(
                ApiKey(
                    provider=provider,
                    value=value.strip(),
                    source=source,
                    label=str(label) if label else None,
                )
            )

    def walk(item: object) -> None:
        if isinstance(item, str):
            add(item)
        elif isinstance(item, dict):
            # {"provider": "groq", "key": "..."} or {"api_key": ...}
            named = item.get("provider") or item.get("vendor") or item.get("name")
            if isinstance(named, str) and named.casefold() == provider:
                for field in ("key", "api_key", "value", "secret", "token"):
                    if field in item:
                        add(item[field], item.get("label") or item.get("id"))
                        return
            for field in ("key", "api_key", "value", "secret", "token"):
                if field in item and named is None:
                    add(item[field], item.get("label") or item.get("id"))
                    return
        elif isinstance(item, list):
            for entry in item:
                walk(entry)

    if isinstance(node, dict):
        # {"keys": [...]} or {"gemini": [...]} or {"providers": {"gemini": ...}}
        for container in ("providers", "keys"):
            inner = node.get(container)
            if isinstance(inner, dict) and provider in inner:
                walk(inner[provider])
            elif isinstance(inner, list):
                walk(inner)
        if provider in node:
            walk(node[provider])
    elif isinstance(node, list):
        walk(node)
    return found


def keys_for(provider: str) -> list[ApiKey]:
    """Every key available for a provider, best source first."""
    provider = provider.strip().casefold()
    found = _from_environment(provider)
    if found:
        return found
    for path, source in ((_LLMKIT_STORE, "llmkit"), (_OWN_STORE, "speech2text")):
        data = _read_json(path)
        if data is not None:
            found.extend(_harvest(data, provider, source))
    # The same key can appear in both stores; keep the first of each.
    unique: dict[str, ApiKey] = {}
    for key in found:
        unique.setdefault(key.value, key)
    return list(unique.values())


def first_key(provider: str) -> ApiKey | None:
    found = keys_for(provider)
    return found[0] if found else None


def providers_with_keys(candidates: tuple[str, ...] = ("gemini", "groq", "openai")) -> dict[str, int]:
    """How many keys exist per provider, for the window to report before a run."""
    return {name: len(keys_for(name)) for name in candidates if keys_for(name)}


def save_key(provider: str, value: str, label: str | None = None) -> Path:
    """Store a key in this application's own file, readable only by its owner."""
    provider = provider.strip().casefold()
    value = value.strip()
    if not value:
        raise ValueError("refusing to store an empty key")
    path = _OWN_STORE.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = _read_json(path)
    store = existing if isinstance(existing, dict) else {}
    entries = store.setdefault("keys", [])
    if not isinstance(entries, list):  # pragma: no cover - corrupt file
        entries = []
        store["keys"] = entries
    entries.append({"provider": provider, "key": value, "label": label})
    path.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR)
    return path


def forget_keys(provider: str) -> int:
    """Remove this application's stored keys for a provider. Returns how many."""
    provider = provider.strip().casefold()
    path = _OWN_STORE.expanduser()
    store = _read_json(path)
    if not isinstance(store, dict) or not isinstance(store.get("keys"), list):
        return 0
    before = store["keys"]
    after = [
        entry for entry in before
        if not (isinstance(entry, dict)
                and str(entry.get("provider", "")).casefold() == provider)
    ]
    store["keys"] = after
    path.write_text(json.dumps(store, indent=2) + "\n", encoding="utf-8")
    return len(before) - len(after)
