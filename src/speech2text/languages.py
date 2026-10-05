"""Which language the speech is in.

Choosing is optional. ``auto`` asks the engine to listen and decide, which is
the right default for a file someone was just handed. Naming the language is
worth it when it is known: detection can misfire on the first few seconds of
music, noise, or a greeting in another language.
"""

from __future__ import annotations

from dataclasses import dataclass

AUTO = "auto"

# Right-to-left scripts, so exported documents can be laid out correctly.
_RTL = frozenset({"ar", "he", "fa", "ur", "ps", "sd", "yi"})

# The languages Whisper is trained on, with their codes as the engines use them.
_NAMES: dict[str, str] = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish",
    "ru": "Russian", "ko": "Korean", "fr": "French", "ja": "Japanese",
    "pt": "Portuguese", "tr": "Turkish", "pl": "Polish", "ca": "Catalan",
    "nl": "Dutch", "ar": "Arabic", "sv": "Swedish", "it": "Italian",
    "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay",
    "cs": "Czech", "ro": "Romanian", "da": "Danish", "hu": "Hungarian",
    "ta": "Tamil", "no": "Norwegian", "th": "Thai", "ur": "Urdu",
    "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian", "la": "Latin",
    "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali",
    "sr": "Serbian", "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada",
    "et": "Estonian", "mk": "Macedonian", "br": "Breton", "eu": "Basque",
    "is": "Icelandic", "hy": "Armenian", "ne": "Nepali", "mn": "Mongolian",
    "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala",
    "km": "Khmer", "sn": "Shona", "yo": "Yoruba", "so": "Somali",
    "af": "Afrikaans", "oc": "Occitan", "ka": "Georgian", "be": "Belarusian",
    "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati", "am": "Amharic",
    "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk",
    "mt": "Maltese", "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Burmese",
    "bo": "Tibetan", "tl": "Tagalog", "mg": "Malagasy", "as": "Assamese",
    "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala", "ha": "Hausa",
    "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}

# Shown beside the English name where it helps someone recognize their own
# language in a long list. Left out rather than guessed.
_NATIVE: dict[str, str] = {
    "ar": "العربية", "zh": "中文", "de": "Deutsch", "es": "Español",
    "ru": "Русский", "ko": "한국어", "fr": "Français", "ja": "日本語",
    "pt": "Português", "tr": "Türkçe", "pl": "Polski", "nl": "Nederlands",
    "sv": "Svenska", "it": "Italiano", "hi": "हिन्दी", "fa": "فارسی",
    "he": "עברית", "ur": "اردو", "el": "Ελληνικά", "uk": "Українська",
    "cs": "Čeština", "da": "Dansk", "fi": "Suomi", "no": "Norsk",
    "th": "ไทย", "vi": "Tiếng Việt", "id": "Bahasa Indonesia", "ro": "Română",
    "hu": "Magyar", "bn": "বাংলা", "ta": "தமிழ்", "ca": "Català",
    "ms": "Bahasa Melayu", "sw": "Kiswahili", "am": "አማርኛ", "ka": "ქართული",
    "hy": "Հայերեն", "ps": "پښتو", "yue": "粵語",
}

# What the dropdown shows first, because most files are in one of these.
_COMMON = ("en", "fr", "ar", "es", "de", "it", "pt", "nl", "ru", "zh", "ja", "hi")


@dataclass(frozen=True)
class Language:
    code: str
    name: str
    native: str | None = None

    @property
    def rtl(self) -> bool:
        return self.code in _RTL

    @property
    def label(self) -> str:
        """``Arabic — العربية``, what a person picks from."""
        return f"{self.name} — {self.native}" if self.native else self.name


LANGUAGES: dict[str, Language] = {
    code: Language(code=code, name=name, native=_NATIVE.get(code))
    for code, name in _NAMES.items()
}

_BY_NAME: dict[str, str] = {}
for _code, _language in LANGUAGES.items():
    _BY_NAME[_language.name.casefold()] = _code
    if _language.native:
        _BY_NAME[_language.native.casefold()] = _code
# Codes people reasonably type that the engines spell differently.
_ALIASES = {
    "jv": "jw", "iw": "he", "in": "id", "nb": "no", "nob": "no",
    "zh-cn": "zh", "zh-tw": "zh", "pt-br": "pt", "en-us": "en", "en-gb": "en",
    "fa-ir": "fa", "farsi": "fa", "mandarin": "zh", "burmese": "my",
    "myanmar": "my", "flemish": "nl",
}


class UnknownLanguage(ValueError):
    """A language was asked for that no engine here knows."""


def is_rtl(code: str | None) -> bool:
    return bool(code) and code.split("-")[0].casefold() in _RTL


def name_of(code: str | None) -> str:
    """A readable name for a code, including ``auto`` and unknown codes."""
    if not code:
        return "Unknown"
    if code == AUTO:
        return "Detect automatically"
    language = LANGUAGES.get(code.casefold())
    return language.name if language else code


def resolve(value: str) -> str:
    """Turn what a person typed into a code: ``French``, ``fr``, ``FR-ca``."""
    text = value.strip()
    if not text:
        raise UnknownLanguage("no language given")
    folded = text.casefold()
    if folded in (AUTO, "automatic", "detect"):
        return AUTO
    if folded in LANGUAGES:
        return folded
    if folded in _ALIASES:
        return _ALIASES[folded]
    if folded in _BY_NAME:
        return _BY_NAME[folded]
    base = folded.split("-")[0].split("_")[0]
    if base in LANGUAGES:
        return base
    if base in _ALIASES:
        return _ALIASES[base]
    raise UnknownLanguage(
        f"{value!r} is not a language Speech2Text knows. "
        f"Use a code such as 'en' or 'ar', a name such as 'French', or 'auto'."
    )


def parse(spec: str | None) -> list[str]:
    """Read a ``--language`` value: empty or ``auto`` means detect.

    Returns ``[]`` for automatic detection, otherwise the resolved codes in the
    order given. More than one code is a hint for engines that accept several;
    the first is what a single-language engine uses.
    """
    if spec is None or not spec.strip():
        return []
    codes: list[str] = []
    for part in spec.replace(";", ",").split(","):
        if not part.strip():
            continue
        code = resolve(part)
        if code == AUTO:
            return []
        if code not in codes:
            codes.append(code)
    return codes


def choices() -> list[Language]:
    """Every language, common ones first and the rest alphabetical."""
    common = [LANGUAGES[c] for c in _COMMON if c in LANGUAGES]
    rest = sorted(
        (lang for code, lang in LANGUAGES.items() if code not in _COMMON),
        key=lambda language: language.name,
    )
    return common + rest
