"""Turning a finished transcript into the file someone actually wants.

Every export reads the *current* text, so corrections are always included.
The raw recognition stays available separately and is never what these
produce, with the deliberate exception of ``transcript.raw.txt``.
"""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

from .languages import is_rtl, name_of
from .media import format_duration
from .model import Transcript

FORMATS = ("txt", "md", "srt", "vtt", "json", "docx", "csv")


class ExportError(ValueError):
    """An export could not be produced."""


# ---------------------------------------------------------------------------
# timestamps
# ---------------------------------------------------------------------------


def timestamp(seconds: float, *, millis_separator: str = ",") -> str:
    """``01:02:03,400`` — the form subtitles use."""
    seconds = max(0.0, float(seconds))
    whole = int(seconds)
    milliseconds = int(round((seconds - whole) * 1000))
    if milliseconds == 1000:  # rounding up a whole second
        whole += 1
        milliseconds = 0
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{millis_separator}{milliseconds:03d}"


def clock(seconds: float) -> str:
    """``1:02:03`` — a position in a recording, for reading."""
    return format_duration(seconds)


# ---------------------------------------------------------------------------
# text formats
# ---------------------------------------------------------------------------


def to_text(transcript: Transcript, *, timestamps: bool = False) -> str:
    """The transcript as plain text, one segment per line."""
    lines = []
    for segment in transcript.segments:
        text = transcript.text_of(segment.index).strip()
        if not text:
            continue
        lines.append(f"[{clock(segment.start)}] {text}" if timestamps else text)
    return "\n".join(lines) + ("\n" if lines else "")


def to_markdown(transcript: Transcript, *, timestamps: bool = True) -> str:
    """The transcript with a short header saying where it came from."""
    media = transcript.media
    header = [
        f"# {media.name}",
        "",
        f"- **Duration** {format_duration(transcript.duration)}",
        f"- **Language** {name_of(transcript.language_detected)}",
        f"- **Recognized by** {transcript.engine}"
        + (f" ({transcript.model})" if transcript.model else ""),
        f"- **Words** {transcript.word_count()}",
    ]
    if transcript.corrections:
        header.append(f"- **Corrected segments** {len(transcript.corrected_segment_indexes())}")
    header += ["", "---", ""]
    body = []
    for segment in transcript.segments:
        text = transcript.text_of(segment.index).strip()
        if not text:
            continue
        body.append(f"**[{clock(segment.start)}]** {text}" if timestamps else text)
    return "\n".join(header) + "\n\n".join(body) + "\n"


def to_srt(transcript: Transcript) -> str:
    """SubRip subtitles, numbered from 1 as players expect."""
    blocks = []
    number = 0
    for segment in transcript.segments:
        text = transcript.text_of(segment.index).strip()
        if not text:
            continue
        number += 1
        blocks.append(
            f"{number}\n"
            f"{timestamp(segment.start)} --> {timestamp(max(segment.end, segment.start + 0.1))}\n"
            f"{text}\n"
        )
    return "\n".join(blocks)


def to_vtt(transcript: Transcript) -> str:
    """WebVTT subtitles, for the browser."""
    blocks = ["WEBVTT", ""]
    for segment in transcript.segments:
        text = transcript.text_of(segment.index).strip()
        if not text:
            continue
        start = timestamp(segment.start, millis_separator=".")
        end = timestamp(max(segment.end, segment.start + 0.1), millis_separator=".")
        blocks.append(f"{start} --> {end}\n{text}\n")
    return "\n".join(blocks)


def to_csv(transcript: Transcript) -> str:
    """Segments as a table, for counting or charting elsewhere."""
    import csv
    import io

    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(["index", "start", "end", "confidence", "corrected", "text"])
    corrected = transcript.corrected_segment_indexes()
    for segment in transcript.segments:
        writer.writerow([
            segment.index,
            f"{segment.start:.3f}",
            f"{segment.end:.3f}",
            "" if segment.confidence is None else f"{segment.confidence:.4f}",
            "yes" if segment.index in corrected else "",
            transcript.text_of(segment.index).strip(),
        ])
    return buffer.getvalue()


def to_json(transcript: Transcript) -> str:
    return transcript.to_json()


# ---------------------------------------------------------------------------
# Word
# ---------------------------------------------------------------------------

_CONTENT_TYPES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
<Default Extension="xml" ContentType="application/xml"/>
<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>
<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-package.core-properties+xml"/>
</Types>"""

_ROOT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/core-properties" Target="docProps/core.xml"/>
</Relationships>"""

_DOCUMENT_RELS = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>"""

_STYLES = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr>
<w:rFonts w:ascii="Calibri" w:hAnsi="Calibri" w:cs="Arial"/><w:sz w:val="22"/><w:szCs w:val="22"/>
</w:rPr></w:rPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/>
<w:pPr><w:spacing w:after="160" w:line="276" w:lineRule="auto"/></w:pPr></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/>
<w:pPr><w:spacing w:after="240"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="52"/><w:szCs w:val="52"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/>
<w:pPr><w:outlineLvl w:val="0"/><w:spacing w:before="320" w:after="140"/></w:pPr>
<w:rPr><w:b/><w:sz w:val="32"/><w:szCs w:val="32"/></w:rPr></w:style>
<w:style w:type="character" w:styleId="Timecode"><w:name w:val="Timecode"/>
<w:rPr><w:color w:val="767676"/><w:sz w:val="18"/><w:szCs w:val="18"/></w:rPr></w:style>
</w:styles>"""


def _core_properties(title: str) -> str:
    stamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f"<dc:title>{escape(title)}</dc:title>"
        "<dc:creator>Speech2Text</dc:creator>"
        f'<dcterms:created xsi:type="dcterms:W3CDTF">{stamp}</dcterms:created>'
        "</cp:coreProperties>"
    )


def _paragraph(text: str, *, style: str | None = None, rtl: bool = False,
               prefix: str | None = None) -> str:
    """One Word paragraph, optionally with a grey timecode run in front."""
    properties = []
    if style:
        properties.append(f'<w:pStyle w:val="{style}"/>')
    if rtl:
        # bidi lays the paragraph out right-to-left; rtl marks the run's text.
        properties.append("<w:bidi/>")
        properties.append('<w:jc w:val="right"/>')
    paragraph_properties = f"<w:pPr>{''.join(properties)}</w:pPr>" if properties else ""
    run_properties = "<w:rPr><w:rtl/></w:rPr>" if rtl else ""

    runs = ""
    if prefix:
        runs += (
            '<w:r><w:rPr><w:rStyle w:val="Timecode"/></w:rPr>'
            f'<w:t xml:space="preserve">{escape(prefix)} </w:t></w:r>'
        )
    runs += (
        f"<w:r>{run_properties}"
        f'<w:t xml:space="preserve">{escape(text)}</w:t></w:r>'
    )
    return f"<w:p>{paragraph_properties}{runs}</w:p>"


def _document_xml(paragraphs: Iterable[str], rtl: bool) -> str:
    section = (
        "<w:sectPr>"
        + ('<w:bidi/>' if rtl else "")
        + '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
        "</w:sectPr>"
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{''.join(paragraphs)}{section}</w:body></w:document>"
    )


def write_docx(
    transcript: Transcript,
    destination: str | Path,
    *,
    timestamps: bool = True,
    include_header: bool = True,
) -> Path:
    """Write a Word document, with no dependency beyond the standard library.

    A .docx is a zip of XML parts, so building one directly keeps the export
    available on every install rather than only where python-docx happens to
    be present. Right-to-left transcripts are laid out right-to-left.
    """
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    rtl = is_rtl(transcript.language_detected)

    paragraphs = []
    if include_header:
        paragraphs.append(_paragraph(transcript.media.name, style="Title", rtl=rtl))
        summary = (
            f"{format_duration(transcript.duration)} · "
            f"{name_of(transcript.language_detected)} · "
            f"{transcript.word_count()} words · "
            f"recognized by {transcript.engine}"
        )
        paragraphs.append(_paragraph(summary, rtl=rtl))

    wrote_any = False
    for segment in transcript.segments:
        text = transcript.text_of(segment.index).strip()
        if not text:
            continue
        wrote_any = True
        paragraphs.append(
            _paragraph(
                text,
                rtl=rtl,
                prefix=f"[{clock(segment.start)}]" if timestamps else None,
            )
        )
    if not wrote_any:
        paragraphs.append(_paragraph("(no speech was recognized)", rtl=rtl))

    # Deterministic timestamps keep the file reproducible between runs.
    info = (1980, 1, 1, 0, 0, 0)
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, content in (
            ("[Content_Types].xml", _CONTENT_TYPES),
            ("_rels/.rels", _ROOT_RELS),
            ("docProps/core.xml", _core_properties(transcript.media.name)),
            ("word/_rels/document.xml.rels", _DOCUMENT_RELS),
            ("word/styles.xml", _STYLES),
            ("word/document.xml", _document_xml(paragraphs, rtl)),
        ):
            item = zipfile.ZipInfo(name, date_time=info)
            item.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(item, content)
    return target


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------


def render(transcript: Transcript, fmt: str, *, timestamps: bool | None = None) -> str:
    """Produce a text format. ``docx`` is binary and has its own function."""
    kind = fmt.strip().lower().lstrip(".")
    if kind == "txt":
        return to_text(transcript, timestamps=bool(timestamps))
    if kind in ("md", "markdown"):
        return to_markdown(transcript, timestamps=True if timestamps is None else timestamps)
    if kind == "srt":
        return to_srt(transcript)
    if kind == "vtt":
        return to_vtt(transcript)
    if kind == "csv":
        return to_csv(transcript)
    if kind == "json":
        return to_json(transcript)
    if kind == "docx":
        raise ExportError("docx is a binary format; use write_docx()")
    raise ExportError(f"unknown format {fmt!r}. Choose one of: {', '.join(FORMATS)}")


def write(
    transcript: Transcript,
    destination: str | Path,
    fmt: str | None = None,
    *,
    timestamps: bool | None = None,
) -> Path:
    """Write any supported format, taking the format from the suffix if given."""
    target = Path(destination)
    kind = (fmt or target.suffix.lstrip(".") or "txt").lower()
    if kind == "docx":
        return write_docx(
            transcript, target, timestamps=True if timestamps is None else timestamps
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(render(transcript, kind, timestamps=timestamps), encoding="utf-8")
    return target
