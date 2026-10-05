# Speech2Text

Speech2Text turns any audio or video recording into text, on your own machine.
Drop in a meeting recording, a voice memo, a lecture video or a file with the
wrong extension; it works out what the file is, reads the speech in it, and
gives you the text to copy or save as Word, plain text, subtitles or JSON.

It is a sibling of DocBridge: DocBridge reads documents, this reads recordings,
and both write the same kind of artifact bundle so other tools can consume
either one.

## Design principles

- Local processing is the default; cloud providers are optional and explicit.
- A file is identified by what is inside it, never by its name or extension.
- Raw recognition is immutable. Human edits are stored as a separate layer.
- Timings, confidence and provenance survive every export.
- Recognizers are replaceable adapters.
- Summarizing is a separate, optional step. The transcript is finished without it.

## What it does

- **Any file.** Audio or video, any container ffmpeg can open, whatever the
  file is called. A `.dat` that is really an MP4, a voice memo with no
  extension, an MP3 with cover art — all handled the same way. A file with no
  sound track is refused with a reason rather than producing an empty result.
- **Any language, or none chosen.** 99 languages, or *Detect automatically*,
  which listens and decides.
- **The text, immediately.** One button copies the whole transcript. **Save
  as…** writes Word (`.docx`), plain text, Markdown, SubRip and WebVTT
  subtitles, CSV or the full JSON.
- **Optionally, something done to it.** A summary, the key points, decisions
  and action items, meeting minutes, an outline, a tidied-up version, or your
  own instruction — handed to a language model only if you ask for it.
- **A review queue.** Segments the recognizer was unsure about are listed
  least-confident first, so checking a long recording is a finite job.

## Getting started

```bash
sudo apt install ffmpeg                  # how any file gets opened
python3 -m venv .venv
.venv/bin/pip install -e '.[dev,whisper,ui]'
.venv/bin/pytest
```

Open the window:

```bash
.venv/bin/speech2text gui
```

Or stay on the command line:

```bash
speech2text transcribe recording.whatever
speech2text transcribe lecture.mp4 --language ar --format docx --format srt
speech2text info mystery-file                 # what is this, really?
speech2text export artifact/ notes.docx
speech2text list                              # everything transcribed so far
```

## Choosing a recognizer

| engine | when to use it | needs | language |
|---|---|---|---|
| `whisper` | almost always — the default | a one-time model download | 99 |
| `sphinx` | no network at all, rough notes | nothing | English |
| `cloud` | accuracy over privacy | an API key | many |

`whisper` runs entirely on your machine through
[faster-whisper](https://github.com/SYSTRAN/faster-whisper). The first use of
each model size downloads it from huggingface.co; everything after that is
offline. Fetch one ahead of time with `speech2text models get <model>`, or
`speech2text models` to see what is already on disk. Bigger models are more accurate and slower — `tiny` and `base` for
quick notes, `small` and `medium` for real work, `large-v3` when the words
matter. Point `SPEECH2TEXT_MODEL_CACHE` at a big disk, or at a cache copied
from another machine, to skip the download.

`sphinx` ships its English model inside the package, so a machine with no
network can still transcribe. It is much less accurate than Whisper and exists
so the application always works, not because it is good.

`cloud` uploads your audio to a third party. It is never a default, never runs
without a key, and the window badge changes to `CLOUD • AUDIO IS UPLOADED`
while it is selected. Any failure falls back to the local engine, so a network
problem never loses a recording.

```bash
export SPEECH2TEXT_GROQ_API_KEY=...
speech2text transcribe interview.m4a --engine cloud --cloud-provider groq
```

**The cloud engine and the summarizer have not been measured against a real
API key.** They are covered by tests against a fake provider — payload shape,
chunking, retry policy, fallback, and that a key never appears in an error —
but no live call has been made. Compare a short run against the local result
before trusting either on something that matters.

## Accuracy

No measured numbers are published here yet, because none have been produced on
this hardware: the environment this was built in could not reach
huggingface.co, so the Whisper models were never downloaded and never run.
Everything else is tested end to end, including real speech through the
`sphinx` engine. Numbers will be added once there is a benchmark to quote
rather than an expectation.

## Where the output goes

Each recording gets a folder:

| file | |
|---|---|
| `transcript.txt` | the text, including your corrections |
| `transcript.raw.txt` | the original recognition, never modified |
| `transcript.json` | everything: timings, words, confidence, corrections |
| `consumer.json` | a small, stable contract for other applications |
| `project.json` | the job: when it ran, where reviewing got to |

`transcript.json` is the one to keep; the rest is regenerated from it.

## Application integration

`consumer.json` is the integration boundary. Another application reads the
corrected text, the segment timings, the source identity and the number of
unresolved reviews without importing Speech2Text or any speech library.

```json
{
  "schema": "1.0",
  "kind": "speech-transcript",
  "source": {"name": "interview.mp4", "sha256": "…", "duration": 742.5},
  "language": "en",
  "text": "…",
  "segments": [{"start": 0.0, "end": 3.2, "text": "…"}],
  "unresolved_reviews": 4
}
```

## Guide

`docs/guide.md` covers transcribing, choosing a language, getting the text out,
summarizing and setting up a key. The **? Guide** button in the window shows
the same text; that copy is generated from the code and checked by the tests,
so it cannot drift. Every control also carries a tooltip.

## Development

A `src/` layout, with the optional recognizers, the window and the cloud
features kept in separate extras so the artifact model and the command line
stay lightweight.

```bash
.venv/bin/pip install -e '.[dev]'     # the model, exports and CLI
.venv/bin/pip install -e '.[dev,whisper,sphinx,ui]'   # everything
.venv/bin/pytest
python3 scripts/build-docs.py         # after changing the in-app guide
```

`AGENTS.md` holds the rules this repository is written to.
