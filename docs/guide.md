# Speech2Text guide

> The same guide is built into the window: press **? Guide** or `F1`.
> That copy is what the tests check, so prefer it. This file is the
> copy for reading on the web.

Every control in the window has a tooltip explaining what it is for.

## Transcribing something
Drop a file onto the window, press **Add files** (`Ctrl+O`), or paste one with
`Ctrl+V`. Then press **Transcribe** (`Ctrl+Return`).

The name of the file does not matter. Speech2Text looks inside it, so a video,
a voice memo, a `.opus` file, a file with the wrong extension and a file with
no extension at all are all handled the same way: if there is sound in it, it
can be transcribed. The picture in a video is ignored; only its audio is read.

A file with no sound track is refused with a reason rather than producing an
empty transcript.

A long recording does not have to be waited out. While it runs, the text so
far is written to `transcript.partial.txt` in the recording's folder, and is
replaced by `transcript.txt` when the run finishes. **Stop** (`Esc`) ends a run
after the current step; what has already finished is kept.

## Choosing the language
**Detect automatically** is the default and is usually right. The recognizer
listens to the opening of the recording and decides.

Naming the language is worth it when you know it. Detection works on the first
part of the audio, so music, noise, or a greeting in another language can send
it the wrong way, and that mistake affects the whole transcript. Pick the
language from the list when the recording is important.

The Standard engine knows 99 languages; `speech2text languages` lists them all.
The Offline engine understands English only.

## Choosing a recognizer
| | when to use it | needs |
|---|---|---|
| **Standard (Whisper)** | almost always | a one-time model download |
| **Offline (PocketSphinx)** | no network at all, English, rough notes | nothing |
| **Cloud** | when accuracy matters more than privacy | an API key |

**Standard** runs on your machine. The first run of each model size downloads
it from huggingface.co; after that it needs no network ever again. Larger
models are more accurate and slower: `tiny` and `base` are for quick notes,
`small` and `medium` for real work, `large-v3` when the words matter.
`small` is the default — the first size good enough to trust on ordinary
speech while still fitting a laptop. `SPEECH2TEXT_MODEL` changes that, and
the window remembers whatever was picked last.

A model can be fetched ahead of time rather than during the first
transcription, which matters on a slow connection or before going offline:

```bash
speech2text models                 # what there is, and what is already here
speech2text models get large-v3    # fetch one now
```

**Where it runs.** A graphics card is used when there is one and it can
actually run the model. A laptop GPU usually holds only the smaller models,
and using one needs NVIDIA's cuBLAS and cuDNN libraries present as well as
the card itself. Neither can be established by asking: a card does not
report how much memory a model will need, and a missing library only shows
itself once recognition starts. So the GPU is tried and the work moves to
the processor if it will not run there — slower, but it finishes, and the
run says that it happened rather than leaving it to be guessed from the
speed. Force one or the other with `--device cuda` or `--device cpu`.

If the GPU is wanted and reports a missing `libcublas` or `libcudnn`, those
libraries are what is absent rather than anything about this application;
NVIDIA publishes them as `nvidia-cublas-cu12` and `nvidia-cudnn-cu12`.

On the processor, `small` is the largest model most laptops are comfortable
with. `large-v3` works but is several times slower than the recording itself.

**Offline** ships its model inside the package, so it works on a machine with
no network. It is English-only and markedly less accurate — it is there so the
application always works, not because it is good.

**Cloud** uploads your audio to a third party. It is never a default and never
runs without a key. While it is selected the badge in the corner changes from
`LOCAL • PRIVATE` to `CLOUD • AUDIO IS UPLOADED`, so the state is never
ambiguous. If a request fails, the recording is transcribed locally instead, so
a network problem never loses it.

## Getting the text out
**Copy text** (`Ctrl+C`) puts the whole transcript on the clipboard. That is
usually all you want.

**Save as…** (`Ctrl+S`) writes it as a file. The formats are:

| | |
|---|---|
| `txt` | plain text |
| `docx` | a Word document |
| `md` | Markdown, with a header saying where it came from |
| `srt`, `vtt` | subtitles, with timings |
| `csv` | one row per segment, for counting or charting |
| `json` | everything, including word timings and confidence |

Timestamps are optional in `txt` and `docx` and always present in subtitles.
A right-to-left transcript is laid out right-to-left in the Word document.

Every bundle on disk already contains `transcript.txt`, so the text is a file
whether or not you export anything.

## Summarizing, and other rewrites
Entirely optional. A transcript is finished and exportable without ever
touching this.

The **After transcribing** box can hand the finished text to a language model:

| task | |
|---|---|
| `summary` | a few paragraphs |
| `key-points` | the main points as a list |
| `actions` | decisions and action items |
| `minutes` | meeting minutes with headings |
| `outline` | the topics, in order |
| `tidy` | the same words with the filler removed |

Or type your own instruction instead of picking a task.

This uploads the **text** (not the audio) to whichever provider you have a key
for. A long recording is sent in parts and the parts are combined, so a
two-hour meeting works. The result is saved next to the transcript as a
Markdown file and has its own copy button.

## Checking the result
Select a finished recording and press **Check it…**.

The window lists every segment with the time it happens at and how sure the
recognizer was. Choosing one plays those few seconds and shows two boxes:
what was heard, and your text. Listen, read, and either fix it or leave it.

This is the fastest way to proofread, because a misheard word almost always
*sounds* wrong before it *looks* wrong. **Play as I move** is on by default,
so moving through segments plays each one: listen, read, fix, next.

**Next to check** (`Ctrl+J`) jumps to the segment the recognizer was least
sure about, anywhere in the recording. That is what turns "check a two-hour
recording" into a finite list — work down the queue and the likeliest
mistakes are seen first, instead of listening to all of it.

| | |
|---|---|
| **Save correction** | store your text for this segment (`Ctrl+S`) |
| **It's correct** | it was flagged but it is right; clears it, changes nothing (`Ctrl+K`) |
| **Undo my change** | put back exactly what was heard |

Use **It's correct** freely. A queue full of false alarms is worse than no
queue, and marking something right is not the same as skipping it: moving on
leaves a segment in the queue for later.

Editing never overwrites what the engine heard. The recognition is kept in
`transcript.raw.txt` exactly as produced, your version goes in
`transcript.txt`, and `transcript.json` holds both — so a correction can
always be compared with the original, or undone.

Checking a long recording is not one sitting. Closing the window remembers
where you were, and opening it again starts there rather than at the top.

The audio is prepared when the window opens, so any format can be played. If
the original recording has been moved or deleted since, the text can still be
corrected; only playback is lost.

## Where the output goes
Each recording gets a folder:

| file | |
|---|---|
| `transcript.txt` | the text, including your corrections |
| `transcript.partial.txt` | the text so far, only while a run is going |
| `transcript.raw.txt` | the original recognition, never modified |
| `transcript.json` | everything: timings, words, confidence, corrections |
| `consumer.json` | a small, stable contract for other applications |
| `project.json` | the job: when it ran, where reviewing got to |

`transcript.json` is the one to keep; everything else is regenerated from it.

`consumer.json` is the integration boundary. Another application reads it
without importing Speech2Text or any speech library.

## API keys
Keys are needed only for the cloud recognizer and for summarizing. Local
transcription never reads one.

```bash
speech2text keys add --provider groq      # prompts, never echoes the key
speech2text keys list                     # shows them masked
```

Keys are looked for in three places, in this order: the environment
(`SPEECH2TEXT_GROQ_API_KEY`, `SPEECH2TEXT_GEMINI_API_KEY`,
`SPEECH2TEXT_OPENAI_API_KEY`, or the vendor's own `GROQ_API_KEY`,
`GEMINI_API_KEY`, `OPENAI_API_KEY`), then `~/.llmkit/keys.json` shared with the
sibling projects, then `~/.speech2text/keys.json`.

A key is never written to a log, an error message, or an artifact.

**Groq** has a generous free tier and runs Whisper server-side, which makes it
a good first choice: <https://console.groq.com/keys>. **Gemini** keys come from
<https://aistudio.google.com/apikey>.

## Settings that live in the environment
| variable | |
|---|---|
| `SPEECH2TEXT_MODEL_CACHE` | where Whisper models are kept |
| `SPEECH2TEXT_MODEL` | which model to use when none is named |
| `SPEECH2TEXT_DEVICE` | `auto`, `cpu` or `cuda` |
| `SPEECH2TEXT_GROQ_API_KEY` | a key for Groq |
| `SPEECH2TEXT_GEMINI_API_KEY` | a key for Gemini |
| `SPEECH2TEXT_OPENAI_API_KEY` | a key for OpenAI |

Point `SPEECH2TEXT_MODEL_CACHE` at a big disk, or at a cache copied from
another machine, and the first run needs no download.

## Keyboard shortcuts
In the main window:

| key | |
|---|---|
| `Ctrl+O` | Add files |
| `Ctrl+V` | Paste a file copied from the file manager |
| `Ctrl+Return` | Transcribe |
| `Ctrl+C` | Copy the text of the selected recording |
| `Ctrl+S` | Save the text as a file |
| `Esc` | Stop the current run |
| `F1` | Open this guide |

While checking a transcript:

| key | |
|---|---|
| `Ctrl+Space` | Play the selected segment, or stop playing |
| `Ctrl+R` | Play it again |
| `Ctrl+S` | Save the correction |
| `Ctrl+K` | Mark the segment correct as it is |
| `Ctrl+Right` | Next segment |
| `Ctrl+Left` | Previous segment |
| `Ctrl+J` | Next segment that needs checking |
