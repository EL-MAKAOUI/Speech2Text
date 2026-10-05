# Repository instructions

## Accuracy and preservation

- Do not guess when making implementation decisions that depend on current
  external behavior. Verify against primary documentation.
- Preserve what the recognizer produced. Never silently replace recognized
  text with an inferred or corrected version.
- Keep raw recognition, engine metadata, timings, confidence, and human
  corrections independently recoverable from the same artifact.

## Identify media by content, never by name

- Nothing may branch on a file extension. A recording is identified by
  inspecting its streams, so a misnamed or extension-less file behaves
  exactly like a correctly named one.
- A file with no audio is refused with a reason, never transcribed as silence.

## The cloud is optional and explicit

- Local processing is the default. A cloud engine requires an explicit
  choice and a key, and the window must say plainly that audio is uploaded.
- A cloud failure falls back to a local engine so a recording is never lost.
- An API key never reaches a log, an error message, or an artifact.

## Storage

- Keep environments, downloaded models, caches, temporary audio and generated
  artifacts out of the repository and off the system disk.

## Documentation follows the code, in the same change

- Any change to behaviour, defaults, shortcuts or settings updates the in-app
  guide (`src/speech2text/ui/guide_window.py`) and the tooltip of every control
  it affects, in the same commit. Stale guidance is worse than none: it is
  trusted and it is wrong.
- Tooltips say what a control is *for* and what happens as a result, not what
  it is called.
- `docs/guide.md` is generated from the in-app guide by
  `scripts/build-docs.py`. Regenerate it in the same commit.
- `tests/test_guide.py` checks the guide against the code — engines, export
  formats, summary tasks, shortcuts, environment variables. Extend it when
  adding anything a user must know about, so drift fails the build rather
  than going unnoticed.
- The README is for someone deciding whether to use or build on Speech2Text;
  the guide is for someone using it. Measured numbers belong in both.

## Testing

- Tests use real media built with ffmpeg, including a video, a misnamed file
  and a file with no extension. Do not replace them with stubs.
- A recognizer's accuracy is not this project's to assert; the shape,
  ordering and timing of what it returns is.

## Git history

- Preserve the configured Git author identity.
- Use short, imperative, feature-focused commits and keep commits logically
  scoped.
