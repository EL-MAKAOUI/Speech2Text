#!/usr/bin/env bash
# Transcribe one recording and show the result, in a single command.
#
#   ./scripts/test-drive.sh /path/to/anything --engine sphinx
#
# Everything it writes goes under artifacts/ in this project, which is ignored.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python="${here}/.venv/bin/python"

if [[ $# -lt 1 ]]; then
    echo "usage: $(basename "$0") <audio-or-video-file> [extra speech2text options]" >&2
    exit 2
fi

if [[ ! -x "${python}" ]]; then
    echo "No virtual environment yet. Create one with:" >&2
    echo "    python3 -m venv .venv && .venv/bin/pip install -e '.[dev,whisper]'" >&2
    exit 1
fi

if ! command -v ffmpeg >/dev/null; then
    echo "ffmpeg is not installed. On Ubuntu: sudo apt install ffmpeg" >&2
    exit 1
fi

source="$1"
shift

output="${here}/artifacts/test-drive"
mkdir -p "${output}"

echo "What this file actually is:"
"${python}" -m speech2text.cli info "${source}"
echo

"${python}" -m speech2text.cli transcribe "${source}" \
    --output "${output}" --format txt --format docx "$@"

echo
echo "Latest result:"
"${python}" -m speech2text.cli list --output "${output}" | head -4
