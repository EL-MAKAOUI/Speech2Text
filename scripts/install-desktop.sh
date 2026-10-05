#!/usr/bin/env bash
# Put Speech2Text in Ubuntu's applications menu and its "Open With" list,
# so a recording can be sent to it straight from the file manager.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="${here}/.venv/bin/speech2text-gui"
applications="${HOME}/.local/share/applications"
entry="${applications}/speech2text.desktop"

if [[ ! -x "${launcher}" ]]; then
    echo "The window is not installed yet. Run:" >&2
    echo "    .venv/bin/pip install -e '.[ui,whisper]'" >&2
    exit 1
fi

mkdir -p "${applications}"
cat > "${entry}" <<DESKTOP
[Desktop Entry]
Type=Application
Name=Speech2Text
Comment=Turn any audio or video recording into text
Exec=${launcher} %F
Terminal=false
Categories=AudioVideo;Audio;Utility;
MimeType=audio/*;video/*;
StartupNotify=true
DESKTOP

chmod +x "${entry}"
if command -v update-desktop-database >/dev/null; then
    update-desktop-database "${applications}" || true
fi

echo "Installed ${entry}"
echo "Search for Speech2Text in the applications overview, or right-click a"
echo "recording and choose Open With."
