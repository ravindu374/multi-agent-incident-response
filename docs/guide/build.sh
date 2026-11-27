#!/usr/bin/env bash
# Rebuild the project guide PDF (Git Bash on Windows):  bash docs/guide/build.sh
# 1. redraws all 9 figures (SVG + PNG) from docs/figures/make_figures.py
# 2. prints docs/guide/project-guide.html to docs/a2a-oncall_Project_Guide.pdf with headless Chrome
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
PY="$ROOT/.venv/Scripts/python.exe"
CHROME="/c/Program Files/Google/Chrome/Application/chrome.exe"
# an isolated profile, so Chrome runs truly headless even when your own Chrome is open
PROFILE="${TEMP:-/tmp}/a2a-guide-chrome-profile"

"$PY" "$ROOT/docs/figures/make_figures.py" --png
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --user-data-dir="$PROFILE" \
  --print-to-pdf="$(cd "$ROOT/docs" && pwd -W)/a2a-oncall_Project_Guide.pdf" \
  "file:///$(cd "$HERE" && pwd -W)/project-guide.html" 2>&1 | grep -v external_registry || true
echo "done: docs/a2a-oncall_Project_Guide.pdf"
