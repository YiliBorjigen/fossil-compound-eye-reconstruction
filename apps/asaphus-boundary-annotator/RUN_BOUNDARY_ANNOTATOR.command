#!/bin/bash
set -e
cd "$(dirname "$0")"

PYTHON=""
for candidate in python3.12 python3.11 python3.10 python3; do
  if command -v "$candidate" >/dev/null 2>&1; then
    PYTHON="$candidate"
    break
  fi
done

if [ -z "$PYTHON" ]; then
  osascript -e 'display dialog "Python 3.10 or newer is required." buttons {"OK"} default button "OK" with icon stop'
  exit 1
fi

if [ ! -d .venv ]; then
  "$PYTHON" -m venv .venv
fi

.venv/bin/python -m pip install -q -r requirements.txt
.venv/bin/python boundary_annotator.py
