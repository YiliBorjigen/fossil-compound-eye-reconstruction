#!/bin/bash
cd "$(dirname "$0")" || exit 1

VENV_PYTHON_FOR_ANNOTATOR=".venv/bin/python"

if [ -x "$VENV_PYTHON_FOR_ANNOTATOR" ]; then
  PYTHON_FOR_ANNOTATOR="$VENV_PYTHON_FOR_ANNOTATOR"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_FOR_ANNOTATOR="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_FOR_ANNOTATOR="$(command -v python)"
else
  echo "Python 3 is required. Install Python 3, then double-click this file again."
  read -r -p "Press Return to close."
  exit 1
fi

if ! "$PYTHON_FOR_ANNOTATOR" -c "import numpy, scipy, sklearn, matplotlib, PIL" >/dev/null 2>&1; then
  echo "First run: installing the scientific packages in a private environment."
  if [ ! -x "$VENV_PYTHON_FOR_ANNOTATOR" ]; then
    "$PYTHON_FOR_ANNOTATOR" -m venv .venv || exit 1
  fi
  PYTHON_FOR_ANNOTATOR="$VENV_PYTHON_FOR_ANNOTATOR"
  "$PYTHON_FOR_ANNOTATOR" -m pip install --upgrade pip || exit 1
  if [ -f "requirements.txt" ]; then
    REQUIREMENTS_FOR_ANNOTATOR="requirements.txt"
  else
    REQUIREMENTS_FOR_ANNOTATOR="../../requirements.txt"
  fi
  "$PYTHON_FOR_ANNOTATOR" -m pip install -r "$REQUIREMENTS_FOR_ANNOTATOR" || exit 1
fi

"$PYTHON_FOR_ANNOTATOR" annotator_gui.py
