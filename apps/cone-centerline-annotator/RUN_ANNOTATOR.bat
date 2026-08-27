@echo off
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "ANNOTATOR_PYTHON=.venv\Scripts\python.exe"
) else (
  where py >nul 2>nul
  if %errorlevel%==0 (
    set "ANNOTATOR_PYTHON=py -3"
  ) else (
    set "ANNOTATOR_PYTHON=python"
  )
)

%ANNOTATOR_PYTHON% -c "import numpy, scipy, sklearn, matplotlib, PIL" >nul 2>nul
if not %errorlevel%==0 (
  echo First run: installing the scientific packages in a private environment.
  %ANNOTATOR_PYTHON% -m venv .venv || goto :failed
  set "ANNOTATOR_PYTHON=.venv\Scripts\python.exe"
  %ANNOTATOR_PYTHON% -m pip install --upgrade pip || goto :failed
  if exist "requirements.txt" (
    set "ANNOTATOR_REQUIREMENTS=requirements.txt"
  ) else (
    set "ANNOTATOR_REQUIREMENTS=..\..\requirements.txt"
  )
  %ANNOTATOR_PYTHON% -m pip install -r "%ANNOTATOR_REQUIREMENTS%" || goto :failed
)

%ANNOTATOR_PYTHON% annotator_gui.py
pause
exit /b 0

:failed
echo Installation stopped. Check your internet connection and try again.
pause
exit /b 1
