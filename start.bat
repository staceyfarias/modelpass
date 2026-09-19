@echo off
setlocal
cd /d "%~dp0"

set "MODELPASS_PYTHON=%CD%\.venv\Scripts\python.exe"
set "MODELPASS_URL=http://127.0.0.1:8765/"

if not exist "%MODELPASS_PYTHON%" (
  echo Creating .venv...
  where py >nul 2>&1
  if not errorlevel 1 (
    py -3 -m venv ".venv"
  ) else (
    where python >nul 2>&1
    if errorlevel 1 (
      echo Python 3.11 or newer was not found. Install Python and try again.
      exit /b 1
    )
    python -m venv ".venv"
  )
  if errorlevel 1 (
    echo Could not create the virtual environment.
    exit /b 1
  )
)

"%MODELPASS_PYTHON%" -c "import flask, modelpass" >nul 2>&1
if errorlevel 1 (
  echo Installing modelpass and the Flask account manager...
  "%MODELPASS_PYTHON%" -m pip install -e ".[bench]"
  if errorlevel 1 (
    echo Installation failed.
    exit /b 1
  )
)

echo.
echo modelpass account manager: %MODELPASS_URL%
echo Close this window or press Ctrl+C to stop it.
echo.

if not defined MODELPASS_START_NO_BROWSER (
  start "" powershell.exe -NoProfile -WindowStyle Hidden -Command "$u='%MODELPASS_URL%'; for ($i=0; $i -lt 40; $i++) { try { Invoke-WebRequest -UseBasicParsing $u | Out-Null; Start-Process $u; break } catch { Start-Sleep -Milliseconds 250 } }"
)

"%MODELPASS_PYTHON%" -c "from modelpass.bench.app import serve; serve()"
set "MODELPASS_EXIT=%ERRORLEVEL%"

if not "%MODELPASS_EXIT%"=="0" (
  echo.
  echo The account manager stopped with exit code %MODELPASS_EXIT%.
)
exit /b %MODELPASS_EXIT%
