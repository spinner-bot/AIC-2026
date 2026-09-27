@echo off
cd /d "%~dp0"
python -c "import PIL" >nul 2>&1
if errorlevel 1 (
    echo Installing Pillow...
    python -m pip install -r "%~dp0requirements.txt"
    if errorlevel 1 goto failed
)
python "%~dp0clean_dataset.py" %*
if errorlevel 1 goto failed
echo Cleaning finished.
pause
exit /b 0
:failed
echo Cleaning failed. Read the error message above.
pause
exit /b 1
