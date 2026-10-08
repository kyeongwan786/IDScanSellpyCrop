@echo off
setlocal EnableExtensions
cd /d "%~dp0"

echo [1/6] Locating Windows x64 Python 3.12...
if defined IDCARD_PYTHON_X64 if not "%IDCARD_PYTHON_X64%"=="" (
    set "BASE_PY=%IDCARD_PYTHON_X64%"
    if not exist "%BASE_PY%" (
        echo ERROR: IDCARD_PYTHON_X64 does not exist: "%BASE_PY%"
        exit /b 1
    )
    call :CHECK_AMD64 "%BASE_PY%"
    if errorlevel 1 (
        echo ERROR: IDCARD_PYTHON_X64 must point to AMD64 Python. Build stopped.
        exit /b 1
    )
    goto BASE_FOUND
)

set "BASE_PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if exist "%BASE_PY%" (
    call :CHECK_AMD64 "%BASE_PY%"
    if not errorlevel 1 goto BASE_FOUND
    echo The Python at the default path is not AMD64. Trying the x64 Python launcher entry...
)

set "BASE_PY="
set "PY_PATH_FILE=%TEMP%\idcard_cropper_python_%RANDOM%.txt"
py -3.12-64 -c "import sys; print(sys.executable)" > "%PY_PATH_FILE%" 2>nul
if not errorlevel 1 set /p "BASE_PY="<"%PY_PATH_FILE%"
del /q "%PY_PATH_FILE%" >nul 2>&1
if not defined BASE_PY (
    echo ERROR: Could not find Python 3.12 x64.
    echo Install x64 Python 3.12 or repair the py launcher registration.
    exit /b 1
)
call :CHECK_AMD64 "%BASE_PY%"
if errorlevel 1 (
    echo ERROR: The selected base Python is not AMD64. Build stopped.
    exit /b 1
)

:BASE_FOUND
echo Using base Python: "%BASE_PY%"

echo [2/6] Checking the existing virtual environment...
set "VENV_PY=.venv-win\Scripts\python.exe"
if exist "%VENV_PY%" (
    call :CHECK_AMD64 "%VENV_PY%"
    if errorlevel 1 (
        echo Existing .venv-win is not AMD64 or cannot run. Recreating it with x64 Python...
        rmdir /s /q .venv-win
        if exist .venv-win (
            echo ERROR: Could not remove the existing .venv-win folder.
            exit /b 1
        )
    )
)

if not exist "%VENV_PY%" (
    echo Creating AMD64 virtual environment...
    "%BASE_PY%" -m venv .venv-win
    if errorlevel 1 (
        echo ERROR: Failed to create the virtual environment.
        exit /b 1
    )
)

if not exist "%VENV_PY%" (
    echo ERROR: The virtual environment Python was not created.
    exit /b 1
)
call :CHECK_AMD64 "%VENV_PY%"
if errorlevel 1 (
    echo ERROR: .venv-win is not AMD64. Build stopped before installing packages.
    exit /b 1
)

echo [3/6] Installing project and build dependencies in AMD64 environment...
"%VENV_PY%" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"%VENV_PY%" -m pip install -e ".[ocr,gui]" pyinstaller
if errorlevel 1 exit /b 1

echo [4/6] Locating Tesseract OCR...
if not defined TESSERACT_DIR set "TESSERACT_DIR=%ProgramFiles%\Tesseract-OCR"
if not exist "%TESSERACT_DIR%\tesseract.exe" (
    echo ERROR: Tesseract was not found at "%TESSERACT_DIR%".
    echo Install Tesseract OCR or set TESSERACT_DIR to its installation folder, then rerun this file.
    exit /b 1
)

if exist "build\win-tesseract" rmdir /s /q "build\win-tesseract"
mkdir "build\win-tesseract"
if errorlevel 1 exit /b 1
copy /y "%TESSERACT_DIR%\tesseract.exe" "build\win-tesseract\tesseract.exe" >nul
if errorlevel 1 exit /b 1
for %%F in ("%TESSERACT_DIR%\*.dll") do if exist "%%~fF" copy /y "%%~fF" "build\win-tesseract\" >nul
"build\win-tesseract\tesseract.exe" --version
if errorlevel 1 (
    echo ERROR: The staged Tesseract executable could not start.
    exit /b 1
)
"build\win-tesseract\tesseract.exe" --list-langs --tessdata-dir "src\idcard_cropper\data"
if errorlevel 1 (
    echo ERROR: Tesseract could not read the bundled Korean language model.
    exit /b 1
)

echo [5/6] Building one-file Windows x64 GUI executable...
set "PYINSTALLER_DIST=%~dp0dist"
if /I "%~1"=="--github-actions" set "PYINSTALLER_DIST=%~dp0"
"%VENV_PY%" -m PyInstaller --noconfirm --clean --onefile --windowed --name IDCardCropper ^
    --distpath "%PYINSTALLER_DIST%" ^
    --paths src ^
    --collect-all cv2 ^
    --collect-all numpy ^
    --collect-all pytesseract ^
    --hidden-import PySide6.QtPrintSupport ^
    --add-data "src\idcard_cropper\data\kor.traineddata;idcard_cropper\data" ^
    --add-binary "build\win-tesseract;tesseract" ^
    run_gui.py
if errorlevel 1 exit /b 1

echo [6/6] Verifying the generated EXE is Windows AMD64...
if not exist "%PYINSTALLER_DIST%\IDCardCropper.exe" (
    echo ERROR: PyInstaller did not create "%PYINSTALLER_DIST%\IDCardCropper.exe".
    exit /b 1
)
echo ::notice title=Built EXE path::%PYINSTALLER_DIST%\IDCardCropper.exe
"%VENV_PY%" -c "import struct,sys; f=open(sys.argv[1],'rb'); d=f.read(64); assert d[:2]==b'MZ','Not a PE executable'; f.seek(struct.unpack_from('<I',d,60)[0]); assert f.read(4)==b'PE\0\0','Invalid PE signature'; machine=struct.unpack('<H',f.read(2))[0]; print('PE machine:',hex(machine)); sys.exit(0 if machine==0x8664 else 1)" "%PYINSTALLER_DIST%\IDCardCropper.exe"
if errorlevel 1 (
    echo ERROR: The generated EXE is not Windows x64 AMD64. Build failed.
    exit /b 1
)

echo Build complete: %PYINSTALLER_DIST%\IDCardCropper.exe (Windows x64)
exit /b 0

:CHECK_AMD64
set "CHECK_ARCH="
set "ARCH_FILE=%TEMP%\idcard_cropper_arch_%RANDOM%.txt"
"%~1" -c "import platform; print(platform.machine())" > "%ARCH_FILE%" 2>nul
if errorlevel 1 (
    del /q "%ARCH_FILE%" >nul 2>&1
    echo Architecture check failed: the interpreter could not run.
    exit /b 1
)
set /p "CHECK_ARCH="<"%ARCH_FILE%"
del /q "%ARCH_FILE%" >nul 2>&1
echo Architecture check: %CHECK_ARCH%
if /I "%CHECK_ARCH%"=="AMD64" exit /b 0
exit /b 1
