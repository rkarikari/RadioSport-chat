@echo off
setlocal EnableDelayedExpansion

:: Initialize logging
set "LOG_FILE=%~dp0setup.log"
echo === Setup Started at %DATE% === > "%LOG_FILE%"
echo === Setup Started at %DATE% ===

:: Sanitize TIME for logging
call :SanitizeTimestamp
echo Sanitized TIMESTAMP is: %TIMESTAMP% >> "%LOG_FILE%" 2>&1
echo Sanitized TIMESTAMP is: %TIMESTAMP%

:: Validate LOG_FILE
echo Validating LOG_FILE...
echo Validating LOG_FILE... >> "%LOG_FILE%" 2>&1
if not defined LOG_FILE (
    set "LOG_FILE=%TEMP%\setup.log"
    echo Warning: LOG_FILE was undefined. Set to %LOG_FILE%.
    echo Warning: LOG_FILE was undefined. Set to %LOG_FILE%. >> "%LOG_FILE%" 2>&1
)
echo LOG_FILE is set to: %LOG_FILE% >> "%LOG_FILE%" 2>&1
dir "%~dp0" >nul 2>&1
if !ERRORLEVEL! neq 0 (
    set "LOG_FILE=%TEMP%\setup.log"
    echo Warning: Directory %~dp0 is not writable. Using fallback LOG_FILE: %LOG_FILE%.
    echo Warning: Directory %~dp0 is not writable. Using fallback LOG_FILE: %LOG_FILE%. >> "%LOG_FILE%" 2>&1
)
echo LOG_FILE validation completed. >> "%LOG_FILE%" 2>&1
echo LOG_FILE validation completed.

:: Check for administrative privileges
echo Starting Administrative Privileges Check...
echo Starting Administrative Privileges Check... >> "%LOG_FILE%" 2>&1
net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: This script requires administrative privileges. Run as Administrator and press any key to exit...
    echo Error: This script requires administrative privileges. Run as Administrator and press any key to exit... >> "%LOG_FILE%" 2>&1
    pause >nul
    exit /b 1
)
echo [%TIMESTAMP%] Administrative privileges confirmed. >> "%LOG_FILE%" 2>&1
echo Administrative privileges check completed.
echo Administrative privileges check completed. >> "%LOG_FILE%" 2>&1

:: Check for RadioSportChat installation
echo Starting RadioSportChat Installation Check...
echo Starting RadioSportChat Installation Check... >> "%LOG_FILE%" 2>&1
set "SCRIPT_DIR=%~dp0"
set "INSTALLER_FOUND=0"
set "APP_ALREADY_INSTALLED=0"
set "SYSTEM_INSTALL_DIR=C:\Program Files\Radiosport\RadioSportChat"
set "USER_INSTALL_DIR1=%LOCALAPPDATA%\Programs\RadioSportChat"
set "USER_INSTALL_DIR2=%LOCALAPPDATA%\Programs\Radiosport\RadioSportChat"
set "USER_INSTALL_DIR3=%LOCALAPPDATA%\Programs\Unknown Developer\RadioSportChat"
for %%D in (
    "%SYSTEM_INSTALL_DIR%"
    "%USER_INSTALL_DIR1%"
    "%USER_INSTALL_DIR2%"
    "%USER_INSTALL_DIR3%"
) do (
    if exist "%%D\RadioSportChat.exe" (
        echo [%TIMESTAMP%] RadioSportChat already installed at %%D. Skipping MSI installation... >> "%LOG_FILE%" 2>&1
        set "APP_ALREADY_INSTALLED=1"
    )
)
if !APP_ALREADY_INSTALLED! equ 0 (
    echo [%TIMESTAMP%] Checking for RadioSportChat installer in %SCRIPT_DIR%... >> "%LOG_FILE%" 2>&1
    for %%f in ("%SCRIPT_DIR%RadioSportChat-*.msi") do (
        echo [%TIMESTAMP%] Found installer: %%f >> "%LOG_FILE%" 2>&1
        echo [%TIMESTAMP%] Starting RadioSportChat installation. Please complete the installation wizard... >> "%LOG_FILE%" 2>&1
        echo Starting RadioSportChat installation. Please complete the installation wizard...
        set "START_TIME=%TIME%"
        start /wait msiexec /i "%%f" /log "%SCRIPT_DIR%radiosportchat_install.log"
        call :SanitizeTimestamp
        set "END_TIME=%TIMESTAMP%"
        if !ERRORLEVEL! equ 0 (
            echo [%TIMESTAMP%] RadioSportChat installation completed successfully. >> "%LOG_FILE%" 2>&1
        ) else (
            echo Error: RadioSportChat installation may have failed. Check %SCRIPT_DIR%radiosportchat_install.log for details.
            echo [%TIMESTAMP%] Error: RadioSportChat installation may have failed. Check %SCRIPT_DIR%radiosportchat_install.log for details. >> "%LOG_FILE%" 2>&1
        )
        echo Press any key to continue with setup...
        pause >nul
        set "INSTALLER_FOUND=1"
    )
    if !INSTALLER_FOUND! equ 0 (
        echo Error: No RadioSportChat installer found in %SCRIPT_DIR%. Press any key to exit...
        echo [%TIMESTAMP%] Error: No RadioSportChat installer found in %SCRIPT_DIR%. Press any key to exit... >> "%LOG_FILE%" 2>&1
        pause >nul
        exit /b 1
    )
)
echo RadioSportChat installation check completed.
echo [%TIMESTAMP%] RadioSportChat installation check completed. >> "%LOG_FILE%" 2>&1

:: Set up virtual environment directory
echo Starting Virtual Environment Directory Setup...
echo Starting Virtual Environment Directory Setup... >> "%LOG_FILE%" 2>&1
set "VENV_DIR=C:\venv"
if not exist "!VENV_DIR!" (
    echo [%TIMESTAMP%] Creating virtual environment directory at !VENV_DIR!... >> "%LOG_FILE%" 2>&1
    mkdir "!VENV_DIR!"
    if !ERRORLEVEL! neq 0 (
        echo Error: Failed to create !VENV_DIR!. Check permissions and press any key to exit...
        echo [%TIMESTAMP%] Error: Failed to create !VENV_DIR!. Check permissions and press any key to exit... >> "%LOG_FILE%" 2>&1
        pause >nul
        exit /b 1
    )
)
echo [%TIMESTAMP%] Setting permissions for !VENV_DIR!... >> "%LOG_FILE%" 2>&1
icacls "!VENV_DIR!" /grant "Authenticated Users:(OI)(CI)M" /T
if !ERRORLEVEL! neq 0 (
    echo Warning: Failed to set permissions for !VENV_DIR!. Manually grant 'Authenticated Users' modify permissions.
    echo [%TIMESTAMP%] Warning: Failed to set permissions for !VENV_DIR!. Manually grant 'Authenticated Users' modify permissions. >> "%LOG_FILE%" 2>&1
)
echo Virtual environment directory setup completed.
echo [%TIMESTAMP%] Virtual environment directory setup completed. >> "%LOG_FILE%" 2>&1

:: Check for Python 3.11
echo Starting Python 3.11 Installation Check...
echo Starting Python 3.11 Installation Check... >> "%LOG_FILE%" 2>&1
set "PYTHON_INSTALLED=0"
python --version 2>nul | findstr "3.11" >nul
if !ERRORLEVEL! equ 0 (
    echo [%TIMESTAMP%] Python 3.11 found in system PATH. Verifying location... >> "%LOG_FILE%" 2>&1
    for /f "tokens=2 delims==" %%a in ('where python ^| findstr /C:"python.exe"') do (
        if /i "%%a"=="C:\Python\python.exe" (
            set "PYTHON_INSTALLED=1"
            echo [%TIMESTAMP%] Python 3.11 already installed at C:\Python. >> "%LOG_FILE%" 2>&1
        ) else (
            echo Warning: Python 3.11 found at %%a, but expected at C:\Python. Proceeding with installation...
            echo [%TIMESTAMP%] Warning: Python 3.11 found at %%a, but expected at C:\Python. Proceeding with installation... >> "%LOG_FILE%" 2>&1
        )
    )
)
if !PYTHON_INSTALLED! equ 0 (
    if not exist "C:\Python\" (
        echo [%TIMESTAMP%] Creating C:\Python directory... >> "%LOG_FILE%" 2>&1
        mkdir "C:\Python" 2>nul
        if !ERRORLEVEL! neq 0 (
            echo Error: Cannot create C:\Python. Check permissions and press any key to exit...
            echo [%TIMESTAMP%] Error: Cannot create C:\Python. Check permissions and press any key to exit... >> "%LOG_FILE%" 2>&1
            pause >nul
            exit /b 1
        )
    ) else (
        echo [%TIMESTAMP%] C:\Python directory exists. Verifying contents... >> "%LOG_FILE%" 2>&1
        if exist "C:\Python\python.exe" (
            C:\Python\python.exe --version 2>nul | findstr "3.11" >nul
            if !ERRORLEVEL! equ 0 (
                echo [%TIMESTAMP%] Python 3.11 verified at C:\Python\python.exe. Skipping installation... >> "%LOG_FILE%" 2>&1
                set "PYTHON_INSTALLED=1"
            ) else (
                echo [%TIMESTAMP%] Python at C:\Python\python.exe is not version 3.11. Proceeding with installation... >> "%LOG_FILE%" 2>&1
            )
        ) else (
            echo [%TIMESTAMP%] No python.exe found in C:\Python. Proceeding with installation... >> "%LOG_FILE%" 2>&1
        )
    )
)
if !PYTHON_INSTALLED! equ 0 (
    echo [%TIMESTAMP%] Checking for Python installer... >> "%LOG_FILE%" 2>&1
    if not exist "%SCRIPT_DIR%python-3.11.9-amd64.exe" (
        echo Error: Python installer not found at %SCRIPT_DIR%python-3.11.9-amd64.exe. Press any key to exit...
        echo [%TIMESTAMP%] Error: Python installer not found at %SCRIPT_DIR%python-3.11.9-amd64.exe. Press any key to exit... >> "%LOG_FILE%" 2>&1
        pause >nul
        exit /b 1
    )
    echo [%TIMESTAMP%] Installing Python 3.11 to C:\Python... >> "%LOG_FILE%" 2>&1
    set "START_TIME=%TIME%"
    start /wait %SCRIPT_DIR%python-3.11.9-amd64.exe /quiet InstallAllUsers=1 TargetDir=C:\Python PrependPath=1
    call :SanitizeTimestamp
    set "END_TIME=%TIMESTAMP%"
    if !ERRORLEVEL! neq 0 (
        echo Error: Failed to install Python 3.11. Check installer logs and press any key to exit...
        echo [%TIMESTAMP%] Error: Failed to install Python 3.11. Check installer logs and press any key to exit... >> "%LOG_FILE%" 2>&1
        pause >nul
        exit /b 1
    )
    if not exist "C:\Python\python.exe" (
        echo Error: Python 3.11 installation verification failed. Press any key to exit...
        echo [%TIMESTAMP%] Error: Python 3.11 installation verification failed. Press any key to exit... >> "%LOG_FILE%" 2>&1
        pause >nul
        exit /b 1
    )
    C:\Python\python.exe --version 2>nul | findstr "3.11" >nul
    if !ERRORLEVEL! equ 0 (
        echo [%TIMESTAMP%] Python 3.11 installed successfully to C:\Python. >> "%LOG_FILE%" 2>&1
    ) else (
        echo Error: Python 3.11 version verification failed. Press any key to exit...
        echo [%TIMESTAMP%] Error: Python 3.11 version verification failed. Press any key to exit... >> "%LOG_FILE%" 2>&1
        pause >nul
        exit /b 1
    )
)
echo Python 3.11 installation check completed.
echo [%TIMESTAMP%] Python 3.11 installation check completed. >> "%LOG_FILE%" 2>&1

:: Create and set up virtual environment
echo Starting Virtual Environment Setup...
echo Starting Virtual Environment Setup... >> "%LOG_FILE%" 2>&1
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [%TIMESTAMP%] Virtual environment exists at %VENV_DIR%. Verifying functionality... >> "%LOG_FILE%" 2>&1
    call "%VENV_DIR%\Scripts\activate.bat"
    pip --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo [%TIMESTAMP%] Virtual environment is functional. >> "%LOG_FILE%" 2>&1
    ) else (
        echo Warning: Virtual environment at %VENV_DIR% is not functional. Recreating...
        echo [%TIMESTAMP%] Warning: Virtual environment at %VENV_DIR% is not functional. Recreating... >> "%LOG_FILE%" 2>&1
        rd /S /Q "%VENV_DIR%" 2>nul
        C:\Python\python.exe -m venv "%VENV_DIR%"
        if !ERRORLEVEL! neq 0 (
            echo Error: Failed to recreate virtual environment. Check permissions and press any key to exit...
            echo [%TIMESTAMP%] Error: Failed to recreate virtual environment. Check permissions and press any key to exit... >> "%LOG_FILE%" 2>&1
            call "%VENV_DIR%\Scripts\deactivate.bat" 2>nul
            pause >nul
            exit /b 1
        )
    )
    call "%VENV_DIR%\Scripts\deactivate.bat"
) else (
    echo [%TIMESTAMP%] Creating virtual environment at %VENV_DIR%... >> "%LOG_FILE%" 2>&1
    set "START_TIME=%TIME%"
    C:\Python\python.exe -m venv "%VENV_DIR%"
    call :SanitizeTimestamp
    set "END_TIME=%TIMESTAMP%"
    if !ERRORLEVEL! neq 0 (
        echo Error: Failed to create virtual environment. Check Python installation and press any key to exit...
        echo [%TIMESTAMP%] Error: Failed to create virtual environment. Check Python installation and press any key to exit... >> "%LOG_FILE%" 2>&1
        pause >nul
        exit /b 1
    )
)
echo Virtual environment setup completed.
echo [%TIMESTAMP%] Virtual environment setup completed. >> "%LOG_FILE%" 2>&1

:: Install pip packages
echo Starting Pip Packages Installation...
echo Starting Pip Packages Installation... >> "%LOG_FILE%" 2>&1
if not exist "%SCRIPT_DIR%venv" (
    echo Error: Package directory %SCRIPT_DIR%venv not found. Press any key to exit...
    echo [%TIMESTAMP%] Error: Package directory %SCRIPT_DIR%venv not found. Press any key to exit... >> "%LOG_FILE%" 2>&1
    pause >nul
    exit /b 1
)
if not exist "%SCRIPT_DIR%venv\requirements.txt" (
    echo Error: requirements.txt not found at %SCRIPT_DIR%venv\requirements.txt. Press any key to exit...
    echo [%TIMESTAMP%] Error: requirements.txt not found at %SCRIPT_DIR%venv\requirements.txt. Press any key to exit... >> "%LOG_FILE%" 2>&1
    pause >nul
    exit /b 1
)
set "PACKAGE_FOUND=0"
for %%f in ("%SCRIPT_DIR%venv\streamlit*.whl" "%SCRIPT_DIR%venv\streamlit*.tar.gz") do (
    set "PACKAGE_FOUND=1"
)
if !PACKAGE_FOUND! equ 0 (
    echo Warning: No streamlit package files found in %SCRIPT_DIR%venv. Installation may fail.
    echo [%TIMESTAMP%] Warning: No streamlit package files found in %SCRIPT_DIR%venv. Installation may fail. >> "%LOG_FILE%" 2>&1
)
echo [%TIMESTAMP%] Installing pip packages from %SCRIPT_DIR%venv... >> "%LOG_FILE%" 2>&1
call "%VENV_DIR%\Scripts\activate.bat"
set "START_TIME=%TIME%"
pip install --no-index --find-links=%SCRIPT_DIR%venv -r %SCRIPT_DIR%venv\requirements.txt >nul
if !ERRORLEVEL! neq 0 (
    echo Error: Failed to install pip packages. Check %LOG_FILE% for details and press any key to exit...
    echo [%TIMESTAMP%] Error: Failed to install pip packages. Check %LOG_FILE% for details... >> "%LOG_FILE%" 2>&1
    call "%VENV_DIR%\Scripts\deactivate.bat"
    pause >nul
    exit /b 1
)
call :SanitizeTimestamp
set "END_TIME=%TIMESTAMP%"
set "PACKAGES_OK=1"
for %%P in (streamlit langchain-ollama) do (
    pip show %%P >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo Warning: Package %%P not found after installation.
        echo [%TIMESTAMP%] Warning: Package %%P not found after installation. >> "%LOG_FILE%" 2>&1
        set "PACKAGES_OK=0"
    )
)
call "%VENV_DIR%\Scripts\deactivate.bat"
if !PACKAGES_OK! equ 1 (
    echo [%TIMESTAMP%] Pip packages installed successfully. >> "%LOG_FILE%" 2>&1
) else (
    echo Warning: Some packages may not have installed correctly. Check requirements.txt.
    echo [%TIMESTAMP%] Warning: Some packages may not have installed correctly. Check requirements.txt. >> "%LOG_FILE%" 2>&1
)
echo Pip packages installation completed.
echo [%TIMESTAMP%] Pip packages installation completed. >> "%LOG_FILE%" 2>&1

:: Check and install Ollama
echo Starting Ollama Installation...
echo Starting Ollama Installation... >> "%LOG_FILE%" 2>&1
set "OLLAMA_INSTALLED=0"
set "OLLAMA_PATH=C:\Program Files\Ollama"
set "OLLAMA_ALT_PATH=C:\Users\%USERNAME%\AppData\Local\Programs\Ollama"
echo [%TIMESTAMP%] Checking for existing Ollama installation with ollama list... >> "%LOG_FILE%" 2>&1
ollama list >nul 2>&1
if !ERRORLEVEL! equ 0 (
    set "OLLAMA_INSTALLED=1"
    echo [%TIMESTAMP%] Ollama already installed and functional. >> "%LOG_FILE%" 2>&1
    :: Check PATH for existing installation
    if exist "!OLLAMA_PATH!\ollama.exe" (
        echo [%TIMESTAMP%] Ollama executable verified at !OLLAMA_PATH!\ollama.exe. >> "%LOG_FILE%" 2>&1
    ) else if exist "!OLLAMA_ALT_PATH!\ollama.exe" (
        echo [%TIMESTAMP%] Ollama executable verified at !OLLAMA_ALT_PATH!\ollama.exe. >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%TIMESTAMP%] Warning: No Ollama executable found at !OLLAMA_PATH! or !OLLAMA_ALT_PATH!. Skipping PATH update for existing installation. >> "%LOG_FILE%" 2>&1
    )
) else (
    echo [%TIMESTAMP%] Ollama not detected. Checking installation paths... >> "%LOG_FILE%" 2>&1
    if exist "!OLLAMA_PATH!\ollama.exe" (
        echo [%TIMESTAMP%] Ollama executable found at !OLLAMA_PATH!, but ollama list failed. Verifying functionality... >> "%LOG_FILE%" 2>&1
        "!OLLAMA_PATH!\ollama.exe" --version >nul 2>&1
        if !ERRORLEVEL! equ 0 (
            set "OLLAMA_INSTALLED=1"
            echo [%TIMESTAMP%] Ollama is installed at !OLLAMA_PATH! and functional. >> "%LOG_FILE%" 2>&1
        ) else (
            echo [%TIMESTAMP%] Warning: Ollama executable at !OLLAMA_PATH! is not functional. Checking alternate path... >> "%LOG_FILE%" 2>&1
        )
    )
    if !OLLAMA_INSTALLED! equ 0 (
        if exist "!OLLAMA_ALT_PATH!\ollama.exe" (
            echo [%TIMESTAMP%] Ollama executable found at !OLLAMA_ALT_PATH!, but ollama list failed. Verifying functionality... >> "%LOG_FILE%" 2>&1
            "!OLLAMA_ALT_PATH!\ollama.exe" --version >nul 2>&1
            if !ERRORLEVEL! equ 0 (
                set "OLLAMA_INSTALLED=1"
                echo [%TIMESTAMP%] Ollama is installed at !OLLAMA_ALT_PATH! and functional. >> "%LOG_FILE%" 2>&1
            ) else (
                echo [%TIMESTAMP%] Warning: Ollama executable at !OLLAMA_ALT_PATH! is not functional. Reinstalling... >> "%LOG_FILE%" 2>&1
            )
        )
    )
)
if !OLLAMA_INSTALLED! equ 0 (
    echo [%TIMESTAMP%] Checking for local Ollama installer... >> "%LOG_FILE%" 2>&1
    if exist "%SCRIPT_DIR%OllamaSetup.exe" (
        echo [%TIMESTAMP%] Using local copy of OllamaSetup.exe... >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%TIMESTAMP%] Downloading OllamaSetup.exe... >> "%LOG_FILE%" 2>&1
        set "START_TIME=%TIME%"
        curl -L https://ollama.com/download/OllamaSetup.exe -o %SCRIPT_DIR%OllamaSetup.exe
        call :SanitizeTimestamp
        set "END_TIME=%TIMESTAMP%"
        if !ERRORLEVEL! neq 0 (
            echo Error: Failed to download OllamaSetup.exe. Check network and press any key to exit...
            echo [%TIMESTAMP%] Error: Failed to download OllamaSetup.exe. Check network and press any key to exit... >> "%LOG_FILE%" 2>&1
            pause >nul
            exit /b 1
        )
    )
    echo [%TIMESTAMP%] Installing Ollama... >> "%LOG_FILE%" 2>&1
    set "START_TIME=%TIME%"
    start /wait %SCRIPT_DIR%OllamaSetup.exe
    call :SanitizeTimestamp
    set "END_TIME=%TIMESTAMP%"
    if !ERRORLEVEL! neq 0 (
        echo Warning: Ollama installation may have failed. Check installer logs.
        echo [%TIMESTAMP%] Warning: Ollama installation may have failed. Check installer logs. >> "%LOG_FILE%" 2>&1
    )
    if not exist "!OLLAMA_PATH!\ollama.exe" (
        if not exist "!OLLAMA_ALT_PATH!\ollama.exe" (
            echo Error: Ollama installation verification failed. Press any key to exit...
            echo [%TIMESTAMP%] Error: Ollama installation verification failed. Press any key to exit... >> "%LOG_FILE%" 2>&1
            pause >nul
            exit /b 1
        )
    )
    echo [%TIMESTAMP%] Ollama installed successfully to !OLLAMA_PATH! or !OLLAMA_ALT_PATH!. >> "%LOG_FILE%" 2>&1
)
echo Ollama installation completed.
echo [%TIMESTAMP%] Ollama installation completed. >> "%LOG_FILE%" 2>&1

:: Update system PATH for Ollama if ollama.exe exists
echo Starting Ollama PATH Update...
echo Starting Ollama PATH Update... >> "%LOG_FILE%" 2>&1
set "PATH_UPDATED=0"
if exist "!OLLAMA_PATH!\ollama.exe" (
    echo [%TIMESTAMP%] Ollama executable verified at !OLLAMA_PATH!\ollama.exe. >> "%LOG_FILE%" 2>&1
    echo [%TIMESTAMP%] Checking if !OLLAMA_PATH! is in system PATH... >> "%LOG_FILE%" 2>&1
    echo %PATH% | findstr /C:"!OLLAMA_PATH!" >nul
    if !ERRORLEVEL! equ 0 (
        echo [%TIMESTAMP%] !OLLAMA_PATH! already in PATH. >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%TIMESTAMP%] Adding !OLLAMA_PATH! to system PATH... >> "%LOG_FILE%" 2>&1
        setx PATH "%PATH%;!OLLAMA_PATH!" /M
        if !ERRORLEVEL! neq 0 (
            echo Warning: Failed to update system PATH with !OLLAMA_PATH!. Manually run: setx PATH "%%PATH%%;!OLLAMA_PATH!" /M
            echo [%TIMESTAMP%] Warning: Failed to update system PATH with !OLLAMA_PATH!. Manually run: setx PATH "%%PATH%%;!OLLAMA_PATH!" /M >> "%LOG_FILE%" 2>&1
        ) else (
            echo [%TIMESTAMP%] !OLLAMA_PATH! added to system PATH successfully. >> "%LOG_FILE%" 2>&1
            set "PATH_UPDATED=1"
        )
    )
)
if exist "!OLLAMA_ALT_PATH!\ollama.exe" (
    echo [%TIMESTAMP%] Ollama executable verified at !OLLAMA_ALT_PATH!\ollama.exe. >> "%LOG_FILE%" 2>&1
    echo [%TIMESTAMP%] Checking if !OLLAMA_ALT_PATH! is in system PATH... >> "%LOG_FILE%" 2>&1
    echo %PATH% | findstr /C:"!OLLAMA_ALT_PATH!" >nul
    if !ERRORLEVEL! equ 0 (
        echo [%TIMESTAMP%] !OLLAMA_ALT_PATH! already in PATH. >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%TIMESTAMP%] Adding !OLLAMA_ALT_PATH! to system PATH... >> "%LOG_FILE%" 2>&1
        setx PATH "%PATH%;!OLLAMA_ALT_PATH!" /M
        if !ERRORLEVEL! neq 0 (
            echo Warning: Failed to update system PATH with !OLLAMA_ALT_PATH!. Manually run: setx PATH "%%PATH%%;!OLLAMA_ALT_PATH!" /M
            echo [%TIMESTAMP%] Warning: Failed to update system PATH with !OLLAMA_ALT_PATH!. Manually run: setx PATH "%%PATH%%;!OLLAMA_ALT_PATH!" /M >> "%LOG_FILE%" 2>&1
        ) else (
            echo [%TIMESTAMP%] !OLLAMA_ALT_PATH! added to system PATH successfully. >> "%LOG_FILE%" 2>&1
            set "PATH_UPDATED=1"
        )
    )
)
if !PATH_UPDATED! equ 0 (
    if not exist "!OLLAMA_PATH!\ollama.exe" (
        if not exist "!OLLAMA_ALT_PATH!\ollama.exe" (
            echo Warning: Ollama executable not found at !OLLAMA_PATH! or !OLLAMA_ALT_PATH!. Skipping PATH update.
            echo [%TIMESTAMP%] Warning: Ollama executable not found at !OLLAMA_PATH! or !OLLAMA_ALT_PATH!. Skipping PATH update. >> "%LOG_FILE%" 2>&1
        )
    )
)
echo Ollama PATH update completed.
echo [%TIMESTAMP%] Ollama PATH update completed. >> "%LOG_FILE%" 2>&1

:: Check and handle Ollama models
echo Starting Ollama Models Check...
echo [%TIMESTAMP%] Checking models... >> "%LOG_FILE%" 2>&1
set "MODELS_MISSING=0"
for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
    ollama list | findstr "%%M" >nul || set "MODELS_MISSING=1"
)
if !MODELS_MISSING! equ 0 (
    echo [%TIMESTAMP%] Required models granite3.3:2b, nomic-embed-text:latest, qwen3:4b, and qwen3:1.7b are already installed. >> "%LOG_FILE%" 2>&1
) else (
    call :HandleOllamaModels
)
echo Ollama models check completed.
echo [%TIMESTAMP%] Ollama models check completed. >> "%LOG_FILE%" 2>&1

:: Check and start Ollama server
echo Starting Ollama Server Check...
echo [%TIMESTAMP%] Checking Ollama server... >> "%LOG_FILE%" 2>&1
tasklist | findstr /I "ollama" >nul 2>&1
if !ERRORLEVEL! equ 0 (
    echo [%TIMESTAMP%] Ollama server is running. >> "%LOG_FILE%" 2>&1
) else (
    echo [%TIMESTAMP%] Starting Ollama server... >> "%LOG_FILE%" 2>&1
    ollama serve >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo [%TIMESTAMP%] Ollama server started. >> "%LOG_FILE%" 2>&1
    ) else (
        echo Warning: Failed to start Ollama server. Check Ollama installation.
        echo [%TIMESTAMP%] Warning: Failed to start Ollama server. Check Ollama installation. >> "%LOG_FILE%" 2>&1
    )
)
echo Ollama server check completed.
echo [%TIMESTAMP%] Ollama server check completed. >> "%LOG_FILE%" 2>&1

:: Detect installation directory
echo Starting RadioSportChat Installation Detection...
echo Starting RadioSportChat Installation Detection... >> "%LOG_FILE%" 2>&1
set "INSTALL_DIR="
if exist "%SYSTEM_INSTALL_DIR%\RadioSportChat.exe" (
    set "INSTALL_DIR=%SYSTEM_INSTALL_DIR%"
    set "IS_SYSTEM_INSTALL=1"
) else if exist "%USER_INSTALL_DIR1%\RadioSportChat.exe" (
    set "INSTALL_DIR=%USER_INSTALL_DIR1%"
    set "IS_SYSTEM_INSTALL=0"
) else if exist "%USER_INSTALL_DIR2%\RadioSportChat.exe" (
    set "INSTALL_DIR=%USER_INSTALL_DIR2%"
    set "IS_SYSTEM_INSTALL=0"
) else if exist "%USER_INSTALL_DIR3%\RadioSportChat.exe" (
    set "INSTALL_DIR=%USER_INSTALL_DIR3%"
    set "IS_SYSTEM_INSTALL=0"
) else (
    echo Error: RadioSportChat not found in expected locations:
    echo - %SYSTEM_INSTALL_DIR%
    echo - %USER_INSTALL_DIR1%
    echo - %USER_INSTALL_DIR2%
    echo - %USER_INSTALL_DIR3%
    echo [%TIMESTAMP%] Error: RadioSportChat not found in expected locations: >> "%LOG_FILE%" 2>&1
    echo - %SYSTEM_INSTALL_DIR% >> "%LOG_FILE%" 2>&1
    echo - %USER_INSTALL_DIR1% >> "%LOG_FILE%" 2>&1
    echo - %USER_INSTALL_DIR2% >> "%LOG_FILE%" 2>&1
    echo - %USER_INSTALL_DIR3% >> "%LOG_FILE%" 2>&1
    echo Press any key to exit...
    echo [%TIMESTAMP%] Install RadioSportChat and press any key to exit... >> "%LOG_FILE%" 2>&1
    pause >nul
    exit /b 1
)
echo [%TIMESTAMP%] RadioSportChat found at !INSTALL_DIR!. >> "%LOG_FILE%" 2>&1
echo RadioSportChat installation detection completed.
echo [%TIMESTAMP%] RadioSportChat installation detection completed. >> "%LOG_FILE%" 2>&1

:: Add C:\venv\Scripts to system PATH
echo Starting System PATH Update for Virtual Environment...
echo Starting System PATH Update for Virtual Environment... >> "%LOG_FILE%" 2>&1
echo [%TIMESTAMP%] Checking if C:\venv\Scripts is in system PATH... >> "%LOG_FILE%" 2>&1
echo %PATH% | findstr /C:"C:\venv\Scripts" >nul
if !ERRORLEVEL! equ 0 (
    echo [%TIMESTAMP%] C:\venv\Scripts already in PATH. >> "%LOG_FILE%" 2>&1
) else (
    echo [%TIMESTAMP%] Adding C:\venv\Scripts to system PATH... >> "%LOG_FILE%" 2>&1
    setx PATH "%PATH%;C:\venv\Scripts" /M
    if !ERRORLEVEL! neq 0 (
        echo Warning: Failed to update system PATH with C:\venv\Scripts. Manually run: setx PATH "%%PATH%%;C:\venv\Scripts" /M
        echo [%TIMESTAMP%] Warning: Failed to update system PATH with C:\venv\Scripts. Manually run: setx PATH "%%PATH%%;C:\venv\Scripts" /M >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%TIMESTAMP%] C:\venv\Scripts added to system PATH successfully. >> "%LOG_FILE%" 2>&1
    )
)
echo System PATH update for virtual environment completed.
echo [%TIMESTAMP%] System PATH update for virtual environment completed. >> "%LOG_FILE%" 2>&1

:: Provide PATH setup instructions
echo Starting Final Instructions...
echo Starting Final Instructions... >> "%LOG_FILE%" 2>&1
echo [%TIMESTAMP%] System PATH updated for RadioSportChat and Ollama (if ollama.exe was verified). >> "%LOG_FILE%" 2>&1
echo [%TIMESTAMP%] Added to PATH if not already present: >> "%LOG_FILE%" 2>&1
echo   - C:\venv\Scripts >> "%LOG_FILE%" 2>&1
echo   - C:\Program Files\Ollama or C:\Users\%USERNAME%\AppData\Local\Programs\Ollama (if ollama.exe exists) >> "%LOG_FILE%" 2>&1
echo [%TIMESTAMP%] If RadioSportChat.exe fails, manually run: >> "%LOG_FILE%" 2>&1
echo   setx PATH "%%PATH%%;C:\venv\Scripts;C:\Program Files\Ollama;C:\Users\%USERNAME%\AppData\Local\Programs\Ollama" /M >> "%LOG_FILE%" 2>&1
echo [%TIMESTAMP%] For user PATH: >> "%LOG_FILE%" 2>&1
echo   setx PATH "%%PATH%%;C:\venv\Scripts;C:\Program Files\Ollama;C:\Users\%USERNAME%\AppData\Local\Programs\Ollama" >> "%LOG_FILE%" 2>&1
echo [%TIMESTAMP%] Optional: For external Python scripts, add to PYTHONPATH: >> "%LOG_FILE%" 2>&1
echo   setx PYTHONPATH "%%PYTHONPATH%%;C:\venv\Lib\site-packages" >> "%LOG_FILE%" 2>&1
echo [%TIMESTAMP%] Restart your command prompt or computer for PATH/PYTHONPATH changes. >> "%LOG_FILE%" 2>&1
echo Setup complete. Run RadioSportChat.exe to start the app.
echo [%TIMESTAMP%] Setup complete. Run RadioSportChat.exe to start the app. >> "%LOG_FILE%" 2>&1
echo Press any key to exit...
echo [%TIMESTAMP%] Press any key to exit... >> "%LOG_FILE%" 2>&1
pause >nul
echo Final instructions completed.
echo [%TIMESTAMP%] Final instructions completed. >> "%LOG_FILE%" 2>&1
exit /b 0

:: Function to sanitize TIME for safe logging
:SanitizeTimestamp
set "TIMESTAMP=%TIME%"
set "TIMESTAMP=%TIMESTAMP:.=_%"
set "TIMESTAMP=%TIMESTAMP::=_%"
set "TIMESTAMP=%TIMESTAMP: =_%"
exit /b 0

:: Function to verify Ollama model storage location
:VerifyOllamaModelStorage
echo Starting Ollama Model Storage Verification...
echo Starting Ollama Model Storage Verification... >> "%LOG_FILE%" 2>&1
echo [%TIMESTAMP%] Verifying model storage location... >> "%LOG_FILE%" 2>&1
set "MODEL_STORAGE_DIR="
echo [%TIMESTAMP%] Checking for existing models with ollama list... >> "%LOG_FILE%" 2>&1
ollama list > "%LOG_FILE%.models"
type "%LOG_FILE%.models" >> "%LOG_FILE%" 2>&1
set "MODELS_EXIST=0"
for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
    ollama list | findstr "%%M" >nul
    if !ERRORLEVEL! equ 0 (
        set "MODELS_EXIST=1"
        echo [%TIMESTAMP%] Model %%M found. >> "%LOG_FILE%" 2>&1
    )
)
set "DEFAULT_STORAGE=C:\Users\%USERNAME%\.ollama\models\blobs"
if exist "!DEFAULT_STORAGE!" (
    echo [%TIMESTAMP%] Default model storage found at !DEFAULT_STORAGE!. >> "%LOG_FILE%" 2>&1
    set "MODEL_STORAGE_DIR=!DEFAULT_STORAGE!"
) else (
    echo [%TIMESTAMP%] Default model storage !DEFAULT_STORAGE! does not exist. >> "%LOG_FILE%" 2>&1
    mkdir "!DEFAULT_STORAGE!" 2>nul
    if !ERRORLEVEL! equ 0 (
        echo [%TIMESTAMP%] Created default model storage at !DEFAULT_STORAGE!. >> "%LOG_FILE%" 2>&1
        set "MODEL_STORAGE_DIR=!DEFAULT_STORAGE!"
    )
)
if not defined MODEL_STORAGE_DIR (
    if defined OLLAMA_MODELS (
        echo [%TIMESTAMP%] OLLAMA_MODELS set to %OLLAMA_MODELS%. >> "%LOG_FILE%" 2>&1
        if exist "%OLLAMA_MODELS%\blobs" (
            echo [%TIMESTAMP%] Custom model storage found at %OLLAMA_MODELS%\blobs. >> "%LOG_FILE%" 2>&1
            set "MODEL_STORAGE_DIR=%OLLAMA_MODELS%\blobs"
        ) else (
            echo [%TIMESTAMP%] Custom model storage %OLLAMA_MODELS%\blobs does not exist. >> "%LOG_FILE%" 2>&1
            mkdir "%OLLAMA_MODELS%\blobs" 2>nul
            if !ERRORLEVEL! equ 0 (
                echo [%TIMESTAMP%] Created custom model storage at %OLLAMA_MODELS%\blobs. >> "%LOG_FILE%" 2>&1
                set "MODEL_STORAGE_DIR=%OLLAMA_MODELS%\blobs"
            )
        )
    )
)
if not defined MODEL_STORAGE_DIR (
    set "ALT_STORAGE=C:\ProgramData\Ollama\models\blobs"
    if exist "!ALT_STORAGE!" (
        echo [%TIMESTAMP%] Alternative model storage found at !ALT_STORAGE!. >> "%LOG_FILE%" 2>&1
        set "MODEL_STORAGE_DIR=!ALT_STORAGE!"
    ) else (
        set "ALT_STORAGE=C:\Users\%USERNAME%\AppData\Local\Ollama\models\blobs"
        if exist "!ALT_STORAGE!" (
            echo [%TIMESTAMP%] Alternative model storage found at !ALT_STORAGE!. >> "%LOG_FILE%" 2>&1
            set "MODEL_STORAGE_DIR=!ALT_STORAGE!"
        )
    )
)
if not defined MODEL_STORAGE_DIR (
    echo [%TIMESTAMP%] No valid model storage found. Using default !DEFAULT_STORAGE!. >> "%LOG_FILE%" 2>&1
    mkdir "!DEFAULT_STORAGE!" 2>nul
    if !ERRORLEVEL! equ 0 (
        set "MODEL_STORAGE_DIR=!DEFAULT_STORAGE!"
    ) else (
        echo Error: Cannot create default model storage at !DEFAULT_STORAGE!. Check permissions and press any key to exit...
        echo [%TIMESTAMP%] Error: Cannot create default model storage at !DEFAULT_STORAGE!. Check permissions and press any key to exit... >> "%LOG_FILE%" 2>&1
        pause >nul
        exit /b 1
    )
)
echo [%TIMESTAMP%] Using model storage location: !MODEL_STORAGE_DIR! >> "%LOG_FILE%" 2>&1
echo Ollama model storage verification completed. >> "%LOG_FILE%" 2>&1
exit /b 0

:: Function to handle Ollama model checks and pulls
:HandleOllamaModels
echo Starting Ollama Models Handling...
echo Starting Ollama Models Handling... >> "%LOG_FILE%" 2>&1
echo [%TIMESTAMP%] Listing current Ollama models... >> "%LOG_FILE%" 2>&1
ollama list > "%LOG_FILE%.models"
type "%LOG_FILE%.models" >> "%LOG_FILE%" 2>&1
call :VerifyOllamaModelStorage
if !ERRORLEVEL! neq 0 (
    echo Error: Failed to verify model storage location. Press any key to exit...
    echo [%TIMESTAMP%] Error: Failed to verify model storage location. Press any key to exit... >> "%LOG_FILE%" 2>&1
    pause >nul
    exit /b 1
)
echo [%TIMESTAMP%] Checking for local Ollama model files... >> "%LOG_FILE%" 2>&1
set "MODEL_FILES_FOUND=0"
for %%f in (%SCRIPT_DIR%sha256-*) do (
    set "MODEL_FILES_FOUND=1"
)
if !MODEL_FILES_FOUND! equ 1 (
    echo [%TIMESTAMP%] Copying local model files to !MODEL_STORAGE_DIR!... >> "%LOG_FILE%" 2>&1
    set "START_TIME=%TIME%"
    copy %SCRIPT_DIR%sha256-* "!MODEL_STORAGE_DIR!\" >nul
    call :SanitizeTimestamp
    set "END_TIME=%TIMESTAMP%"
    if !ERRORLEVEL! neq 0 (
        echo Warning: Failed to copy model files to !MODEL_STORAGE_DIR!. Check disk space or permissions.
        echo [%TIMESTAMP%] Warning: Failed to copy model files to !MODEL_STORAGE_DIR!. Check disk space or permissions. >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%TIMESTAMP%] Model files copied successfully to !MODEL_STORAGE_DIR!. >> "%LOG_FILE%" 2>&1
    )
    echo [%TIMESTAMP%] Verifying models after copying files... >> "%LOG_FILE%" 2>&1
    ollama list > "%LOG_FILE%.models"
    type "%LOG_FILE%.models" >> "%LOG_FILE%" 2>&1
    set "MODELS_MISSING=0"
    for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
        ollama list | findstr "%%M" >nul
        if !ERRORLEVEL! neq 0 (
            echo Warning: Model %%M not found after copying files.
            echo [%TIMESTAMP%] Warning: Model %%M not found after copying files. >> "%LOG_FILE%" 2>&1
            set "MODELS_MISSING=1"
        )
    )
    if !MODELS_MISSING! equ 0 (
        echo [%TIMESTAMP%] All required models verified successfully after copying. >> "%LOG_FILE%" 2>&1
    ) else (
        echo [%TIMESTAMP%] Warning: Some models not recognized after copying. Proceeding to pull missing models... >> "%LOG_FILE%" 2>&1
    )
) else (
    echo [%TIMESTAMP%] No local model files found in current directory. >> "%LOG_FILE%" 2>&1
)
echo [%TIMESTAMP%] Checking and pulling required Ollama models... >> "%LOG_FILE%" 2>&1
for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
    ollama list | findstr "%%M" >nul
    if %ERRORLEVEL% neq 0 (
        echo [%TIMESTAMP%] Pulling %%M model... >> "%LOG_FILE%" 2>&1
        set "START_TIME=%TIME%"
        echo [%TIMESTAMP%] Processing %%M download... >> "%LOG_FILE%" 2>&1
        ollama pull %%M >nul
        call :SanitizeTimestamp
        set "END_TIME=%TIMESTAMP%"
        if !ERRORLEVEL! neq 0 (
            echo Warning: Failed to pull %%M model. Check network or Ollama repository.
            echo [%TIMESTAMP%] Warning: Failed to pull %%M model. Check network or Ollama repository. >> "%LOG_FILE%" 2>&1
        ) else (
            echo [%TIMESTAMP%] Model %%M pulled successfully. >> "%LOG_FILE%" 2>&1
        )
    ) else (
        echo [%TIMESTAMP%] Model %%M already installed. >> "%LOG_FILE%" 2>&1
    )
)
echo [%TIMESTAMP%] Verifying all required models after pulling... >> "%LOG_FILE%" 2>&1
set "MODELS_MISSING=0"
for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
    ollama list | findstr "%%M" >nul
    if !ERRORLEVEL! neq 0 (
        echo Warning: Model %%M not found after pulling.
        echo [%TIMESTAMP%] Warning: Model %%M not found after pulling. >> "%LOG_FILE%" 2>&1
        set "MODELS_MISSING=1"
    )
)
if !MODELS_MISSING! equ 0 (
    echo [%TIMESTAMP%] All required models verified successfully after pulling. >> "%LOG_FILE%" 2>&1
    ollama list > "%LOG_FILE%.models"
    type "%LOG_FILE%.models" >> "%LOG_FILE%" 2>&1
) else (
    echo Warning: Some models are missing after pulling. Check %LOG_FILE%.models for details.
    echo [%TIMESTAMP%] Warning: Some models are missing after pulling. Check %LOG_FILE%.models for details. >> "%LOG_FILE%" 2>&1
)
echo Ollama models handling completed. >> "%LOG_FILE%" 2>&1
exit /b