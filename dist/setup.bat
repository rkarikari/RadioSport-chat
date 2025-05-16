@echo off
setlocal EnableDelayedExpansion

:: Initialize logging
set "LOG_FILE=%~dp0setup.log"
echo === Setup Started at %DATE% %TIME% === > "%LOG_FILE%"
echo === Setup Started at %DATE% %TIME% ===

:: Check for administrative privileges
echo === Checking Administrative Privileges === | tee -a "%LOG_FILE%"
net session >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: This script requires administrative privileges. Run as Administrator and press any key to exit... | tee -a "%LOG_FILE%"
    pause >nul
    exit /b 1
)
echo [%TIME%] Administrative privileges confirmed. | tee -a "%LOG_FILE%"

:: Check for RadioSportChat installation
echo === Checking RadioSportChat Installation === | tee -a "%LOG_FILE%"
set "SCRIPT_DIR=%~dp0"
set "INSTALLER_FOUND=0"
set "APP_ALREADY_INSTALLED=0"
set "SYSTEM_INSTALL_DIR=C:\Program Files\RadioSportChat"
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
        echo [%TIME%] RadioSportChat already installed at %%D. Skipping MSI installation... | tee -a "%LOG_FILE%"
        set "APP_ALREADY_INSTALLED=1"
    )
)
if !APP_ALREADY_INSTALLED! equ 0 (
    echo [%TIME%] Checking for RadioSportChat installer in %SCRIPT_DIR%... | tee -a "%LOG_FILE%"
    for %%f in ("%SCRIPT_DIR%RadioSportChat-*.msi") do (
        echo [%TIME%] Found installer: %%f | tee -a "%LOG_FILE%"
        echo [%TIME%] Starting RadioSportChat installation. Please complete the installation wizard... | tee -a "%LOG_FILE%"
        set "START_TIME=%TIME%"
        start /wait msiexec /i "%%f" /log "%SCRIPT_DIR%radiosportchat_install.log"
        set "END_TIME=%TIME%"
        if !ERRORLEVEL! equ 0 (
            echo [%TIME%] RadioSportChat installation completed successfully. | tee -a "%LOG_FILE%"
        ) else (
            echo [%TIME%] Warning: RadioSportChat installation may have failed. Check %SCRIPT_DIR%radiosportchat_install.log for details. | tee -a "%LOG_FILE%"
        )
        echo [%TIME%] Press any key to continue with setup... | tee -a "%LOG_FILE%"
        pause >nul
        set "INSTALLER_FOUND=1"
    )
    if !INSTALLER_FOUND! equ 0 (
        echo [%TIME%] Error: No RadioSportChat installer found in %SCRIPT_DIR%. Press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
)

:: Set up virtual environment directory
echo === Setting Up Virtual Environment Directory === | tee -a "%LOG_FILE%"
set "VENV_DIR=C:\venv"
if not exist "!VENV_DIR!" (
    echo [%TIME%] Creating virtual environment directory at !VENV_DIR!... | tee -a "%LOG_FILE%"
    mkdir "!VENV_DIR!"
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Error: Failed to create !VENV_DIR!. Check permissions and press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
)
echo [%TIME%] Setting permissions for !VENV_DIR!... | tee -a "%LOG_FILE%"
icacls "!VENV_DIR!" /grant "Authenticated Users:(OI)(CI)M" /T
if !ERRORLEVEL! neq 0 (
    echo [%TIME%] Warning: Failed to set permissions for !VENV_DIR!. Manually grant 'Authenticated Users' modify permissions. | tee -a "%LOG_FILE%"
)
echo [%TIME%] Virtual environment directory setup completed. | tee -a "%LOG_FILE%"

:: Check for Python 3.11
echo === Checking Python 3.11 Installation === | tee -a "%LOG_FILE%"
set "PYTHON_INSTALLED=0"
python --version 2>nul | findstr "3.11" >nul
if !ERRORLEVEL! equ 0 (
    echo [%TIME%] Python 3.11 found in system PATH. Verifying location... | tee -a "%LOG_FILE%"
    for /f "tokens=2 delims==" %%a in ('where python ^| findstr /C:"python.exe"') do (
        if /i "%%a"=="C:\Python\python.exe" (
            set "PYTHON_INSTALLED=1"
            echo [%TIME%] Python 3.11 already installed at C:\Python. | tee -a "%LOG_FILE%"
        ) else (
            echo [%TIME%] Warning: Python 3.11 found at %%a, but expected at C:\Python. Proceeding with installation... | tee -a "%LOG_FILE%"
        )
    )
)
if !PYTHON_INSTALLED! equ 0 (
    if exist "C:\Python\python.exe" (
        C:\Python\python.exe --version 2>nul | findstr "3.11" >nul
        if !ERRORLEVEL! equ 0 (
            set "PYTHON_INSTALLED=1"
            echo [%TIME%] Python 3.11 already installed at C:\Python. | tee -a "%LOG_FILE%"
        ) else (
            echo [%TIME%] Error: Python found at C:\Python but version is not 3.11. Uninstall and press any key to exit... | tee -a "%LOG_FILE%"
            pause >nul
            exit /b 1
        )
    )
)
if !PYTHON_INSTALLED! equ 0 (
    echo [%TIME%] Checking for Python installer... | tee -a "%LOG_FILE%"
    if not exist "%SCRIPT_DIR%python-3.11.9-amd64.exe" (
        echo [%TIME%] Error: Python installer not found at %SCRIPT_DIR%python-3.11.9-amd64.exe. Press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
    echo [%TIME%] Installing Python 3.11 to C:\Python... | tee -a "%LOG_FILE%"
    mkdir "C:\Python" 2>nul
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Error: Cannot create C:\Python. Check permissions and press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
    set "START_TIME=%TIME%"
    start /wait %SCRIPT_DIR%python-3.11.9-amd64.exe /quiet InstallAllUsers=1 TargetDir=C:\Python PrependPath=1
    set "END_TIME=%TIME%"
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Error: Failed to install Python 3.11. Check installer logs and press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
    if not exist "C:\Python\python.exe" (
        echo [%TIME%] Error: Python 3.11 installation verification failed. Press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
    C:\Python\python.exe --version 2>nul | findstr "3.11" >nul
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Error: Python 3.11 version verification failed. Press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
    echo [%TIME%] Python 3.11 installed successfully to C:\Python. | tee -a "%LOG_FILE%"
)

:: Create and set up virtual environment
echo === Setting Up Virtual Environment === | tee -a "%LOG_FILE%"
if exist "%VENV_DIR%\Scripts\python.exe" (
    echo [%TIME%] Virtual environment exists at %VENV_DIR%. Verifying functionality... | tee -a "%LOG_FILE%"
    call "%VENV_DIR%\Scripts\activate.bat"
    pip --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        echo [%TIME%] Virtual environment is functional. | tee -a "%LOG_FILE%"
    ) else (
        echo [%TIME%] Warning: Virtual environment at %VENV_DIR% is not functional. Recreating... | tee -a "%LOG_FILE%"
        rd /S /Q "%VENV_DIR%" 2>nul
        C:\Python\python.exe -m venv "%VENV_DIR%"
        if !ERRORLEVEL! neq 0 (
            echo [%TIME%] Error: Failed to recreate virtual environment. Check permissions and press any key to exit... | tee -a "%LOG_FILE%"
            call "%VENV_DIR%\Scripts\deactivate.bat" 2>nul
            pause >nul
            exit /b 1
        )
    )
    call "%VENV_DIR%\Scripts\deactivate.bat"
) else (
    echo [%TIME%] Creating virtual environment at %VENV_DIR%... | tee -a "%LOG_FILE%"
    set "START_TIME=%TIME%"
    C:\Python\python.exe -m venv "%VENV_DIR%"
    set "END_TIME=%TIME%"
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Error: Failed to create virtual environment. Check Python installation and press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
)
echo [%TIME%] Virtual environment setup completed. | tee -a "%LOG_FILE%"

:: Install pip packages
echo === Installing Pip Packages === | tee -a "%LOG_FILE%"
if not exist "%SCRIPT_DIR%venv" (
    echo [%TIME%] Error: Package directory %SCRIPT_DIR%venv not found. Press any key to exit... | tee -a "%LOG_FILE%"
    pause >nul
    exit /b 1
)
if not exist "%SCRIPT_DIR%venv\requirements.txt" (
    echo [%TIME%] Error: requirements.txt not found at %SCRIPT_DIR%venv\requirements.txt. Press any key to exit... | tee -a "%LOG_FILE%"
    pause >nul
    exit /b 1
)
set "PACKAGE_FOUND=0"
for %%f in ("%SCRIPT_DIR%venv\streamlit*.whl" "%SCRIPT_DIR%venv\streamlit*.tar.gz") do (
    set "PACKAGE_FOUND=1"
)
if !PACKAGE_FOUND! equ 0 (
    echo [%TIME%] Warning: No streamlit package files found in %SCRIPT_DIR%venv. Installation may fail. | tee -a "%LOG_FILE%"
)
echo [%TIME%] Installing pip packages from %SCRIPT_DIR%venv... | tee -a "%LOG_FILE%"
call "%VENV_DIR%\Scripts\activate.bat"
set "START_TIME=%TIME%"
echo [%TIME%] Processing package installation... | tee -a "%LOG_FILE%"
pip install --no-index --find-links=%SCRIPT_DIR%venv -r %SCRIPT_DIR%venv\requirements.txt >nul
if !ERRORLEVEL! neq 0 (
    echo [%TIME%] Error: Failed to install pip packages. Check %SCRIPT_DIR%venv and press any key to exit... | tee -a "%LOG_FILE%"
    call "%VENV_DIR%\Scripts\deactivate.bat"
    pause >nul
    exit /b 1
)
set "END_TIME=%TIME%"
set "PACKAGES_OK=1"
for %%P in (streamlit langchain-ollama) do (
    pip show %%P >nul 2>&1
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Warning: Package %%P not found after installation. | tee -a "%LOG_FILE%"
        set "PACKAGES_OK=0"
    )
)
call "%VENV_DIR%\Scripts\deactivate.bat"
if !PACKAGES_OK! equ 1 (
    echo [%TIME%] Pip packages installed successfully. | tee -a "%LOG_FILE%"
) else (
    echo [%TIME%] Warning: Some packages may not have installed correctly. Check requirements.txt. | tee -a "%LOG_FILE%"
)

:: Check and install Ollama
echo === Installing Ollama === | tee -a "%LOG_FILE%"
set "OLLAMA_INSTALLED=0"
if exist "C:\Program Files\Ollama\ollama.exe" (
    ollama --version >nul 2>&1
    if !ERRORLEVEL! equ 0 (
        set "OLLAMA_INSTALLED=1"
        echo [%TIME%] Ollama is already installed. Checking models... | tee -a "%LOG_FILE%"
        set "MODELS_MISSING=0"
        for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
            ollama list | findstr "%%M" >nul || set "MODELS_MISSING=1"
        )
        if !MODELS_MISSING! equ 0 (
            echo [%TIME%] Required models granite3.3:2b, nomic-embed-text:latest, qwen3:4b, and qwen3:1.7b are already installed. | tee -a "%LOG_FILE%"
        ) else (
            call :HandleOllamaModels
        )
    )
)
if !OLLAMA_INSTALLED! equ 0 (
    echo [%TIME%] Checking for local Ollama installer... | tee -a "%LOG_FILE%"
    if exist "%SCRIPT_DIR%OllamaSetup.exe" (
        echo [%TIME%] Using local copy of OllamaSetup.exe... | tee -a "%LOG_FILE%"
    ) else (
        echo [%TIME%] Downloading OllamaSetup.exe... | tee -a "%LOG_FILE%"
        set "START_TIME=%TIME%"
        curl -L https://ollama.com/download/OllamaSetup.exe -o %SCRIPT_DIR%OllamaSetup.exe
        set "END_TIME=%TIME%"
        if !ERRORLEVEL! neq 0 (
            echo [%TIME%] Error: Failed to download OllamaSetup.exe. Check network and press any key to exit... | tee -a "%LOG_FILE%"
            pause >nul
            exit /b 1
        )
    )
    echo [%TIME%] Installing Ollama... | tee -a "%LOG_FILE%"
    set "START_TIME=%TIME%"
    start /wait %SCRIPT_DIR%OllamaSetup.exe
    set "END_TIME=%TIME%"
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Warning: Ollama installation may have failed. Check installer logs. | tee -a "%LOG_FILE%"
    )
    if not exist "C:\Program Files\Ollama\ollama.exe" (
        echo [%TIME%] Error: Ollama installation verification failed. Press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
    echo [%TIME%] Ollama installed successfully. | tee -a "%LOG_FILE%"
    call :HandleOllamaModels
)

:: Start Ollama server
echo === Starting Ollama Server === | tee -a "%LOG_FILE%"
echo [%TIME%] Starting Ollama server... | tee -a "%LOG_FILE%"
ollama serve >nul 2>&1
if %ERRORLEVEL% equ 0 (
    echo [%TIME%] Ollama server started successfully. | tee -a "%LOG_FILE%"
) else (
    echo [%TIME%] Warning: Failed to start Ollama server. It may already be running. | tee -a "%LOG_FILE%"
)

:: Detect installation directory
echo === Detecting RadioSportChat Installation === | tee -a "%LOG_FILE%"
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
    echo [%TIME%] Error: RadioSportChat not found in expected locations: | tee -a "%LOG_FILE%"
    echo - %SYSTEM_INSTALL_DIR% | tee -a "%LOG_FILE%"
    echo - %USER_INSTALL_DIR1% | tee -a "%LOG_FILE%"
    echo - %USER_INSTALL_DIR2% | tee -a "%LOG_FILE%"
    echo - %USER_INSTALL_DIR3% | tee -a "%LOG_FILE%"
    echo [%TIME%] Install RadioSportChat and press any key to exit... | tee -a "%LOG_FILE%"
    pause >nul
    exit /b 1
)
echo [%TIME%] RadioSportChat found at !INSTALL_DIR!. | tee -a "%LOG_FILE%"

:: Add C:\venv\Scripts to system PATH
echo === Updating System PATH === | tee -a "%LOG_FILE%"
echo [%TIME%] Checking if C:\venv\Scripts is in system PATH... | tee -a "%LOG_FILE%"
echo %PATH% | findstr /C:"C:\venv\Scripts" >nul
if !ERRORLEVEL! equ 0 (
    echo [%TIME%] C:\venv\Scripts already in PATH. | tee -a "%LOG_FILE%"
) else (
    echo [%TIME%] Adding C:\venv\Scripts to system PATH... | tee -a "%LOG_FILE%"
    setx PATH "%PATH%;C:\venv\Scripts" /M
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Warning: Failed to update system PATH. Add C:\venv\Scripts manually. | tee -a "%LOG_FILE%"
    ) else (
        echo [%TIME%] C:\venv\Scripts added to system PATH successfully. | tee -a "%LOG_FILE%"
    )
)

:: Provide PATH setup instructions
echo === Final Instructions === | tee -a "%LOG_FILE%"
echo [%TIME%] System PATH updated for RadioSportChat. | tee -a "%LOG_FILE%"
echo [%TIME%] C:\venv\Scripts added to PATH. If RadioSportChat.exe fails, manually run: | tee -a "%LOG_FILE%"
echo   setx PATH "%PATH%;C:\venv\Scripts" /M | tee -a "%LOG_FILE%"
echo [%TIME%] For user PATH: | tee -a "%LOG_FILE%"
echo   setx PATH "%PATH%;C:\venv\Scripts" | tee -a "%LOG_FILE%"
echo [%TIME%] Optional: For external Python scripts, add to PYTHONPATH: | tee -a "%LOG_FILE%"
echo   setx PYTHONPATH "%PYTHONPATH%;C:\venv\Lib\site-packages" | tee -a "%LOG_FILE%"
echo [%TIME%] Restart your command prompt or computer for PATH/PYTHONPATH changes. | tee -a "%LOG_FILE%"
echo [%TIME%] Setup complete. Run RadioSportChat.exe to start the app. | tee -a "%LOG_FILE%"
echo [%TIME%] Press any key to exit... | tee -a "%LOG_FILE%"
pause >nul
exit /b 0

:: Function to verify Ollama model storage location
:VerifyOllamaModelStorage
echo === Verifying Ollama Model Storage === | tee -a "%LOG_FILE%"
echo [%TIME%] Verifying model storage location... | tee -a "%LOG_FILE%"
set "MODEL_STORAGE_DIR="
echo [%TIME%] Checking for existing models with ollama list... | tee -a "%LOG_FILE%"
ollama list > "%LOG_FILE%.models"
type "%LOG_FILE%.models" >> "%LOG_FILE%"
set "MODELS_EXIST=0"
for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
    ollama list | findstr "%%M" >nul
    if !ERRORLEVEL! equ 0 (
        set "MODELS_EXIST=1"
        echo [%TIME%] Model %%M found. | tee -a "%LOG_FILE%"
    )
)
set "DEFAULT_STORAGE=C:\Users\%USERNAME%\.ollama\models\blobs"
if exist "!DEFAULT_STORAGE!" (
    echo [%TIME%] Default model storage found at !DEFAULT_STORAGE!. | tee -a "%LOG_FILE%"
    set "MODEL_STORAGE_DIR=!DEFAULT_STORAGE!"
) else (
    echo [%TIME%] Default model storage !DEFAULT_STORAGE! does not exist. | tee -a "%LOG_FILE%"
    mkdir "!DEFAULT_STORAGE!" 2>nul
    if !ERRORLEVEL! equ 0 (
        echo [%TIME%] Created default model storage at !DEFAULT_STORAGE!. | tee -a "%LOG_FILE%"
        set "MODEL_STORAGE_DIR=!DEFAULT_STORAGE!"
    )
)
if not defined MODEL_STORAGE_DIR (
    if defined OLLAMA_MODELS (
        echo [%TIME%] OLLAMA_MODELS set to %OLLAMA_MODELS%. | tee -a "%LOG_FILE%"
        if exist "%OLLAMA_MODELS%\blobs" (
            echo [%TIME%] Custom model storage found at %OLLAMA_MODELS%\blobs. | tee -a "%LOG_FILE%"
            set "MODEL_STORAGE_DIR=%OLLAMA_MODELS%\blobs"
        ) else (
            echo [%TIME%] Custom model storage %OLLAMA_MODELS%\blobs does not exist. | tee -a "%LOG_FILE%"
            mkdir "%OLLAMA_MODELS%\blobs" 2>nul
            if !ERRORLEVEL! equ 0 (
                echo [%TIME%] Created custom model storage at %OLLAMA_MODELS%\blobs. | tee -a "%LOG_FILE%"
                set "MODEL_STORAGE_DIR=%OLLAMA_MODELS%\blobs"
            )
        )
    )
)
if not defined MODEL_STORAGE_DIR (
    set "ALT_STORAGE=C:\ProgramData\Ollama\models\blobs"
    if exist "!ALT_STORAGE!" (
        echo [%TIME%] Alternative model storage found at !ALT_STORAGE!. | tee -a "%LOG_FILE%"
        set "MODEL_STORAGE_DIR=!ALT_STORAGE!"
    ) else (
        set "ALT_STORAGE=C:\Users\%USERNAME%\AppData\Local\Ollama\models\blobs"
        if exist "!ALT_STORAGE!" (
            echo [%TIME%] Alternative model storage found at !ALT_STORAGE!. | tee -a "%LOG_FILE%"
            set "MODEL_STORAGE_DIR=!ALT_STORAGE!"
        )
    )
)
if not defined MODEL_STORAGE_DIR (
    echo [%TIME%] No valid model storage found. Using default !DEFAULT_STORAGE!. | tee -a "%LOG_FILE%"
    mkdir "!DEFAULT_STORAGE!" 2>nul
    if !ERRORLEVEL! equ 0 (
        set "MODEL_STORAGE_DIR=!DEFAULT_STORAGE!"
    ) else (
        echo [%TIME%] Error: Cannot create default model storage at !DEFAULT_STORAGE!. Check permissions and press any key to exit... | tee -a "%LOG_FILE%"
        pause >nul
        exit /b 1
    )
)
echo [%TIME%] Using model storage location: !MODEL_STORAGE_DIR! | tee -a "%LOG_FILE%"
exit /b 0

:: Function to handle Ollama model checks and pulls
:HandleOllamaModels
echo === Handling Ollama Models === | tee -a "%LOG_FILE%"
echo [%TIME%] Listing current Ollama models... | tee -a "%LOG_FILE%"
ollama list > "%LOG_FILE%.models"
type "%LOG_FILE%.models" >> "%LOG_FILE%"
call :VerifyOllamaModelStorage
if !ERRORLEVEL! neq 0 (
    echo [%TIME%] Error: Failed to verify model storage location. Press any key to exit... | tee -a "%LOG_FILE%"
    pause >nul
    exit /b 1
)
echo [%TIME%] Checking for local Ollama model files... | tee -a "%LOG_FILE%"
set "MODEL_FILES_FOUND=0"
for %%f in (%SCRIPT_DIR%sha256-*) do (
    set "MODEL_FILES_FOUND=1"
)
if !MODEL_FILES_FOUND! equ 1 (
    echo [%TIME%] Copying local model files to !MODEL_STORAGE_DIR!... | tee -a "%LOG_FILE%"
    set "START_TIME=%TIME%"
    copy %SCRIPT_DIR%sha256-* "!MODEL_STORAGE_DIR!\" >nul
    set "END_TIME=%TIME%"
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Warning: Failed to copy model files to !MODEL_STORAGE_DIR!. Check disk space or permissions. | tee -a "%LOG_FILE%"
    ) else (
        echo [%TIME%] Model files copied successfully to !MODEL_STORAGE_DIR!. | tee -a "%LOG_FILE%"
    )
    echo [%TIME%] Verifying models after copying files... | tee -a "%LOG_FILE%"
    ollama list > "%LOG_FILE%.models"
    type "%LOG_FILE%.models" >> "%LOG_FILE%"
    set "MODELS_MISSING=0"
    for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
        ollama list | findstr "%%M" >nul
        if !ERRORLEVEL! neq 0 (
            echo [%TIME%] Warning: Model %%M not found after copying files. | tee -a "%LOG_FILE%"
            set "MODELS_MISSING=1"
        )
    )
    if !MODELS_MISSING! equ 0 (
        echo [%TIME%] All required models verified successfully after copying. | tee -a "%LOG_FILE%"
    ) else (
        echo [%TIME%] Warning: Some models not recognized after copying. Proceeding to pull missing models... | tee -a "%LOG_FILE%"
    )
) else (
    echo [%TIME%] No local model files found in current directory. | tee -a "%LOG_FILE%"
)
echo [%TIME%] Checking and pulling required Ollama models... | tee -a "%LOG_FILE%"
for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
    ollama list | findstr "%%M" >nul
    if %ERRORLEVEL% neq 0 (
        echo [%TIME%] Pulling %%M model... | tee -a "%LOG_FILE%"
        set "START_TIME=%TIME%"
        echo [%TIME%] Processing %%M download... | tee -a "%LOG_FILE%"
        ollama pull %%M >nul
        set "END_TIME=%TIME%"
        if !ERRORLEVEL! neq 0 (
            echo [%TIME%] Warning: Failed to pull %%M model. Check network or Ollama repository. | tee -a "%LOG_FILE%"
        ) else (
            echo [%TIME%] Model %%M pulled successfully. | tee -a "%LOG_FILE%"
        )
    ) else (
        echo [%TIME%] Model %%M already installed. | tee -a "%LOG_FILE%"
    )
)
echo [%TIME%] Verifying all required models after pulling... | tee -a "%LOG_FILE%"
set "MODELS_MISSING=0"
for %%M in ("granite3.3:2b" "nomic-embed-text:latest" "qwen3:4b" "qwen3:1.7b") do (
    ollama list | findstr "%%M" >nul
    if !ERRORLEVEL! neq 0 (
        echo [%TIME%] Warning: Model %%M not found after pulling. | tee -a "%LOG_FILE%"
        set "MODELS_MISSING=1"
    )
)
if !MODELS_MISSING! equ 0 (
    echo [%TIME%] All required models verified successfully after pulling. | tee -a "%LOG_FILE%"
    ollama list > "%LOG_FILE%.models"
    type "%LOG_FILE%.models" >> "%LOG_FILE%"
) else (
    echo [%TIME%] Warning: Some models are missing after pulling. Check %LOG_FILE%.models for details. | tee -a "%LOG_FILE%"
)
exit /b