# RadioSportChat Setup Script

## Overview

The `setup.bat` script automates the installation and configuration of **RadioSportChat**, a chat application powered by Ollama models. It installs dependencies, sets up a virtual environment, verifies model storage, and ensures all required components are properly configured on a Windows system. The script provides minimal console output (section headers, errors, prompts, and success messages) and logs all actions to `setup.log` for troubleshooting. All required files are located in the script's directory or its `venv` subfolder.

### Key Features
- Installs RadioSportChat, Python 3.11, and Ollama (if not already installed).
- Creates and configures a virtual environment at `C:\venv`.
- Installs pip packages from `venv\requirements.txt` in the script's directory, logging errors to `setup.log` and showing only the success message in the console. Note: Some pip messages (e.g., `processed file`) may appear in the console due to stderr output.
- Verifies and configures Ollama model storage (default: `C:\Users\%USERNAME%\.ollama\models\blobs`).
- Copies `sha256-*` model files and pulls required models (`granite3.3:2b`, `nomic-embed-text:latest`, `qwen3:4b`, `qwen3:1.7b`).
- Performs `ollama list` checks to verify model recognition before and after copying/pulling.
- Checks for existing Ollama installation using `ollama list`. If present:
  - Skips installation to prevent overwriting.
  - Checks for `ollama.exe` in:
    - `C:\Program Files\Ollama` (default).
    - `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama` (alternate).
  - Adds the appropriate path to system PATH if missing and `ollama.exe` exists.
- Checks if the Ollama server is running using `tasklist | findstr ollama`:
  - If the process is found, skips starting the server.
  - If not found, runs `ollama serve`.
- Updates system PATH with `C:\venv\Scripts` and either `C:\Program Files\Ollama` or `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama` (if `ollama.exe` exists, avoiding invalid paths).
- Sanitizes `%TIME%` into `%TIMESTAMP%` (e.g., `17_57_23_45`) to handle locale-specific formats and avoid parsing errors.
- Validates `%LOG_FILE%` (default: `setup.log` in script directory) and falls back to `%TEMP%\setup.log` if the directory is unwritable.
- Provides console output for:
  - Section headers (e.g., `Starting Ollama Installation...`).
  - Errors (e.g., `Error: Failed to install Python...`).
  - Prompts (e.g., `Press any key to continue with setup...`).
  - Success messages (e.g., `Ollama installation completed.`).
  - All other output is logged to `setup.log`.

## Prerequisites

Before running `setup.bat`, ensure the following:

- **Operating System**: Windows 10 or later.
- **Administrative Privileges**: Run the script as Administrator.
- **Installer Files** (all in the same directory as `setup.bat`):
  - RadioSportChat MSI installer (`RadioSportChat-*.msi`).
  - Python 3.11 installer (`python-3.11.9-amd64.exe`).
  - `venv` subfolder containing:
    - `requirements.txt`
    - Pip package files (e.g., `streamlit*.whl`, `streamlit*.tar.gz`).
- **Optional**: `sha256-*` model files in the script's directory for offline model installation.
- **Optional**: `OllamaSetup.exe` in the script's directory (downloaded if absent and Ollama is not installed).
- **Internet Connection**: Required for downloading Ollama (if not present) and pulling models.
- **Disk Space**: At least 10 GB free for models and dependencies.
- **Ollama Models**: If using custom model storage, set the `OLLAMA_MODELS` environment variable.

## Usage

1. **Prepare the Environment**:
   - Place `setup.bat`, `RadioSportChat-*.msi`, and `python-3.11.9-amd64.exe` in the same directory.
   - Create a `venv` subfolder containing `requirements.txt` and pip package files (e.g., `streamlit*.whl`).
   - Optionally, include `sha256-*` model files and `OllamaSetup.exe` in the script's directory.

2. **Run the Script**:
   - Open a Command Prompt as Administrator (`Run as Administrator`).
   - Navigate to the script directory: `cd path\to\script\directory`.
   - Execute the script: `setup.bat`.
   - Follow any on-screen prompts (e.g., complete the RadioSportChat installer wizard).

3. **Monitor Progress**:
   - Console output includes:
     - Section headers (e.g., `Starting Python 3.11 Installation Check...`).
     - Errors (e.g., `Error: Python installer not found...`).
     - Prompts (e.g., `Press any key to continue with setup...`).
     - Success messages (e.g., `Python 3.11 installation completed.`).
   - All actions are logged to `setup.log` and `setup.log.models` (for `ollama list` output) in the script's directory or `%TEMP%` if unwritable.
   - Check `setup.log` for detailed progress (e.g., `[17_57_23_45] Python 3.11 installed successfully`).

4. **Post-Setup**:
   - After completion, run `RadioSportChat.exe` from its installation directory (e.g., `C:\Program Files\Radiosport\RadioSportChat`).
   - If the app fails to start, check `setup.log` and follow the troubleshooting steps below.
   - Restart your command prompt or computer to apply PATH changes.

## Script Functionality

### Installation Steps
- **RadioSportChat**: Installs from `RadioSportChat-*.msi` or skips if already installed at `C:\Program Files\Radiosport\RadioSportChat` or user-specific paths.
- **Python 3.11**: Checks if `C:\Python` exists. If it does, verifies `python.exe` is version 3.11. If valid, skips installation. If not, installs to `C:\Python` from `python-3.11.9-amd64.exe` (creating `C:\Python` if needed).
- **Virtual Environment**: Creates at `C:\venv`, ensures functionality.
- **Pip Packages**: Installs from `venv\requirements.txt`, logs errors to `setup.log`, verifies `streamlit` and `langchain-ollama`. The success message (`Pip packages installation completed.`) appears in the console, but some stderr messages (e.g., `processed file`) may also appear.
- **Ollama**: Checks for existing installation with `ollama list`. If present:
  - Skips installation.
  - Verifies `ollama.exe` in `C:\Program Files\Ollama` or `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama`.
  - Adds the appropriate path to system PATH if missing.
  - If not present, installs to `C:\Program Files\Ollama` or `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama` from `OllamaSetup.exe` (local or downloaded).
- **Ollama PATH Update**: Adds `C:\Program Files\Ollama` or `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama` to system PATH if `ollama.exe` exists and the path is not already present. Skips duplicates.
- **Ollama Server**: Checks if the server is running using `tasklist | findstr ollama`:
  - If the process is detected, logs that the server is running.
  - If not detected, runs `ollama serve` and logs the result.

### Model Handling
- **Storage Verification**: Checks default (`C:\Users\%USERNAME%\.ollama\models\blobs`), custom (`%OLLAMA_MODELS%`), and alternative locations.
- **Model Files**: Copies `sha256-*` files from the script's directory to the verified storage location.
- **Model Pulling**: Pulls required models if not present.
- **Verification**: Runs `ollama list` before and after copying/pulling to confirm model recognition.

### Feedback
- **Console Output**: Includes section headers, errors, prompts, and success messages for each section.
- **Logging**: Appends all output to `setup.log` and model lists to `setup.log.models` using standard Windows redirection (`>>`).
- **Warnings/Errors**: Logs warnings (e.g., "Warning: Model qwen3:4b not found") and errors (e.g., "Error: Failed to download OllamaSetup.exe") with actionable advice.

## Troubleshooting

- **Script Fails with Permission Error**:
  - Ensure you run `setup.bat` as Administrator.
  - Check write permissions for `C:\Python`, `C:\venv`, and the Ollama model storage directory.

- **Cannot Create C:\Python**:
  - If you see `Error: Cannot create C:\Python. Check permissions...`:
    - Verify if `C:\Python` exists:
      ```cmd
      dir C:\Python
      ```
      - If it exists, ensure it’s a directory (not a file):
        ```cmd
        dir C:\Python /AD
        ```
        If it’s a file, rename or delete it:
        ```cmd
        ren C:\Python C:\Python.bak
        ```
      - Check permissions:
        ```cmd
        icacls C:\Python
        ```
        Grant Administrators full control:
        ```cmd
        icacls C:\Python /grant Administrators:F /T
        ```
      - If `C:\Python` contains `python.exe`, verify the version:
        ```cmd
        C:\Python\python.exe --version
        ```
        If not 3.11, uninstall the existing Python or delete `C:\Python` and rerun the script.
      - Ensure you’re running as Administrator:
        ```cmd
        net session
        ```
        If it fails, reopen Command Prompt as Administrator.
      - If the issue persists, check `setup.log` for details and ensure no other process is locking `C:\Python`.

- **RadioSportChat Installer Not Found**:
  - Verify `RadioSportChat-*.msi` is in the script's directory.
  - Check the log (`radiosportchat_install.log`) for installer errors.
  - Ensure RadioSportChat is installed at `C:\Program Files\Radiosport\RadioSportChat` or a user-specific path.

- **Syntax Error in RadioSportChat Installation Detection**:
  - If you see `The syntax of the command is incorrect` after `Starting RadioSportChat Installation Detection...`:
    - Ensure the script uses the corrected condition:
      ```bat
      ) else if exist "%USER_INSTALL_DIR3%\RadioSportChat.exe" (
      ```
    - Verify the installation directories:
      ```cmd
      echo %SYSTEM_INSTALL_DIR%
      echo %USER_INSTALL_DIR1%
      echo %USER_INSTALL_DIR2%
      echo %USER_INSTALL_DIR3%
      ```
      Expected output:
      ```
      C:\Program Files\Radiosport\RadioSportChat
      %LOCALAPPDATA%\Programs\RadioSportChat
      %LOCALAPPDATA%\Programs\Radiosport\RadioSportChat
      %LOCALAPPDATA%\Programs\Unknown Developer\RadioSportChat
      ```
    - Check if `RadioSportChat.exe` exists:
      ```cmd
      dir "%SYSTEM_INSTALL_DIR%\RadioSportChat.exe"
      dir "%USER_INSTALL_DIR1%\RadioSportChat.exe"
      dir "%USER_INSTALL_DIR2%\RadioSportChat.exe"
      dir "%USER_INSTALL_DIR3%\RadioSportChat.exe"
      ```
    - If missing, run `RadioSportChat-*.msi` manually.

- **Python Installation Fails**:
  - Confirm `python-3.11.9-amd64.exe` is in the script's directory.
  - If `C:\Python` exists, verify `python.exe`:
    ```cmd
    dir C:\Python\python.exe
    C:\Python\python.exe --version
    ```
    If missing or not 3.11, delete `C:\Python` and rerun the script.
  - Check `setup.log` for errors like `Error: Failed to install Python 3.11...`.

- **Excessive Console Output During Pip Installation**:
  - If you see multiple `processed file C:\venv ...` messages in the console:
    - These messages are from pip’s stderr, which the script does not suppress (uses `>nul` for stdout only).
    - To suppress both stdout and stderr, modify the `pip install` command in `setup.bat`:
      ```bat
      pip install --no-index --find-links=%SCRIPT_DIR%venv -r %SCRIPT_DIR%venv\requirements.txt >nul 2>nul
      ```
      This redirects stderr (`2>nul`) in addition to stdout (`>nul`).
    - Test manually to confirm:
      ```cmd
      call "C:\venv\Scripts\activate.bat"
      pip install --no-index --find-links=%SCRIPT_DIR%venv -r %SCRIPT_DIR%venv\requirements.txt >nul 2>nul
      call "C:\venv\Scripts\deactivate.bat"
      ```
      If `processed file` messages still appear, they may come from another command (e.g., `pip show`).
    - Capture pip output for debugging:
      ```cmd
      call "C:\venv\Scripts\activate.bat"
      pip install --no-index --find-links=%SCRIPT_DIR%venv -r %SCRIPT_DIR%venv\requirements.txt > pip_output.log 2>&1
      type pip_output.log
      call "C:\venv\Scripts\deactivate.bat"
      ```
      Check `pip_output.log` for `processed file` messages and errors.
    - Verify `pip show` commands are not outputting to the console:
      ```bat
      pip show streamlit >nul 2>&1
      pip show langchain-ollama >nul 2>&1
      ```
      These already use `>nul 2>&1`, so they should be silent.
    - If the issue persists, check `setup.log` for pip errors and share the console output for further analysis.

- **Pip Packages Fail to Install**:
  - Ensure `venv\requirements.txt` and package files are in the `venv` subfolder.
  - Check `setup.log` for specific package errors (e.g., `[TIMESTAMP] Error: Failed to install pip packages...`).
  - Verify the `Pip packages installation completed.` message appears in the console. If missing, check `setup.log` for errors like `Error: Failed to install pip packages...`.
  - Manually run the pip command to debug:
    ```cmd
    call "C:\venv\Scripts\activate.bat"
    pip install --no-index --find-links=%SCRIPT_DIR%venv -r %SCRIPT_DIR%venv\requirements.txt
    call "C:\venv\Scripts\deactivate.bat"
    ```

- **Ollama Installation Issues**:
  - If PATH is not updated for an existing Ollama installation:
    - Verify `ollama.exe` exists in either path:
      ```cmd
      dir "C:\Program Files\Ollama\ollama.exe"
      dir "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe"
      ```
      If missing, reinstall Ollama using `OllamaSetup.exe`.
    - Check if either path is in PATH:
      ```cmd
      echo %PATH% | findstr "C:\Program Files\Ollama"
      echo %PATH% | findstr "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama"
      ```
      If not, add the appropriate path manually:
      ```cmd
      setx PATH "%PATH%;C:\Program Files\Ollama" /M
      setx PATH "%PATH%;C:\Users\%USERNAME%\AppData\Local\Programs\Ollama" /M
      ```
    - Check `setup.log` for warnings (e.g., "Ollama executable not found at C:\Program Files\Ollama\ollama.exe or C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe").
  - If installation fails, check `setup.log` for errors and ensure `OllamaSetup.exe` is accessible.
  - If the `Ollama installation completed.` message is missing, check `setup.log` for errors like `Error: Ollama installation verification failed...`.

- **Ollama Server Not Starting**:
  - Verify the server process:
    ```cmd
    tasklist | findstr /I "ollama"
    ```
    If no process is found, start it manually:
    ```cmd
    ollama serve
    ```
    Check `setup.log` for warnings (e.g., "Failed to start Ollama server").
  - Ensure `ollama.exe` exists and is in PATH:
    ```cmd
    dir "C:\Program Files\Ollama\ollama.exe"
    dir "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama\ollama.exe"
    echo %PATH% | findstr "C:\Program Files\Ollama"
    echo %PATH% | findstr "C:\Users\%USERNAME%\AppData\Local\Programs\Ollama"
    ollama --version
    ```
    If not in PATH, add it:
    ```cmd
    setx PATH "%PATH%;C:\Program Files\Ollama;C:\Users\%USERNAME%\AppData\Local\Programs\Ollama" /M
    ```
  - Confirm Ollama is functional:
    ```cmd
    ollama list
    ```
    If this fails, reinstall Ollama or check for conflicting instances.

- **Ollama Models Not Recognized**:
  - Verify `sha256-*` files in the script’s directory match the required models.
  - Check `setup.log.models` for `ollama list` output.
  - Ensure internet connectivity for model pulling.
  - Confirm the model storage location (`C:\Users\%USERNAME%\.ollama\models\blobs` or `%OLLAMA_MODELS%`) is writable.
  - If the `Ollama models check completed.` message is shown but models are missing, check `setup.log` for warnings like `Warning: Model qwen3:4b not found after pulling.`.

- **RadioSportChat Fails to Start**:
  - Verify `C:\venv\Scripts` and Ollama paths are in PATH:
    ```cmd
    echo %PATH% | findstr "C:\venv\Scripts;C:\Program Files\Ollama;C:\Users\%USERNAME%\AppData\Local\Programs\Ollama"
    ```
    If missing, add them:
    ```cmd
    setx PATH "%PATH%;C:\venv\Scripts;C:\Program Files\Ollama;C:\Users\%USERNAME%\AppData\Local\Programs\Ollama" /M
    ```
  - Restart your command prompt or computer.
  - Check `setup.log` for missing dependencies or models.
  - Confirm `RadioSportChat.exe` exists at `C:\Program Files\Radiosport\RadioSportChat`.

- **Missing Success Messages**:
  - If a section’s success message (e.g., `Python 3.11 installation completed.`) is not shown, the section likely failed.
  - Check `setup.log` for errors or warnings in that section (e.g., `Error: Python installer not found...`).
  - Verify the script exited cleanly (no `Press any key to exit...` without `Final instructions completed.`).
  - Rerun the script and note which success messages are missing.

- **General Issues**:
  - Review `setup.log` and `setup.log.models` for detailed errors.
  - Ensure sufficient disk space (10+ GB recommended).
  - Contact support with logs if issues persist.

## Notes
- **Custom Model Storage**: Set the `OLLAMA_MODELS` environment variable to use a non-default storage path.
- **Offline Setup**: Include `sha256-*` files and `OllamaSetup.exe` in the script’s directory to avoid downloads.
- **Logs**: Always check `setup.log` for a complete record of the setup process.
- **PATH Changes**: Restart your system to apply PATH updates for `C:\venv\Scripts`, `C:\Program Files\Ollama`, or `C:\Users\%USERNAME%\AppData\Local\Programs\Ollama` (if added).

## License
This script is provided as-is for use with RadioSportChat. No warranty is implied. Use at your own risk.

---
*Generated on May 16, 2025*