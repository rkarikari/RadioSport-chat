# RadioSportChat Setup Script

## Overview

The `setup.bat` script automates the installation and configuration of **RadioSportChat**, a chat application powered by Ollama models. It installs dependencies, sets up a virtual environment, verifies model storage, and ensures all required components are properly configured on a Windows system. The script provides real-time progress feedback and logs all actions for troubleshooting. All required files are located in the script's directory or its `venv` subfolder.

### Key Features
- Installs RadioSportChat, Python 3.11, and Ollama.
- Creates and configures a virtual environment at `C:\venv`.
- Installs pip packages from `venv\requirements.txt` in the script's directory.
- Verifies and configures Ollama model storage (default: `C:\Users\%USERNAME%\.ollama\models\blobs`).
- Copies `sha256-*` model files and pulls required models (`granite3.3:2b`, `nomic-embed-text:latest`, `qwen3:4b`, `qwen3:1.7b`).
- Performs `ollama list` checks to verify model recognition before and after copying/pulling.
- Updates system PATH for script execution.
- Provides detailed real-time progress with timestamps in `setup.log`.

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
- **Optional**: `OllamaSetup.exe` in the script's directory (downloaded if absent).
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
   - The script displays real-time progress with timestamps (e.g., `[13:45:23] Installing Python 3.11...`).
   - All actions are logged to `setup.log` and `setup.log.models` (for `ollama list` output) in the script's directory.
   - Watch for warnings (e.g., missing models) or errors (e.g., permission issues).

4. **Post-Setup**:
   - After completion, run `RadioSportChat.exe` from its installation directory (e.g., `C:\Program Files\RadioSportChat`).
   - If the app fails to start, check `setup.log` and follow the troubleshooting steps below.
   - Restart your command prompt or computer to apply PATH changes.

## Script Functionality

### Installation Steps
- **RadioSportChat**: Installs from `RadioSportChat-*.msi` or skips if already installed.
- **Python 3.11**: Installs to `C:\Python` from `python-3.11.9-amd64.exe` if not present, verifies version.
- **Virtual Environment**: Creates at `C:\venv`, ensures functionality.
- **Pip Packages**: Installs from `venv\requirements.txt`, verifies `streamlit` and `langchain-ollama`.
- **Ollama**: Installs to `C:\Program Files\Ollama` from `OllamaSetup.exe` or uses existing installation.

### Model Handling
- **Storage Verification**: Checks default (`C:\Users\%USERNAME%\.ollama\models\blobs`), custom (`%OLLAMA_MODELS%`), and alternative locations.
- **Model Files**: Copies `sha256-*` files from the script's directory to the verified storage location.
- **Model Pulling**: Pulls required models if not present.
- **Verification**: Runs `ollama list` before and after copying/pulling to confirm model recognition.

### Feedback
- **Real-Time Progress**: Displays timestamped messages for each step (e.g., `[13:45:23] Pulling granite3.3:2b model...`).
- **Logging**: Saves all output to `setup.log` and model lists to `setup.log.models`.
- **Warnings/Errors**: Highlights issues (e.g., "Warning: Model qwen3:4b not found") with actionable advice.

## Troubleshooting

- **Script Fails with Permission Error**:
  - Ensure you run `setup.bat` as Administrator.
  - Check write permissions for `C:\Python`, `C:\venv`, and the Ollama model storage directory.

- **RadioSportChat Installer Not Found**:
  - Verify `RadioSportChat-*.msi` is in the script's directory.
  - Check the log (`radiosportchat_install.log`) for installer errors.

- **Python Installation Fails**:
  - Confirm `python-3.11.9-amd64.exe` is in the script's directory.
  - Uninstall any existing Python versions and retry.

- **Pip Packages Fail to Install**:
  - Ensure `venv\requirements.txt` and package files are in the `venv` subfolder.
  - Check `setup.log` for specific package errors.

- **Ollama Models Not Recognized**:
  - Verify `sha256-*` files in the script's directory match the required models.
  - Check `setup.log.models` for `ollama list` output.
  - Ensure internet connectivity for model pulling.
  - Confirm the model storage location (`C:\Users\%USERNAME%\.ollama\models\blobs` or `%OLLAMA_MODELS%`) is writable.

- **RadioSportChat Fails to Start**:
  - Add `C:\venv\Scripts` to the system PATH manually:
    ```cmd
    setx PATH "%PATH%;C:\venv\Scripts" /M
    ```
  - Restart your command prompt or computer.
  - Check `setup.log` for missing dependencies or models.

- **General Issues**:
  - Review `setup.log` and `setup.log.models` for detailed errors.
  - Ensure sufficient disk space (10+ GB recommended).
  - Contact support with logs if issues persist.

## Notes
- **Custom Model Storage**: Set the `OLLAMA_MODELS` environment variable to use a non-default storage path.
- **Offline Setup**: Include `sha256-*` files and `OllamaSetup.exe` in the script's directory to avoid downloads.
- **Logs**: Always check `setup.log` for a complete record of the setup process.
- **PATH Changes**: Restart your system to apply PATH updates for `C:\venv\Scripts`.

## License
This script is provided as-is for use with RadioSportChat. No warranty is implied. Use at your own risk.

---
*Generated on May 16, 2025*