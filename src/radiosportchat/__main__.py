import subprocess
import os

# Get the path to main.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MAIN_SCRIPT = os.path.join(BASE_DIR, "rag.py")

# Run the Streamlit app
if __name__ == "__main__":
    subprocess.run(["streamlit", "run", MAIN_SCRIPT], check=True, creationflags=subprocess.CREATE_NO_WINDOW)
