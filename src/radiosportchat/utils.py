import os
import streamlit as st
import shutil
import time
import logging
import json
import requests
import bcrypt
from tenacity import retry, stop_after_attempt, wait_fixed
from config import APP_VERSION
from rag_pipeline import embedchain_bot

# Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_assistant")

# Version tracking
logger.info(f"Application version: {APP_VERSION}")

def initialize_credentials():
    """Initialize admin_credentials in session state."""
    credentials = {
        "admin": "$2b$12$pAQdUIvM9kgJtdhL91Tn/.ePnQ4MzKlUX7KKKY2vO7Y2ivDzkwA8y",  # Hash for "admin123"
        "user": "$2b$12$1ee8jsGcBYcxdvkv26cjXurxLcuAIQ3BOOdYmp7Cv73pQT0K5go4K"   # Hash for "user123"
    }
    if 'admin_credentials' not in st.session_state or not st.session_state.admin_credentials:
        st.session_state.admin_credentials = credentials
        logger.debug("Initialized admin_credentials in session state")
    else:
        logger.debug(f"admin_credentials already initialized: {list(st.session_state.admin_credentials.keys())}")
    return credentials

# Attempt module-level initialization with fallback
try:
    initialize_credentials()
except Exception as e:
    logger.error(f"Failed to initialize credentials at module level: {str(e)}")

@retry(stop=stop_after_attempt(3), wait=wait_fixed(20))
def safe_rmtree(path, attempt=1, debug_mode=False):
    try:
        if debug_mode:
            logger.debug(f"Attempt {attempt} to delete directory: {path}")
        if os.path.exists(path):
            shutil.rmtree(path)
            if debug_mode:
                logger.debug(f"Successfully deleted directory: {path}")
        else:
            if debug_mode:
                logger.debug(f"Directory does not exist: {path}")
    except Exception as e:
        logger.error(f"Error deleting directory {path} on attempt {attempt}: {str(e)}")
        raise

def fetch_ollama_models(debug_mode=False):
    """Fetch available models from Ollama server."""
    try:
        if debug_mode:
            logger.debug("Fetching available models from Ollama server")
        url = "http://localhost:11434/api/tags"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        models = response.json().get("models", [])
        model_names = [model["name"] for model in models]
        if debug_mode:
            logger.debug(f"Retrieved models: {model_names}")
        return model_names
    except Exception as e:
        logger.error(f"Failed to fetch Ollama models: {str(e)}")
        return []

def get_available_embedders(debug_mode=False):
    """Get available embedders (all-minilm* and nomic-embed*) from Ollama server."""
    models = fetch_ollama_models(debug_mode)
    embedders = [model for model in models if model.startswith(("all-minilm", "nomic-embed"))]
    if debug_mode:
        logger.debug(f"Available embedders: {embedders}")
    return embedders if embedders else ["nomic-embed-text:latest"]

def get_available_llms(debug_mode=False):
    """Get available language models, excluding embedders."""
    models = fetch_ollama_models(debug_mode)
    llms = [model for model in models if not model.startswith(("all-minilm", "nomic-embed"))]
    if debug_mode:
        logger.debug(f"Available LLMs: {llms}")
    return llms if llms else ["granite3.3:2b"]

def ollama_raw_stream(prompt, debug_mode=False):
    try:
        if debug_mode:
            logger.debug(f"Ollama raw streaming with prompt: '{prompt[:50]}...'")
        start_time = time.time()
        chunk_count = 0
        history = ""
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                history += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                history += f"Assistant: {msg['content']}\n"
        full_prompt = f"{history}User: {prompt}\nAssistant: " if history else f"User: {prompt}\nAssistant: "
        if debug_mode:
            logger.debug(f"Full prompt with history: '{full_prompt[:100]}...'")

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": st.session_state.selected_llm,  # Use selected LLM
            "prompt": full_prompt,
            "stream": True
        }
        with requests.post(url, json=payload, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    chunk_count += 1
                    chunk = line.decode('utf-8')
                    try:
                        json_data = json.loads(chunk)
                        text = json_data.get('response', '')
                    except json.JSONDecodeError:
                        text = chunk
                    if debug_mode:
                        logger.debug(
                            f"Ollama raw chunk {chunk_count} received at {time.time() - start_time:.2f}s: "
                            f"'{chunk[:50]}...' (len={len(chunk)}), text: '{text[:50]}...'"
                        )
                    yield text
        duration = time.time() - start_time
        if debug_mode:
            logger.debug(f"Ollama raw streaming completed in {duration:.2f}s, total chunks: {chunk_count}")
    except Exception as e:
        logger.error(f"Ollama raw streaming error: {str(e)}")
        yield f"Error: {str(e)}"

def save_chat_history():
    try:
        history_file = "C:\\ProgramData\\RadioSport\\RadioSportChat\\chat_history.json"
        os.makedirs(os.path.dirname(history_file), exist_ok=True)
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
        if st.session_state.debug_enabled:
            logger.debug(f"Saved chat history to {history_file}")
    except Exception as e:
        logger.error(f"Failed to save chat history: {str(e)}")

def load_chat_history():
    try:
        history_file = "C:\\ProgramData\\RadioSport\\RadioSportChat\\chat_history.json"
        if os.path.exists(history_file):
            with open(history_file, "r", encoding="utf-8") as f:
                messages = json.load(f)
                if isinstance(messages, list):
                    st.session_state.messages = messages
                    if st.session_state.debug_enabled:
                        logger.debug(f"Loaded {len(messages)} messages from {history_file}")
                else:
                    logger.warning(f"Invalid chat history format in {history_file}")
        else:
            if st.session_state.debug_enabled:
                logger.debug(f"No chat history file found at {history_file}")
    except Exception as e:
        logger.error(f"Failed to load chat history: {str(e)}")

def save_config():
    try:
        config_file = "C:\\ProgramData\\RadioSport\\RadioSportChat\\config.json"
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        config = {
            "use_streaming": st.session_state.use_streaming,
            "selected_embedder": st.session_state.selected_embedder,
            "selected_llm": st.session_state.selected_llm
        }
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        if st.session_state.debug_enabled:
            logger.debug(f"Saved config to {config_file}")
    except Exception as e:
        logger.error(f"Failed to save config: {str(e)}")

def load_config():
    try:
        config_file = "C:\\ProgramData\\RadioSport\\RadioSportChat\\config.json"
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                if isinstance(config, dict):
                    st.session_state.use_streaming = config.get("use_streaming", False)
                    st.session_state.selected_embedder = config.get("selected_embedder", "nomic-embed-text:latest")
                    st.session_state.selected_llm = config.get("selected_llm", "granite3.3:2b")
                    if st.session_state.debug_enabled:
                        logger.debug(f"Loaded config from {config_file}: use_streaming={st.session_state.use_streaming}, "
                                     f"selected_embedder={st.session_state.selected_embedder}, "
                                     f"selected_llm={st.session_state.selected_llm}")
                else:
                    logger.warning(f"Invalid config format in {config_file}")
        else:
            if st.session_state.debug_enabled:
                logger.debug(f"No config file found at {config_file}")
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")

def authenticate(username, password):
    logger.debug(f"Authentication attempt for username: {username}")
    # Ensure credentials are initialized
    credentials = initialize_credentials()
    logger.debug(f"Current admin_credentials keys: {list(st.session_state.admin_credentials.keys())}")
    if not username or not password:
        logger.warning("Authentication failed: Empty username or password")
        return False
    if username in credentials:
        logger.debug(f"User {username} found in credentials")
        try:
            hashed_password = credentials[username].encode('utf-8')
            if bcrypt.checkpw(password.encode('utf-8'), hashed_password):
                logger.info(f"Authentication successful for user: {username}")
                return True
            else:
                logger.warning(f"Authentication failed for user: {username} (incorrect password)")
                return False
        except Exception as e:
            logger.error(f"Authentication error for user {username}: {str(e)}")
            return False
    else:
        logger.warning(f"Authentication failed: Unknown user: {username}")
        return False

def get_app(db_path, debug_mode=False):
    try:
        if debug_mode:
            logger.debug(f"Creating app with db_path: {db_path}")
        app = embedchain_bot(db_path=db_path, debug_mode=debug_mode)
        return app
    except Exception as e:
        logger.error(f"Failed to create app in get_app: {str(e)}")
        raise

def initialize_app():
    try:
        if st.session_state.debug_enabled:
            logger.debug("Attempting to initialize app")
        db_path = "C:\\ProgramData\\RadioSport\\RadioSportChat"
        os.makedirs(db_path, exist_ok=True)
        st.session_state.db_dir = db_path

        load_config()
        load_chat_history()

        if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, 'db'):
            try:
                count = st.session_state.app.db.count()
                if st.session_state.debug_enabled:
                    logger.debug(f"App already initialized, document count: {count}")
                return
            except Exception as e:
                logger.warning(f"Existing app is invalid: {str(e)}, reinitializing")

        try:
            st.session_state.app = get_app(db_path, debug_mode=st.session_state.debug_enabled)
        except Exception as e:
            logger.warning(f"Detected DB initialization error: {str(e)}, attempting reset")
            if hasattr(st.session_state, 'app'):
                try:
                    st.session_state.app.force_reset_db(debug_mode=st.session_state.debug_enabled)
                    if st.session_state.debug_enabled:
                        logger.debug("Force reset DB before clearing directory")
                except Exception as e:
                    logger.warning(f"Failed to force reset DB: {str(e)}")
            try:
                safe_rmtree(db_path, debug_mode=st.session_state.debug_enabled)
                os.makedirs(db_path, exist_ok=True)
                st.session_state.app = get_app(db_path, debug_mode=st.session_state.debug_enabled)
                if st.session_state.debug_enabled:
                    logger.debug("Successfully reinitialized app after clearing database")
            except Exception as e:
                logger.error(f"Failed to reinitialize database: {str(e)}")
                raise
    except Exception as e:
        st.error(
            f"Critical initialization failure: {str(e)}. "
            "Please close Ollama server, Streamlit, and other processes, "
            "disable antivirus temporarily, and manually delete 'C:\\ProgramData\\RadioSport\\RadioSportChat'. "
            "If the issue persists, restart your computer."
        )
        logger.error(f"App initialization error: {str(e)}")
        raise