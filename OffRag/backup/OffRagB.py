import os
import shutil
import tempfile
import base64
import re
import time
import logging
from datetime import datetime
import chromadb
import embedchain
from embedchain.app import App
from tenacity import retry, stop_after_attempt, wait_fixed
import streamlit as st
from streamlit_chat import message
from PIL import Image
import pytesseract
from io import BytesIO
import numpy as np
import concurrent.futures
import pdfplumber
import multiprocessing
import bcrypt
import pkg_resources
import requests
import json
import uuid
import types  # For generator detection
import hashlib  # For generating embedding IDs

# Version tracking
APP_VERSION = "v1.0.3"

# Uncomment and update if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.environ.pop("OPENAI_API_KEY", None)  # Enforce offline mode

# --- GUI Configuration ---
st.set_page_config(
    page_title="RadioSport Chat",
    page_icon="🌍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Include MathJax for LaTeX rendering (locally hosted)
st.markdown(
    """
    <script src="static/mathjax/tex-chtml.js?config=TeX-AMS-MML_HTMLorMML"></script>
    <script type="text/x-mathjax-config">
        MathJax.Hub.Config({
            tex2jax: {
                inlineMath: [['$', '$'], ['\\(', '\\)']],
                displayMath: [['$$', '$$'], ['\\[', '\\]']],
                processEscapes: true
            }
        });
    </script>
    """,
    unsafe_allow_html=True
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_assistant")

# Log versions
logger.info(f"Using chromadb version: {chromadb.__version__}")
logger.info(f"Using embedchain version: {embedchain.__version__}")
logger.info(f"Application version: {APP_VERSION}")
try:
    streamlit_version = pkg_resources.get_distribution("streamlit").version
    logger.info(f"Using streamlit version: {streamlit_version}")
except pkg_resources.DistributionNotFound:
    logger.warning("Streamlit version not found")
    streamlit_version = "unknown"

# Verify Streamlit version
if streamlit_version != "unknown":
    try:
        from packaging import version
        if version.parse(streamlit_version) < version.parse("1.12.0"):
            st.error(
                "Streamlit version <1.12.0 lacks st.write_stream, limiting streaming capabilities. "
                "Upgrade to 1.42.0 or higher: `pip install --upgrade streamlit`"
            )
            logger.error("Streamlit version <1.12.0 detected; st.write_stream unavailable")
    except Exception as e:
        logger.warning(f"Failed to parse Streamlit version: {str(e)}")

# --- RAG Pipeline: DebugApp Class ---
class DebugApp(App):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.debug_info = {
            "add_operations": [],
            "query_operations": [],
            "current_session": {
                "chunks": [],
                "embeddings": [],
                "retrieved_docs": [],
                "contexts": [],
                "prompt": "",
                "response": "",
                "streaming_info": {"chunk_count": 0, "response_type": "", "chunks_received": []},
            },
        }
        self._embedding_dimension = None  # Lazy computation

    @property
    def embedder(self):
        if hasattr(self, '_embedder'):
            return self._embedder
        try:
            if hasattr(self, 'db') and hasattr(self.db, 'embedder'):
                self._embedder = self.db.embedder
                if st.session_state.debug_enabled:
                    logger.debug("Using embedder from self.db.embedder")
            elif hasattr(self, 'config') and 'embedder' in self.config:
                from embedchain.embedder.ollama import OllamaEmbedder
                embedder_config = self.config['embedder'].get('config', {})
                self._embedder = OllamaEmbedder(**embedder_config)
                if st.session_state.debug_enabled:
                    logger.debug("Created embedder from config")
            else:
                from embedchain.embedder.ollama import OllamaEmbedder
                self._embedder = OllamaEmbedder(
                    model="nomic-embed-text:latest",
                    base_url="http://localhost:11434"
                )
                if st.session_state.debug_enabled:
                    logger.debug("Created default embedder")
            return self._embedder
        except Exception as e:
            logger.error(f"Failed to get/create embedder: {str(e)}")
            raise

    @embedder.setter
    def embedder(self, value):
        self._embedder = value

    def _compute_embedding_dimension(self, text="test"):
        """Lazily compute embedding dimension."""
        if self._embedding_dimension is None:
            try:
                embeddings = self.embedder.to_embeddings([text])
                embedding = embeddings[0] if isinstance(embeddings, list) else embeddings
                self._embedding_dimension = len(embedding)
            except Exception as e:
                logger.error(f"Failed to compute embedding dimension: {str(e)}")
                self._embedding_dimension = 0
        return self._embedding_dimension

    def _generate_embedding_id(self, text, file_name="Unknown"):
        """Generate a unique embedding ID based on text and file name."""
        content = f"{file_name}:{text}"
        return f"default-app-id--{hashlib.sha256(content.encode('utf-8')).hexdigest()}"

    def add(self, *args, debug_mode=False, file_name="Unknown", **kwargs):
        start_time = time.time()
        data_type = kwargs.get("data_type", "unknown")
        text_snippet = str(args[0])[:100] + "..." if args and args[0] else "No text"
        if debug_mode:
            logger.debug(f"Adding document (type={data_type}, file_name={file_name}): snippet='{text_snippet}'")

        try:
            if not args or not args[0].strip():
                raise ValueError("Empty text provided to embed")
            text = args[0]
            embedding_id = self._generate_embedding_id(text, file_name)
            # Check if embedding ID already exists
            try:
                collection = self.db.client.get_collection("default")
                existing = collection.get(ids=[embedding_id])
                if existing['ids']:
                    if debug_mode:
                        logger.info(f"Skipping duplicate embedding ID: {embedding_id}")
                    duration = time.time() - start_time
                    operation_info = {
                        "timestamp": datetime.now().isoformat(),
                        "data_type": data_type,
                        "file_name": file_name,
                        "text_snippet": text_snippet,
                        "duration": duration,
                        "success": True,
                        "embedding_dimension": self._compute_embedding_dimension(),
                        "status": "Skipped (Duplicate)",
                        "total_chunks": 1,
                        "successful_chunks": 0,
                        "total_duration": duration,
                        "avg_duration": duration,
                    }
                    self.debug_info["add_operations"].append(operation_info)
                    return operation_info
            except Exception as e:
                if debug_mode:
                    logger.warning(f"Error checking existing embedding ID {embedding_id}: {str(e)}")

            embedding_dimension = self._compute_embedding_dimension() if debug_mode else 0
            result = super().add(text, **kwargs)
            if result is None:
                raise ValueError("Embedding result is None")
            duration = time.time() - start_time
            operation_info = {
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "file_name": file_name,
                "text_snippet": text_snippet,
                "duration": duration,
                "success": True,
                "embedding_dimension": embedding_dimension,
                "status": "Success",
                "total_chunks": 1,
                "successful_chunks": 1,
                "total_duration": duration,
                "avg_duration": duration,
            }
        except Exception as e:
            logger.error(f"Error in add method: {e}")
            duration = time.time() - start_time
            operation_info = {
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "file_name": file_name,
                "text_snippet": text_snippet,
                "duration": duration,
                "success": False,
                "error": str(e),
                "embedding_dimension": 0,
                "status": "Failed",
                "total_chunks": 1,
                "successful_chunks": 0,
                "total_duration": duration,
                "avg_duration": duration,
            }
        if debug_mode:
            logger.debug(f"Operation info created: {operation_info}")
        self.debug_info["add_operations"].append(operation_info)
        if len(self.debug_info["add_operations"]) > 100:
            self.debug_info["add_operations"] = self.debug_info["add_operations"][-100:]
        return operation_info

    def chat(self, prompt, debug_mode=False, **kwargs):
        try:
            if debug_mode:
                logger.debug(f"Chat called with prompt: '{prompt[:50]}...'")
            self.debug_info["current_session"] = {
                "prompt": prompt,
                "response": "",
                "retrieved_docs": [],
                "contexts": [],
                "chunks": [],
                "embeddings": [],
                "streaming_info": {"chunk_count": 0, "response_type": "", "chunks_received": []},
            }
            start_time = time.time()
            chunk_count = 0
            # Remove 'stream' from kwargs to avoid duplication
            kwargs = {k: v for k, v in kwargs.items() if k != 'stream'}
            # Explicitly enable streaming
            response = super().chat(prompt, stream=True, **kwargs)
            if debug_mode:
                response_type = type(response).__name__
                self.debug_info["current_session"]["streaming_info"]["response_type"] = response_type
                logger.debug(f"super().chat response type: {response_type}")
            # Ensure response is iterable
            if not hasattr(response, '__iter__') or isinstance(response, (str, bytes)):
                if debug_mode:
                    logger.warning(f"Non-iterable response from super().chat: {type(response).__name__}, wrapping as generator")
                response = iter([str(response)])  # Wrap non-iterable response
            for chunk in response:
                chunk_count += 1
                if isinstance(chunk, dict):
                    chunk = chunk.get('text', '')  # Handle dict chunks
                elif not isinstance(chunk, str):
                    chunk = str(chunk)  # Convert non-string chunks
                if debug_mode:
                    self.debug_info["current_session"]["streaming_info"]["chunks_received"].append({
                        "chunk_number": chunk_count,
                        "content": chunk[:50] + "..." if len(chunk) > 50 else chunk,
                        "length": len(chunk),
                        "timestamp": time.time() - start_time,
                    })
                    logger.debug(
                        f"Raw chunk {chunk_count} from super().chat at {time.time() - start_time:.2f}s: "
                        f"'{chunk[:50]}...' (len={len(chunk)})"
                    )
                # Filter out unwanted tags
                filtered_chunk = re.sub(r"<think>.*?</think>", "", chunk, flags=re.DOTALL)
                if filtered_chunk.strip():
                    if debug_mode:
                        logger.debug(
                            f"Yielding filtered chunk {chunk_count} at {time.time() - start_time:.2f}s: "
                            f"'{filtered_chunk[:50]}...' (len={len(filtered_chunk)})"
                        )
                    yield filtered_chunk
                    self.debug_info["current_session"]["response"] += filtered_chunk
            # Check if response is empty
            if not self.debug_info["current_session"]["response"].strip():
                if debug_mode:
                    logger.warning("Empty response from chat, returning unavailable message")
                self.debug_info["current_session"]["response"] = "Sorry, the requested information is unavailable at this time"
                yield self.debug_info["current_session"]["response"]
            # Update chunk count
            if debug_mode:
                self.debug_info["current_session"]["streaming_info"]["chunk_count"] = chunk_count
                logger.debug(f"Chat response completed in {time.time() - start_time:.2f}s, total chunks: {chunk_count}")
            st.session_state.debug_sessions.append(self.debug_info["current_session"])
            if len(st.session_state.debug_sessions) > 100:
                st.session_state.debug_sessions = st.session_state.debug_sessions[-100:]
            return self.debug_info["current_session"]["response"]
        except Exception as e:
            logger.error(f"Chat error: {e}")
            self.debug_info["current_session"]["response"] = "Sorry, the requested information is unavailable at this time"
            st.session_state.debug_sessions.append(self.debug_info["current_session"])
            yield self.debug_info["current_session"]["response"]
            if debug_mode:
                logger.debug("Yielded unavailable message due to error")

    def force_reset_db(self, debug_mode=False):
        try:
            if hasattr(self, 'db') and hasattr(self.db, 'client'):
                if debug_mode:
                    logger.debug("Force resetting ChromaDB database")
                client = self.db.client
                collections = client.list_collections()
                for collection in collections:
                    if debug_mode:
                        logger.debug(f"Deleting collection: {collection.name}")
                    client.delete_collection(collection.name)
                try:
                    if hasattr(client, '_system') and client._system is not None:
                        if debug_mode:
                            logger.debug("Stopping ChromaDB client system")
                        client._system.stop()
                        time.sleep(1)
                    if hasattr(client, '_system') and hasattr(client._system, 'persistence'):
                        persistence = client._system.persistence
                        if hasattr(persistence, 'db'):
                            if debug_mode:
                                logger.debug("Closing SQLite connections")
                            persistence.db.close()
                            time.sleep(1)
                except Exception as e:
                    logger.warning(f"Failed to stop client system or close connections: {str(e)}")
                self.db = None
            else:
                logger.warning("No valid DB client found for force reset")
        except Exception as e:
            logger.error(f"Error force resetting ChromaDB: {str(e)}")
            raise

    def test_embedding(self, text):
        try:
            start_time = time.time()
            embeddings = self.embedder.to_embeddings([text])
            embedding = embeddings[0] if isinstance(embeddings, list) else embeddings
            if isinstance(embedding, np.float32):
                if st.session_state.debug_enabled:
                    logger.error("Detected single float embedding, raising TypeError")
                raise TypeError("Embedding is a single float value")
            self.add(text, data_type="text", debug_mode=st.session_state.debug_enabled, file_name="Test Input")
            duration = time.time() - start_time
            if st.session_state.debug_enabled:
                logger.debug(f"Embedding test successful, dimension: {len(embedding)}, duration: {duration:.2f}s")
            return {
                "success": True,
                "embedding_dimension": len(embedding),
                "embedding_sample": embedding[:5].tolist(),
                "duration": duration,
            }
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            return {"success": False, "error": str(e)}

    def get_debug_info(self):
        return self.debug_info

# --- RAG Pipeline: Utility Functions ---
def embedchain_bot(db_path, debug_mode=False):
    try:
        if debug_mode:
            logger.debug(f"Creating DebugApp with db_path: {db_path}")
        app = DebugApp.from_config(
            config={
                "llm": {
                    "provider": "ollama",
                    "config": {
                        "model": "granite3.3:2b",
                        "max_tokens": 500,
                        "temperature": 0.5,
                        "stream": True,
                        "base_url": "http://localhost:11434",
                    },
                },
                "vectordb": {
                    "provider": "chroma",
                    "config": {"dir": db_path},
                },
                "embedder": {
                    "provider": "ollama",
                    "config": {
                        "model": "nomic-embed-text:latest",
                        "base_url": "http://localhost:11434",
                    },
                },
            }
        )
        if debug_mode:
            logger.debug("DebugApp created successfully")
        return app
    except Exception as e:
        logger.error(f"Failed to create DebugApp: {str(e)}")
        raise

def display_file(file):
    if file is None:
        return
    try:
        file.seek(0)
        mime_type = file.type
        if mime_type == "application/pdf":
            base64_pdf = base64.b64encode(file.read()).decode("utf-8")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                'width="100%" height="600px" frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True,
            )
        elif mime_type.startswith("image/"):
            st.image(file, use_container_width=True)
        elif mime_type.startswith("audio/"):
            st.audio(file, format=mime_type)
        elif mime_type.startswith("video/"):
            st.video(file)
        elif mime_type == "text/plain":
            text_content = file.read().decode("utf-8")
            st.text(text_content)
    except Exception as e:
        st.error(f"Preview error: {str(e)}")

def get_app(db_path, debug_mode=False):
    try:
        if debug_mode:
            logger.debug(f"Creating app with db_path: {db_path}")
        app = embedchain_bot(db_path=db_path, debug_mode=debug_mode)
        return app
    except Exception as e:
        logger.error(f"Failed to create app in get_app: {str(e)}")
        raise

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

def ollama_raw_stream(prompt, debug_mode=False):
    """Utility function for Ollama raw streaming with conversation history."""
    try:
        if debug_mode:
            logger.debug(f"Ollama raw streaming with prompt: '{prompt[:50]}...'")
        start_time = time.time()
        chunk_count = 0
        response_text = ""
        
        # Build conversation history
        history = ""
        for msg in st.session_state.messages[:-1]:  # Exclude the current prompt
            if msg["role"] == "user":
                history += f"User: {msg['content']}\n"
            elif msg["role"] == "assistant":
                history += f"Assistant: {msg['content']}\n"
        # Append current prompt
        full_prompt = f"{history}User: {prompt}\nAssistant: " if history else f"User: {prompt}\nAssistant: "
        if debug_mode:
            logger.debug(f"Full prompt with history: '{full_prompt[:100]}...'")

        url = "http://localhost:11434/api/generate"
        payload = {
            "model": "granite3.3:2b",
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
                        text = chunk  # Fallback to raw chunk if not JSON
                    if debug_mode:
                        logger.debug(
                            f"Ollama raw chunk {chunk_count} received at {time.time() - start_time:.2f}s: "
                            f"'{chunk[:50]}...' (len={len(chunk)}), text: '{text[:50]}...'"
                        )
                    response_text += text
                    yield text
        duration = time.time() - start_time
        if not response_text.strip():
            if debug_mode:
                logger.warning("Empty response from Ollama, yielding unavailable message")
            yield "Sorry, the requested information is unavailable at this time"
        if debug_mode:
            logger.debug(f"Ollama raw streaming completed in {duration:.2f}s, total chunks: {chunk_count}")
    except Exception as e:
        logger.error(f"Ollama raw streaming error: {str(e)}")
        yield "Sorry, the requested information is unavailable at this time"

def save_chat_history():
    """Save chat history to a JSON file."""
    try:
        history_file = "chat_history.json"
        with open(history_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
        if st.session_state.debug_enabled:
            logger.debug(f"Saved chat history to {history_file}")
    except Exception as e:
        logger.error(f"Failed to save chat history: {str(e)}")

def load_chat_history():
    """Load chat history from a JSON file if it exists."""
    try:
        history_file = "chat_history.json"
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
    """Save configuration (e.g., use_streaming) to a JSON file."""
    try:
        config_file = "config.json"
        config = {"use_streaming": st.session_state.use_streaming}
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        if st.session_state.debug_enabled:
            logger.debug(f"Saved config to {config_file}")
    except Exception as e:
        logger.error(f"Failed to save config: {str(e)}")

def load_config():
    """Load configuration from a JSON file if it exists."""
    try:
        config_file = "config.json"
        if os.path.exists(config_file):
            with open(config_file, "r", encoding="utf-8") as f:
                config = json.load(f)
                if isinstance(config, dict):
                    st.session_state.use_streaming = config.get("use_streaming", False)
                    if st.session_state.debug_enabled:
                        logger.debug(f"Loaded config from {config_file}: use_streaming={st.session_state.use_streaming}")
                else:
                    logger.warning(f"Invalid config format in {config_file}")
        else:
            if st.session_state.debug_enabled:
                logger.debug(f"No config file found at {config_file}")
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")

def initialize_app():
    try:
        if st.session_state.debug_enabled:
            logger.debug("Attempting to initialize app")
        db_path = "./chroma_db"
        os.makedirs(db_path, exist_ok=True)
        st.session_state.db_dir = db_path

        # Load configuration and chat history
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
            "disable antivirus temporarily, and manually delete './chroma_db'. "
            "If the issue persists, restart your computer."
        )
        logger.error(f"App initialization error: {str(e)}")
        raise

# --- Authentication ---
def authenticate(username, password):
    logger.debug(f"Authentication attempt for username: {username}")
    if not username or not password:
        logger.warning("Authentication failed: Empty username or password")
        return False
    if username in st.session_state.admin_credentials:
        logger.debug(f"User {username} found in credentials")
        try:
            hashed_password = st.session_state.admin_credentials[username].encode('utf-8')
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

# --- File Processing ---
def estimate_total_chunks(uploaded_files):
    total_chunks = 0
    for file in uploaded_files:
        try:
            file.seek(0)
            if file.type == "application/pdf":
                with pdfplumber.open(file) as pdf:
                    pages = [page for page in pdf.pages if page.extract_text()]
                    total_chunks += len(pages)
            elif file.type.startswith("image/"):
                total_chunks += 1
            elif file.type == "text/plain":
                try:
                    text_content = file.read().decode("utf-8")
                except UnicodeDecodeError:
                    text_content = file.read().decode("latin1", errors="ignore")
                if text_content.strip():
                    chunk_size = 2000
                    chunks = [text_content[i:i + chunk_size] for i in range(0, len(text_content), chunk_size)]
                    total_chunks += len([chunk for chunk in chunks if chunk.strip()])
                else:
                    total_chunks += 1
            else:
                total_chunks += 1
            file.seek(0)
        except Exception as e:
            logger.warning(f"Error estimating chunks for {file.name}: {str(e)}")
            total_chunks += 1
    return max(total_chunks, 1)

def extract_chunks(uploaded_files, debug_mode=False):
    chunks = []
    messages = []
    errors = []
    with tempfile.TemporaryDirectory() as temp_dir:  # Auto-cleanup
        temp_files = []
        for file in uploaded_files:
            try:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                temp_files.append(temp_path)

                if file.type == "application/pdf":
                    try:
                        with pdfplumber.open(temp_path) as pdf:
                            pages_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
                    except Exception as e:
                        errors.append(f"Error extracting text from PDF {file.name}: {str(e)}")
                        logger.error(f"PDF text extraction error: {e}")
                        chunks.append((file.name, "", "text"))
                        messages.append(f"No text extracted from {file.name}")
                        continue
                    if not pages_text:
                        errors.append(f"No text extracted from {file.name}")
                        logger.warning(f"No text extracted from PDF: {file.name}")
                        chunks.append((file.name, "", "text"))
                        messages.append(f"No text extracted from {file.name}")
                        continue
                    messages.append(f"Extracted text from {len(pages_text)} pages in {file.name}")
                    for text in pages_text:
                        if text.strip():
                            chunks.append((file.name, text, "text"))
                elif file.type.startswith("image/"):
                    img = Image.open(temp_path)
                    max_size = 1024
                    if img.width > max_size or img.height > max_size:
                        img.thumbnail((max_size, max_size))
                        resized_path = os.path.join(temp_dir, f"resized_{file.name}")
                        img.save(resized_path)
                        temp_files.append(resized_path)
                        temp_path = resized_path
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if not ocr_text:
                        errors.append(f"No text extracted from {file.name}")
                        logger.warning(f"No OCR text extracted from image: {file.name}")
                        chunks.append((file.name, "", "text"))
                        messages.append(f"No text extracted from {file.name}")
                        continue
                    messages.append(f"Extracted OCR text from {file.name}")
                    chunks.append((file.name, ocr_text, "text"))
                elif file.type == "text/plain":
                    try:
                        with open(temp_path, "r", encoding="utf-8") as f:
                            text_content = f.read()
                    except UnicodeDecodeError:
                        with open(temp_path, "r", encoding="latin1") as f:
                            text_content = f.read()
                    if not text_content.strip():
                        errors.append(f"Empty text file: {file.name}")
                        logger.warning(f"Empty text file: {file.name}")
                        chunks.append((file.name, "", "text"))
                        messages.append(f"No text extracted from {file.name}")
                        continue
                    chunk_size = 2000
                    text_chunks = [text_content[i:i + chunk_size] for i in range(0, len(text_content), chunk_size)]
                    messages.append(f"Split {file.name} into {len(text_chunks)} chunks")
                    for chunk in text_chunks:
                        if chunk.strip():
                            chunks.append((file.name, chunk, "text"))
                else:
                    errors.append(f"Unsupported file type: {file.type}")
                    logger.warning(f"Unsupported file type: {file.type}")
                    chunks.append((file.name, "", "text"))
                    messages.append(f"Unsupported file type for {file.name}")
            except Exception as e:
                errors.append(f"Error processing {file.name}: {str(e)}")
                logger.error(f"File processing error: {e}")
                chunks.append((file.name, "", "text"))
                messages.append(f"Error processing {file.name}")
        return chunks, messages, errors, temp_files

def process_chunk(app, file_name, chunk_text, data_type, debug_mode, completed_chunks, total_chunks):
    try:
        if not chunk_text.strip():
            raise ValueError("Empty chunk text")
        stat = app.add(chunk_text, data_type=data_type, debug_mode=debug_mode, file_name=file_name)
        with completed_chunks.get_lock():
            completed_chunks.value += 1
        if debug_mode:
            logger.debug(f"Incremented completed_chunks to {completed_chunks.value} for chunk in {file_name}")
        status_message = f"Processed chunk {completed_chunks.value}/{total_chunks} for {file_name}: {stat['status']}"
        return file_name, stat, status_message
    except Exception as e:
        if debug_mode:
            logger.error(f"Error processing chunk for {file_name}: {str(e)}")
        with completed_chunks.get_lock():
            completed_chunks.value += 1
        if debug_mode:
            logger.debug(f"Incremented completed_chunks to {completed_chunks.value} for failed chunk in {file_name}")
        status_message = f"Processed chunk {completed_chunks.value}/{total_chunks} for {file_name}: Failed ({str(e)})"
        return file_name, {
            "success": False,
            "error": str(e),
            "duration": 0,
            "embedding_dimension": 0,
            "status": "Failed",
            "total_chunks": 1,
            "successful_chunks": 0,
            "total_duration": 0,
            "avg_duration": 0,
        }, status_message

# --- Session State Initialization ---
session_defaults = {
    "messages": [],
    "last_uploaded_image": None,
    "debug_enabled": False,
    "debug_sessions": [],
    "is_authenticated": False,
    "admin_credentials": {
        "admin": "$2b$12$G.v2ZJlD4HasyM5Yy0XYFeEysl2SCJSUX0N8RXctoLoxcHPyScf8G"  # Hash for "admin123"
    },
    "show_login_panel": True,
    "use_streaming": False,
    "partial_response": "",
    "streaming_session_id": 0,  # For debugging, not used in keys
    "streaming_active": False,  # Lock to prevent concurrent streaming
    "confirm_clear_chat": False,
    "confirm_reset_session": False,
}
for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Initialize app
try:
    initialize_app()
except Exception as e:
    st.error(
        f"Critical initialization failure: {str(e)}. "
        "Please close Ollama server, Streamlit, and other processes, "
        "disable antivirus temporarily, and manually delete './chroma_db'. "
        "If the issue persists, restart your computer."
    )
    st.stop()

# --- GUI: Sidebar ---
with st.sidebar:
    if st.session_state.is_authenticated:
        st.title("🗂️ File Management")
        st.header("Upload Your Files")
        uploaded_files = st.file_uploader(
            "Select files",
            type=["pdf", "png", "jpg", "jpeg", "txt"],
            accept_multiple_files=True,
            key="file_uploader",
        )

        if uploaded_files:
            if any(f.type.startswith("image/") for f in uploaded_files):
                st.session_state.last_uploaded_image = next(
                    (f for f in reversed(uploaded_files) if f.type.startswith("image/")), None
                )

            if st.button("🚀 Add to Knowledge Base", type="primary"):
                with st.status("Processing files...", expanded=True) as status_container:
                    if not hasattr(st.session_state, 'app'):
                        try:
                            logger.warning("st.session_state.app missing, reinitializing")
                            initialize_app()
                        except Exception as e:
                            status_container.error(f"Failed to reinitialize app: {str(e)}")
                            st.stop()

                    # Extract chunks
                    status_container.write("Extracting file content...")
                    chunks, extract_messages, extract_errors, temp_files = extract_chunks(uploaded_files, st.session_state.debug_enabled)
                    total_chunks = estimate_total_chunks(uploaded_files)
                    completed_chunks = multiprocessing.Value('i', 0)

                    # Process chunks with streaming updates
                    status_container.write(f"Processing {total_chunks} chunks...")
                    file_stats = {}
                    all_messages = extract_messages[:]
                    all_errors = extract_errors[:]
                    status_messages = []
                    max_workers = min(multiprocessing.cpu_count() * 2, 16)  # Dynamic for I/O-bound tasks
                    debug_mode = st.session_state.debug_enabled

                    # Initialize progress bar
                    progress_bar = status_container.progress(0.0, text="Processing chunks...")

                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(
                                process_chunk, st.session_state.app, file_name, chunk_text, data_type, debug_mode, completed_chunks, total_chunks
                            )
                            for file_name, chunk_text, data_type in chunks
                        ]
                        processed_chunks = 0
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                file_name, stat, status_message = future.result()
                                if file_name not in file_stats:
                                    file_stats[file_name] = []
                                file_stats[file_name].append(stat)
                                status_messages.append(status_message)
                                processed_chunks += 1
                                # Update progress bar in main thread
                                progress_bar.progress(
                                    min(processed_chunks / total_chunks, 1.0),
                                    text=f"Processed {processed_chunks}/{total_chunks} chunks"
                                )
                            except Exception as e:
                                all_errors.append(f"Thread execution error: {str(e)}")
                                logger.error(f"Thread execution error: {e}", exc_info=True)
                                with completed_chunks.get_lock():
                                    completed_chunks.value += 1
                                processed_chunks += 1
                                status_messages.append(f"Processed chunk {completed_chunks.value}/{total_chunks}: Error ({str(e)})")
                                # Update progress bar on error
                                progress_bar.progress(
                                    min(processed_chunks / total_chunks, 1.0),
                                    text=f"Processed {processed_chunks}/{total_chunks} chunks"
                                )

                    # Display status messages in the main thread
                    for msg in status_messages:
                        status_container.write(msg)

                    # Generate file-level summaries
                    for file_name, stats in file_stats.items():
                        total_chunks_file = len(stats)
                        successful_chunks = sum(1 for stat in stats if stat["success"])
                        skipped_chunks = sum(1 for stat in stats if stat["status"] == "Skipped (Duplicate)")
                        sum_duration = sum(stat["duration"] for stat in stats)
                        avg_duration = sum_duration / total_chunks_file if total_chunks_file > 0 else 0
                        embedding_dimension = next(
                            (stat["embedding_dimension"] for stat in stats if stat["success"]), 0
                        )
                        error_messages = [stat["error"] for stat in stats if not stat["success"]]
                        status = "Success" if successful_chunks + skipped_chunks == total_chunks_file else f"Failed: {', '.join(error_messages)}"
                        all_messages.append(
                            f"Processed {file_name}: {successful_chunks}/{total_chunks_file} chunks successful, "
                            f"{skipped_chunks} skipped (duplicates), "
                            f"Embedding Dimension: {embedding_dimension}, "
                            f"Total Duration: {sum_duration:.2f}s, "
                            f"Avg Duration: {avg_duration:.2f}s, "
                            f"Status: {status}"
                        )
                        if debug_mode:
                            logger.debug(f"File summary for {file_name}: {status}, {successful_chunks}/{total_chunks_file} chunks")

                    # Finalize status
                    progress_bar.progress(1.0, text="Processing complete")
                    status_container.update(label="File processing complete", state="complete")
                    for msg in all_messages:
                        st.write(msg)

                # Display errors outside the status container
                if all_errors:
                    st.subheader("Processing Errors")
                    for file_name in set(file_name for file_name, _, _ in chunks):
                        file_errors = [err for err in all_errors if file_name in err]
                        if file_errors:
                            st.write(f"**{file_name}**:")
                            for err in file_errors:
                                st.error(err)
                else:
                    st.success("✅ All files processed successfully")

            st.divider()
            st.subheader("📄 File Preview")
            if uploaded_files:
                file_names = [f.name for f in uploaded_files]
                selected_file = st.selectbox("Select file to preview", file_names)
                selected_file_obj = next(f for f in uploaded_files if f.name == selected_file)
                display_file(selected_file_obj)
            else:
                st.write("No files uploaded.")

        st.divider()
        st.checkbox(
            "🐞 Enable Debug Mode",
            value=st.session_state.debug_enabled,
            key="debug_checkbox",
            on_change=lambda: st.session_state.update(debug_enabled=st.session_state.debug_checkbox)
        )
        st.checkbox(
            "🔬 Use Streaming Mode (Disables RAG)",
            value=st.session_state.use_streaming,
            key="streaming_checkbox",
            on_change=lambda: (
                st.session_state.update(use_streaming=st.session_state.streaming_checkbox),
                save_config()
            )
        )

        st.divider()
        with st.container():
            col1, col2 = st.columns([2, 1], gap="small")
            with col1:
                st.button(
                    "🔄 Reset Session State",
                    disabled=not st.session_state.confirm_reset_session,
                    on_click=lambda: (
                        st.session_state.update({k: v for k, v in session_defaults.items()}),
                        os.remove("chat_history.json") if os.path.exists("chat_history.json") else None,
                        os.remove("config.json") if os.path.exists("config.json") else None,
                        logger.info("Session state reset"),
                        st.rerun()
                    )
                )
            with col2:
                st.checkbox(
                    "Confirm Reset",
                    value=False,
                    key="confirm_reset_session"
                )

        st.divider()
        if st.button("🔒 Logout"):
            st.session_state.is_authenticated = False
            st.session_state.debug_enabled = False
            st.session_state.use_streaming = False
            st.session_state.last_uploaded_image = None
            st.session_state.show_login_panel = True
            st.session_state.debug_sessions = []
            save_config()  # Save use_streaming=False
            logger.info("User logged out")
            st.rerun()

    elif st.session_state.show_login_panel:
        st.subheader("🔐 Admin Login")
        with st.form(key="login_form"):
            username = st.text_input("Username", help="Enter your username")
            password = st.text_input("Password", type="password", help="Enter your password")
            submit_button = st.form_submit_button(label="Login")
            if submit_button:
                if not username or not password:
                    st.error("Username and password cannot be empty")
                elif username != "admin":
                    st.error("Invalid username")
                    logger.warning(f"Invalid username entered: {username}")
                else:
                    if authenticate(username, password):
                        st.session_state.is_authenticated = True
                        st.session_state.debug_enabled = False
                        st.session_state.show_login_panel = False
                        st.success("Login successful!")
                        logger.info("Login successful")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")

# --- GUI: Main Content ---
st.title("RadioSport Chat")
st.caption(f"Version {APP_VERSION}")

# Safe message rendering for user messages
def safe_message(content, is_user=False, key=None):
    try:
        if isinstance(content, types.GeneratorType):
            if st.session_state.debug_enabled:
                logger.warning("Generator object detected in message content, converting to string")
            try:
                content = "".join(content)
            except Exception as e:
                content = f"Error consuming generator: {str(e)}"
                logger.error(f"Failed to consume generator in message: {str(e)}")
        elif not isinstance(content, str):
            if st.session_state.debug_enabled:
                logger.warning(f"Non-string content type {type(content)} in message, converting to string")
            content = str(content)
        message(content, is_user=is_user, key=key)
    except Exception as e:
        logger.error(f"Error rendering message: {str(e)}")
        message(f"Error rendering message: {str(e)}", is_user=is_user, key=key)

# Function to format content with LaTeX
def format_latex_content(content):
    """Detect and wrap LaTeX expressions for proper rendering."""
    # Simple regex to detect common LaTeX patterns (e.g., $...$, \(...\), [...], or \commands)
    latex_pattern = r'(\$.*?\$|\[.*?]|\(.*?\)|\\[\w{}]+)'
    parts = []
    last_end = 0
    
    for match in re.finditer(latex_pattern, content):
        start, end = match.span()
        # Add text before the LaTeX
        if start > last_end:
            parts.append(content[last_end:start])
        # Handle the LaTeX expression
        latex = match.group(0)
        if latex.startswith('$') and latex.endswith('$'):
            parts.append(latex)  # Already formatted inline
        elif latex.startswith('[') and latex.endswith(']'):
            # Convert [equation] to $$equation$$
            equation = latex[1:-1].strip()
            parts.append(f"$${equation}$$")
        elif latex.startswith('(') and latex.endswith(')'):
            # Convert (expression) to \(expression\)
            expression = latex[1:-1].strip()
            parts.append(f"\\({expression}\\)")
        else:
            # Wrap standalone LaTeX commands in $
            parts.append(f"${latex}$")
        last_end = end
    
    # Add remaining text
    if last_end < len(content):
        parts.append(content[last_end:])
    
    return ''.join(parts)

# Render chat history
for i, msg in enumerate(st.session_state.messages):
    if msg["role"] == "user":
        safe_message(msg["content"], is_user=True, key=str(i))
    else:
        # Assistant messages with bold "Assistant:" and LaTeX formatting
        content = msg["content"]
        if isinstance(content, types.GeneratorType):
            if st.session_state.debug_enabled:
                logger.warning("Generator object detected in assistant message content, converting to string")
            try:
                content = "".join(content)
            except Exception as e:
                content = f"Error consuming generator: {str(e)}"
                logger.error(f"Failed to consume generator in assistant message: {str(e)}")
        elif not isinstance(content, str):
            if st.session_state.debug_enabled:
                logger.warning(f"Non-string content type {type(content)} in assistant message, converting to string")
            content = str(content)
        formatted_content = format_latex_content(content)
        st.markdown(f"**Assistant:** {formatted_content}", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1])

with col1:
    st.checkbox(
        "Confirm: Clear chat history",
        value=False,
        key="confirm_clear_chat"
    )
    if st.button("🧹 Clear Chat History", disabled=not st.session_state.confirm_clear_chat):
        st.session_state.messages = []
        st.session_state.debug_sessions = []
        st.session_state.streaming_session_id = 0  # Reset session ID
        st.session_state.streaming_active = False  # Reset streaming lock
        if os.path.exists("chat_history.json"):
            os.remove("chat_history.json")
        logger.info("Chat history cleared")
        st.rerun()

prompt = st.chat_input("Ask about your files or images...")

if prompt:
    # Validate messages is a list
    if not isinstance(st.session_state.messages, list):
        st.session_state.messages = []
        logger.warning("st.session_state.messages was not a list, reinitialized as empty list")
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    safe_message(prompt, is_user=True, key=str(len(st.session_state.messages) - 1))
    save_chat_history()

    with st.spinner("🔍 Analyzing..."):
        try:
            if not hasattr(st.session_state, 'app'):
                st.error("st.session_state.app is not initialized. Please restart the app.")
                st.session_state.messages.append({"role": "assistant", "content": "st.session_state.app is not initialized. Please restart the app."})
                save_chat_history()
                st.rerun()
            # Ensure streaming_active is initialized
            if 'streaming_active' not in st.session_state:
                st.session_state.streaming_active = False
                logger.warning("Initialized missing st.session_state.streaming_active to False")
            if st.session_state.streaming_active:
                st.session_state.messages.append({"role": "assistant", "content": "A streaming session is already active. Please wait for it to complete."})
                save_chat_history()
                if st.session_state.debug_enabled:
                    logger.debug("Appended message: Streaming session already active")
                st.rerun()
            else:
                if st.session_state.debug_enabled:
                    logger.debug(f"Processing chat prompt: '{prompt[:50]}...'")
                st.session_state.streaming_active = True
                if st.session_state.use_streaming:
                    # Streaming mode: Use Ollama raw streaming with history
                    st.session_state.streaming_session_id += 1
                    session_id = st.session_state.streaming_session_id
                    response_chunks = ollama_raw_stream(prompt, debug_mode=st.session_state.debug_enabled)
                    text = ""
                    progress_bar = st.progress(0)
                    chunk_count = 0
                    total_chunks_estimated = 10  # Estimate for progress bar
                    placeholder = st.empty()  # Placeholder for streaming text
                    try:
                        for chunk in st.write_stream(response_chunks):
                            chunk_count += 1
                            text += chunk
                            formatted_text = format_latex_content(text)
                            placeholder.markdown(f"**Assistant:** {formatted_text}", unsafe_allow_html=True)
                            if st.session_state.debug_enabled:
                                logger.debug(
                                    f"Streamed chunk {chunk_count} at {time.time():.2f}s: "
                                    f"'{chunk[:50]}...' (len={len(chunk)}, total_len={len(text)}, session_id={session_id})"
                                )
                            progress_bar.progress(min(chunk_count / total_chunks_estimated, 1.0))
                        # Append final response
                        st.session_state.messages.append({"role": "assistant", "content": text})
                        save_chat_history()
                        progress_bar.progress(1.0)
                        st.session_state.streaming_active = False
                        if st.session_state.debug_enabled:
                            logger.debug(f"Streaming completed, final response: '{text[:50]}...'")
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Rendering error (streaming): {str(e)}")
                        formatted_error = format_latex_content(f"Sorry, the requested information is unavailable at this time")
                        placeholder.markdown(f"**Assistant:** {formatted_error}", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": "Sorry, the requested information is unavailable at this time"})
                        save_chat_history()
                        progress_bar.progress(1.0)
                        st.session_state.streaming_active = False
                        st.rerun()
                else:
                    # Non-streaming: Use embedchain's chat for RAG
                    response_chunks = st.session_state.app.chat(
                        prompt, debug_mode=st.session_state.debug_enabled
                    )
                    if st.session_state.debug_enabled:
                        logger.debug(f"app.chat response type: {type(response_chunks).__name__}")
                    text = ""
                    progress_bar = st.progress(0)
                    chunk_count = 0
                    total_chunks_estimated = 10  # Estimate for progress bar
                    placeholder = st.empty()  # Placeholder for streaming text
                    st.session_state.streaming_session_id += 1
                    session_id = st.session_state.streaming_session_id
                    try:
                        for chunk in st.write_stream(response_chunks):
                            chunk_count += 1
                            text += chunk
                            formatted_text = format_latex_content(text)
                            placeholder.markdown(f"**Assistant:** {formatted_text}", unsafe_allow_html=True)
                            if st.session_state.debug_enabled:
                                logger.debug(
                                    f"Streamed chunk {chunk_count} at {time.time():.2f}s: "
                                    f"'{chunk[:50]}...' (len={len(chunk)}, total_len={len(text)}, session_id={session_id})"
                                )
                            progress_bar.progress(min(chunk_count / total_chunks_estimated, 1.0))
                        # Append final response
                        st.session_state.messages.append({"role": "assistant", "content": text})
                        save_chat_history()
                        progress_bar.progress(1.0)
                        st.session_state.streaming_active = False
                        if st.session_state.debug_enabled:
                            logger.debug(f"Streaming completed, final response: '{text[:50]}...'")
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Rendering error (streaming): {str(e)}")
                        formatted_error = format_latex_content(f"Sorry, the requested information is unavailable at this time")
                        placeholder.markdown(f"**Assistant:** {formatted_error}", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": "Sorry, the requested information is unavailable at this time"})
                        save_chat_history()
                        progress_bar.progress(1.0)
                        st.session_state.streaming_active = False
                        st.rerun()
        except Exception as e:
            st.error(f"Chat error: {str(e)}")
            logger.error(f"Chat error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": "Sorry, the requested information is unavailable at this time"})
            save_chat_history()
            st.session_state.streaming_active = False
            st.rerun()

# --- Debug Information ---
def get_debug_info():
    if hasattr(st.session_state, 'app'):
        return st.session_state.app.get_debug_info()
    return {}

if st.session_state.debug_enabled and st.session_state.is_authenticated:
    st.divider()
    st.subheader("🔍 RAG Pipeline Debug Information")

    debug_info = get_debug_info()

    debug_tab1, debug_tab2, debug_tab3 = st.tabs(
        ["Current Query", "Add Operations", "DB Stats"]
    )

    with debug_tab1:
        st.write("**Recent Queries:**")
        if st.session_state.debug_sessions:
            for i, session in enumerate(reversed(st.session_state.debug_sessions[-5:])):
                st.write(f"**Query {len(st.session_state.debug_sessions) - i}:**")
                st.write(f"**Prompt:** {session.get('prompt', 'No prompt available')}")
                st.write(f"**Response:** {session.get('response', 'No response available')}")
                st.write("**Streaming Info:**")
                streaming_info = session.get('streaming_info', {})
                st.write(f"- Response Type: {streaming_info.get('response_type', 'Unknown')}")
                st.write(f"- Total Chunks: {streaming_info.get('chunk_count', 0)}")
                if streaming_info.get('chunks_received'):
                    st.write("- Chunks Received:")
                    for chunk_info in streaming_info['chunks_received'][:5]:  # Limit to 5 for brevity
                        st.write(
                            f"  - Chunk {chunk_info['chunk_number']}: '{chunk_info['content']}' "
                            f"(len={chunk_info['length']}, time={chunk_info['timestamp']:.2f}s)"
                        )
                st.write("**Retrieved Documents:** Not available due to Embedchain API limitations.")
                st.write("---")
        else:
            st.write("No queries recorded yet.")

    with debug_tab2:
        st.write("**File Embedding Reports:**")
        if debug_info.get("add_operations", []):
            for op in debug_info["add_operations"]:
                st.write(
                    f"- [{op['timestamp']}] File: {op.get('file_name', 'Unknown')} (Type: {op['data_type']}), "
                    f"Status: {op.get('status', 'Unknown')}, "
                    f"Chunks: {op.get('successful_chunks', 1 if op.get('success', False) else 0)}/{op.get('total_chunks', 1)}, "
                    f"Embedding Dimension: {op['embedding_dimension']}, "
                    f"Total Duration: {op.get('total_duration', op.get('duration', 0)):.2f}s, "
                    f"Avg Duration: {op.get('avg_duration', op.get('duration', 0)):.2f}s, "
                    f"Snippet: {op['text_snippet']}"
                )
        else:
            st.write("No file embedding reports recorded.")

    with debug_tab3:
        st.write("**Vector DB Stats:**")
        try:
            if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, "db"):
                if st.session_state.debug_enabled:
                    logger.debug("Accessing DB stats")
                count = st.session_state.app.db.count()
                st.write(f"📈 Documents in DB: {count}")
                if st.session_state.debug_enabled:
                    logger.debug(f"Current DB document count: {count}")
        except Exception as e:
            st.error(f"Error accessing vector DB: {str(e)}")
            logger.error(f"Vector DB access error: {e}")

# --- RAG Component Testing ---
if st.session_state.is_authenticated:
    with st.sidebar.expander("🧪 RAG Component Testing", expanded=False):
        st.write("Test individual RAG components for troubleshooting.")
        test_text = st.text_area("Enter text to test embedding:", value="The whole Duty of man. Fear God, keep his commands.")
        if st.button("Test Embedding", key="test_embed_btn"):
            with st.spinner("Testing embedding..."):
                try:
                    if not hasattr(st.session_state, 'app'):
                        st.session_state.messages.append({"role": "assistant", "content": "st.session_state.app is not initialized. Please restart the app."})
                        save_chat_history()
                        if st.session_state.debug_enabled:
                            logger.debug("Appended message: st.session_state.app is not initialized")
                        st.rerun()
                    else:
                        result = st.session_state.app.test_embedding(test_text)
                        if result["success"]:
                            output = (
                                f"✅ Embedding successful - Dimension: {result['embedding_dimension']}\n"
                                f"Sample of embedding vector: {result['embedding_sample']}\n"
                                f"Process took {result['duration']:.4f} seconds"
                            )
                            st.session_state.messages.append({"role": "assistant", "content": output})
                            save_chat_history()
                            if st.session_state.debug_enabled:
                                logger.debug(f"Appended message: {output[:50]}...")
                        else:
                            st.session_state.messages.append({"role": "assistant", "content": f"❌ Embedding failed: {result['error']}"})
                            save_chat_history()
                            if st.session_state.debug_enabled:
                                logger.debug(f"Appended message: Embedding failed: {result['error'][:50]}...")
                        st.rerun()
                except Exception as e:
                    st.session_state.messages.append({"role": "assistant", "content": f"Test failed with error: {str(e)}"})
                    save_chat_history()
                    if st.session_state.debug_enabled:
                        logger.debug(f"Appended message: Test failed: {str(e)[:50]}...")
                    st.rerun()