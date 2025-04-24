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
from tenacity import retry, stop_after_attempt, wait_fixed
import streamlit as st
from embedchain import App
from streamlit_chat import message
from PIL import Image
import pytesseract
from io import BytesIO
import numpy as np
import concurrent.futures
import pdfplumber
import multiprocessing
import bcrypt

# Uncomment and update if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.environ.pop("OPENAI_API_KEY", None)  # Enforce offline mode

# --- GUI Configuration ---
st.set_page_config(
    page_title="Multimodal Chat Assistant",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_assistant")

logger.info(f"Using chromadb version: {chromadb.__version__}")
logger.info(f"Using embedchain version: {embedchain.__version__}")

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

    def add(self, *args, debug_mode=False, file_name="Unknown", **kwargs):
        start_time = time.time()
        data_type = kwargs.get("data_type", "unknown")
        text_snippet = str(args[0])[:100] + "..." if args and args[0] else "No text"
        if debug_mode:
            logger.debug(f"Adding document (type={data_type}, file_name={file_name}): snippet='{text_snippet}'")

        try:
            if not args or not args[0].strip():
                raise ValueError("Empty text provided to embed")
            embedding_dimension = self._compute_embedding_dimension() if debug_mode else 0
            result = super().add(args[0], **kwargs)
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
            }
            start_time = time.time()
            response = super().chat(prompt, **kwargs)
            self.debug_info["current_session"]["response"] = response
            duration = time.time() - start_time
            if debug_mode:
                logger.debug(f"Chat response generated in {duration:.2f}s")
            st.session_state.debug_sessions.append(self.debug_info["current_session"])
            if len(st.session_state.debug_sessions) > 100:
                st.session_state.debug_sessions = st.session_state.debug_sessions[-100:]
            return response
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            self.debug_info["current_session"]["response"] = f"Error: {str(e)}"
            st.session_state.debug_sessions.append(self.debug_info["current_session"])
            raise

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
                        "max_tokens": 1000,
                        "temperature": 0.3,
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

def initialize_app():
    try:
        if st.session_state.debug_enabled:
            logger.debug("Attempting to initialize app")
        db_path = "./chroma_db"
        os.makedirs(db_path, exist_ok=True)
        st.session_state.db_dir = db_path

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
        return file_name, stat
    except Exception as e:
        if debug_mode:
            logger.error(f"Error processing chunk for {file_name}: {str(e)}")
        with completed_chunks.get_lock():
            completed_chunks.value += 1
        if debug_mode:
            logger.debug(f"Incremented completed_chunks to {completed_chunks.value} for failed chunk in {file_name}")
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
        }

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
                with st.spinner("Processing files..."):
                    if not hasattr(st.session_state, 'app'):
                        try:
                            logger.warning("st.session_state.app missing, reinitializing")
                            initialize_app()
                        except Exception as e:
                            st.error(f"Failed to reinitialize app: {str(e)}")
                            st.stop()

                    # Extract chunks and initialize progress
                    chunks, extract_messages, extract_errors, temp_files = extract_chunks(uploaded_files, st.session_state.debug_enabled)
                    total_chunks = estimate_total_chunks(uploaded_files)
                    completed_chunks = multiprocessing.Value('i', 0)
                    progress_bar = st.progress(0.0, text="Starting file processing...")

                    # Process chunks
                    file_stats = {}
                    all_messages = extract_messages[:]
                    all_errors = extract_errors[:]
                    max_workers = min(multiprocessing.cpu_count() * 2, 16)  # Dynamic for I/O-bound tasks
                    debug_mode = st.session_state.debug_enabled
                    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                        futures = [
                            executor.submit(
                                process_chunk, st.session_state.app, file_name, chunk_text, data_type, debug_mode, completed_chunks, total_chunks
                            )
                            for file_name, chunk_text, data_type in chunks
                        ]
                        for future in concurrent.futures.as_completed(futures):
                            try:
                                file_name, stat = future.result()
                                if file_name not in file_stats:
                                    file_stats[file_name] = []
                                file_stats[file_name].append(stat)
                                progress = min(completed_chunks.value / total_chunks, 1.0)
                                progress_text = f"Processing {file_name} (chunk {completed_chunks.value}/{total_chunks})"
                                progress_bar.progress(progress, text=progress_text)
                                if debug_mode:
                                    logger.debug(f"Progress: {progress_text}, {progress*100:.1f}%")
                            except Exception as e:
                                all_errors.append(f"Thread execution error: {str(e)}")
                                logger.error(f"Thread execution error: {e}", exc_info=True)
                                with completed_chunks.get_lock():
                                    completed_chunks.value += 1
                                progress = min(completed_chunks.value / total_chunks, 1.0)
                                progress_text = f"Processing files (chunk {completed_chunks.value}/{total_chunks})"
                                progress_bar.progress(progress, text=progress_text)
                                if debug_mode:
                                    logger.debug(f"Progress (thread error): {progress_text}, {progress*100:.1f}%")

                    # Generate file-level summaries
                    for file_name, stats in file_stats.items():
                        total_chunks_file = len(stats)
                        successful_chunks = sum(1 for stat in stats if stat["success"])
                        sum_duration = sum(stat["duration"] for stat in stats)
                        avg_duration = sum_duration / total_chunks_file if total_chunks_file > 0 else 0
                        embedding_dimension = next(
                            (stat["embedding_dimension"] for stat in stats if stat["success"]), 0
                        )
                        error_messages = [stat["error"] for stat in stats if not stat["success"]]
                        status = "Success" if successful_chunks == total_chunks_file else f"Failed: {', '.join(error_messages)}"
                        all_messages.append(
                            f"Processed {file_name}: {successful_chunks}/{total_chunks_file} chunks successful, "
                            f"Embedding Dimension: {embedding_dimension}, "
                            f"Total Duration: {sum_duration:.2f}s, "
                            f"Avg Duration: {avg_duration:.2f}s, "
                            f"Status: {status}"
                        )
                        if debug_mode:
                            logger.debug(f"File summary for {file_name}: {status}, {successful_chunks}/{total_chunks_file} chunks")

                    # Ensure progress bar completion
                    progress_bar.progress(1.0, text="File processing complete")
                    if debug_mode:
                        logger.debug(f"Processing complete, total chunks: {total_chunks}")

                    for msg in all_messages:
                        st.write(msg)
                    if all_errors:
                        with st.expander("View Processing Errors"):
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

        st.divider()
        if st.button("🔒 Logout"):
            st.session_state.is_authenticated = False
            st.session_state.debug_enabled = False
            st.session_state.last_uploaded_image = None
            st.session_state.show_login_panel = True
            st.session_state.debug_sessions = []
            logger.info("User logged out")
            st.rerun()

    elif st.session_state.show_login_panel:
        st.subheader("🔐 Admin Login")
        # Modified: Removed st.info with login credentials
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
st.title("🌐 Multimodal Chat Assistant")
st.caption("Chat with documents, images, audio, and video using gemma3:4b vision model")

for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(i))

col1, col2 = st.columns([1, 1])

with col1:
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.debug_sessions = []
        st.rerun()

prompt = st.chat_input("Ask about your files or images...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    with st.spinner("🔍 Analyzing..."):
        try:
            if not hasattr(st.session_state, 'app'):
                st.error("st.session_state.app is not initialized. Please restart the app.")
            else:
                if st.session_state.debug_enabled:
                    logger.debug(f"Processing chat prompt: '{prompt[:50]}...'")
                if st.session_state.last_uploaded_image is not None:
                    img_file = st.session_state.last_uploaded_image
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(img_file.name)[1]
                    ) as tmp_img:
                        tmp_img.write(img_file.getvalue())
                        img_path = tmp_img.name
                    response = st.session_state.app.chat(
                        prompt, debug_mode=st.session_state.debug_enabled, image=img_path
                    )
                    os.remove(img_path)
                else:
                    response = st.session_state.app.chat(
                        prompt, debug_mode=st.session_state.debug_enabled
                    )
                filtered_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
                st.session_state.messages.append({"role": "assistant", "content": filtered_response})
                message(filtered_response)
                if st.session_state.debug_enabled:
                    logger.debug("Chat response processed and displayed")
        except Exception as e:
            st.error(f"Chat response error: {str(e)}")
            logger.error(f"Chat processing error: {e}")

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
                        st.error("st.session_state.app is not initialized. Please restart the app.")
                    else:
                        result = st.session_state.app.test_embedding(test_text)
                        if result["success"]:
                            st.success(f"✅ Embedding successful - Dimension: {result['embedding_dimension']}")
                            st.write("Sample of embedding vector:", result['embedding_sample'])
                            st.write(f"Process took {result['duration']:.4f} seconds")
                        else:
                            st.error(f"❌ Embedding failed: {result['error']}")
                except Exception as e:
                    st.error(f"Test failed with error: {str(e)}")