import os
import shutil
import tempfile
import base64
import re
import gc
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

# Uncomment and update if Tesseract is not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.environ.pop("OPENAI_API_KEY", None)  # Enforce offline mode

st.set_page_config(
    page_title="Multimodal Chat Assistant",
    page_icon="🌐",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_assistant")

logger.info(f"Using chromadb version: {chromadb.__version__}")
logger.info(f"Using embedchain version: {embedchain.__version__}")

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
        self._embedding_dimension = None
        # Compute embedding dimension once
        try:
            test_text = "test"
            embeddings = self.embedder.to_embeddings([test_text])
            embedding = embeddings[0] if isinstance(embeddings, list) else embeddings
            self._embedding_dimension = len(embedding)
        except Exception as e:
            logger.error(f"Failed to compute embedding dimension: {str(e)}")
            self._embedding_dimension = 0

    @property
    def embedder(self):
        if hasattr(self, '_embedder'):
            return self._embedder
        try:
            if hasattr(self, 'db') and hasattr(self.db, 'embedder'):
                self._embedder = self.db.embedder
                if st.session_state.debug_mode:
                    logger.info("Using embedder from self.db.embedder")
            elif hasattr(self, 'config') and 'embedder' in self.config:
                from embedchain.embedder.ollama import OllamaEmbedder
                embedder_config = self.config['embedder'].get('config', {})
                self._embedder = OllamaEmbedder(**embedder_config)
                if st.session_state.debug_mode:
                    logger.info("Created embedder from config")
            else:
                from embedchain.embedder.ollama import OllamaEmbedder
                self._embedder = OllamaEmbedder(
                    model="nomic-embed-text:latest",
                    base_url="http://localhost:11434"
                )
                if st.session_state.debug_mode:
                    logger.info("Created default embedder")
            return self._embedder
        except Exception as e:
            logger.error(f"Failed to get/create embedder: {str(e)}")
            raise

    @embedder.setter
    def embedder(self, value):
        self._embedder = value

    def add(self, *args, **kwargs):
        start_time = time.time()
        data_type = kwargs.get("data_type", "unknown")
        text_snippet = str(args[0])[:100] + "..." if args and args[0] else "No text"
        if st.session_state.debug_mode:
            logger.info(f"Adding document (type={data_type}): snippet='{text_snippet}'")

        try:
            if not args or not args[0].strip():
                raise ValueError("Empty text provided to embed")
            embedding_dimension = self._embedding_dimension if st.session_state.debug_mode else 0
            result = super().add(args[0], **kwargs)
            if result is None:
                raise ValueError("Embedding result is None")
        except Exception as e:
            logger.error(f"Error in add method: {e}")
            operation_info = {
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "text_snippet": text_snippet,
                "duration": time.time() - start_time,
                "success": False,
                "error": str(e),
                "embedding_dimension": 0,
            }
            return operation_info
        else:
            operation_info = {
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "text_snippet": text_snippet,
                "duration": time.time() - start_time,
                "success": True,
                "embedding_dimension": embedding_dimension,
            }
            self.debug_info["add_operations"].append(operation_info)
            if len(self.debug_info["add_operations"]) > 100:
                self.debug_info["add_operations"] = self.debug_info["add_operations"][-100:]
            return operation_info

    def chat(self, prompt, **kwargs):
        try:
            if st.session_state.debug_mode:
                logger.info(f"Chat called with prompt: '{prompt[:50]}...'")
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
            if st.session_state.debug_mode:
                logger.info(f"Chat response generated in {duration:.2f}s")
            st.session_state.debug_sessions.append(self.debug_info["current_session"])
            if len(st.session_state.debug_sessions) > 100:
                st.session_state.debug_sessions = st.session_state.debug_sessions[-100:]
            return response
        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            self.debug_info["current_session"]["response"] = f"Error: {str(e)}"
            st.session_state.debug_sessions.append(self.debug_info["current_session"])
            raise

    def force_reset_db(self):
        try:
            if hasattr(self, 'db') and hasattr(self.db, 'client'):
                if st.session_state.debug_mode:
                    logger.info("Force resetting ChromaDB database")
                client = self.db.client
                collections = client.list_collections()
                for collection in collections:
                    if st.session_state.debug_mode:
                        logger.info(f"Deleting collection: {collection.name}")
                    client.delete_collection(collection.name)
                for attempt in range(3):
                    try:
                        if hasattr(client, '_system') and client._system is not None:
                            if st.session_state.debug_mode:
                                logger.info(f"Stopping ChromaDB client system (attempt {attempt + 1})")
                            client._system.stop()
                            time.sleep(1)
                        break
                    except Exception as e:
                        logger.warning(f"Failed to stop client system (attempt {attempt + 1}): {str(e)}")
                try:
                    if hasattr(client, '_system') and hasattr(client._system, 'persistence'):
                        persistence = client._system.persistence
                        if hasattr(persistence, 'db'):
                            if st.session_state.debug_mode:
                                logger.info("Closing SQLite connections")
                            persistence.db.close()
                            time.sleep(1)
                except Exception as e:
                    logger.warning(f"Failed to close SQLite connections: {str(e)}")
                self.db = None
                gc.collect()
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
                raise TypeError("Embedding is a single float value")
            self.add(text, data_type="text")
            duration = time.time() - start_time
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

def embedchain_bot(db_path):
    try:
        if st.session_state.debug_mode:
            logger.info(f"Creating DebugApp with db_path: {db_path}")
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
        if st.session_state.debug_mode:
            logger.info("DebugApp created successfully")
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

def get_app(db_path):
    try:
        if st.session_state.debug_mode:
            logger.info(f"Creating app with db_path: {db_path}")
        app = embedchain_bot(db_path=db_path)
        return app
    except Exception as e:
        logger.error(f"Failed to create app in get_app: {str(e)}")
        raise

@retry(stop=stop_after_attempt(3), wait=wait_fixed(20))
def safe_rmtree(path, attempt=1):
    try:
        if st.session_state.debug_mode:
            logger.info(f"Attempt {attempt} to delete directory: {path}")
        gc.collect()
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        os.remove(file_path)
                        if st.session_state.debug_mode:
                            logger.info(f"Deleted file: {file_path}")
                    except Exception as e:
                        logger.error(f"Failed to delete file {file_path}: {str(e)}")
                        raise
            shutil.rmtree(path)
            if st.session_state.debug_mode:
                logger.info(f"Successfully deleted directory: {path}")
        else:
            if st.session_state.debug_mode:
                logger.info(f"Directory does not exist: {path}")
    except Exception as e:
        logger.error(f"Error deleting directory {path} on attempt {attempt}: {str(e)}")
        raise

def initialize_app():
    try:
        if st.session_state.debug_mode:
            logger.info("Attempting to initialize app")
        db_path = "./chroma_db"
        os.makedirs(db_path, exist_ok=True)
        st.session_state.db_dir = db_path

        if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, 'db'):
            try:
                count = st.session_state.app.db.count()
                if st.session_state.debug_mode:
                    logger.info(f"App already initialized and valid, document count: {count}")
                return
            except Exception as e:
                logger.warning(f"Existing app is invalid: {str(e)}, reinitializing")

        try:
            st.session_state.app = get_app(db_path)
        except Exception as e:
            logger.warning(f"Detected DB initialization error: {str(e)}, attempting reset")
            if hasattr(st.session_state, 'app'):
                try:
                    st.session_state.app.force_reset_db()
                    if st.session_state.debug_mode:
                        logger.info("Force reset DB before clearing directory")
                except Exception as e:
                    logger.warning(f"Failed to force reset DB: {str(e)}")
            try:
                safe_rmtree(db_path)
                os.makedirs(db_path, exist_ok=True)
                st.session_state.app = get_app(db_path)
                if st.session_state.debug_mode:
                    logger.info("Successfully reinitialized app after clearing database")
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

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_uploaded_image" not in st.session_state:
    st.session_state.last_uploaded_image = None
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False
if "debug_sessions" not in st.session_state:
    st.session_state.debug_sessions = []

with st.sidebar:
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

                def add_file(file, app):
                    messages = []
                    errors = []
                    chunk_stats = []
                    first_snippet = None
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=os.path.splitext(file.name)[1]
                        ) as f:
                            f.write(file.getvalue())
                            file_path = f.name

                        if file.type == "application/pdf":
                            try:
                                with pdfplumber.open(file_path) as pdf:
                                    pages_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
                            except Exception as e:
                                errors.append(f"Error extracting text from PDF {file.name}: {str(e)}")
                                logger.error(f"PDF text extraction error: {e}")
                                return messages, errors, None
                            if not pages_text:
                                errors.append(f"No text extracted from {file.name}")
                                logger.warning(f"No text extracted from PDF: {file.name}")
                                return messages, errors, None
                            messages.append(f"Extracted text from {len(pages_text)} pages in {file.name}")
                            start_time = time.time()
                            max_workers = min(10, multiprocessing.cpu_count())
                            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = [
                                    executor.submit(app.add, text, data_type="text")
                                    for text in pages_text if text.strip()
                                ]
                                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                                    try:
                                        stat = future.result()
                                        chunk_stats.append(stat)
                                        if i == 0:
                                            first_snippet = stat["text_snippet"]
                                    except Exception as e:
                                        errors.append(f"Error adding PDF page: {e}")
                                        logger.error(f"ThreadPoolExecutor error (PDF): {e}")
                            end_time = time.time()
                        elif file.type.startswith("image/"):
                            img = Image.open(file_path)
                            max_size = 1024
                            if img.width > max_size or img.height > max_size:
                                img.thumbnail((max_size, max_size))
                                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as f:
                                    img.save(f.name)
                                    file_path = f.name
                            ocr_text = pytesseract.image_to_string(img).strip()
                            if not ocr_text:
                                errors.append(f"No text extracted from {file.name}")
                                logger.warning(f"No OCR text extracted from image: {file.name}")
                                return messages, errors, None
                            messages.append(f"Extracted OCR text from {file.name}")
                            start_time = time.time()
                            stat = app.add(ocr_text, data_type="text")
                            end_time = time.time()
                            chunk_stats.append(stat)
                            first_snippet = stat["text_snippet"]
                        elif file.type == "text/plain":
                            try:
                                text_content = file.read().decode("utf-8")
                            except UnicodeDecodeError:
                                text_content = file.read().decode("latin1", errors="ignore")
                            if not text_content.strip():
                                errors.append(f"Empty text file: {file.name}")
                                logger.warning(f"Empty text file: {file.name}")
                                return messages, errors, None
                            chunk_size = 2000
                            chunks = [text_content[i:i + chunk_size] for i in range(0, len(text_content), chunk_size)]
                            messages.append(f"Split {file.name} into {len(chunks)} chunks")
                            start_time = time.time()
                            max_workers = min(10, multiprocessing.cpu_count())
                            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                                futures = [
                                    executor.submit(app.add, chunk, data_type="text")
                                    for chunk in chunks if chunk.strip()
                                ]
                                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                                    try:
                                        stat = future.result()
                                        chunk_stats.append(stat)
                                        if i == 0:
                                            first_snippet = stat["text_snippet"]
                                    except Exception as e:
                                        errors.append(f"Error adding text chunk: {e}")
                                        logger.error(f"ThreadPoolExecutor error (text): {e}")
                            end_time = time.time()
                        else:
                            errors.append(f"Unsupported file type: {file.type}")
                            logger.warning(f"Unsupported file type: {file.type}")
                            return messages, errors, None

                        total_chunks = len(chunk_stats)
                        successful_chunks = sum(1 for stat in chunk_stats if stat["success"])
                        sum_duration = sum(stat["duration"] for stat in chunk_stats)
                        avg_duration = sum_duration / total_chunks if total_chunks > 0 else 0
                        total_duration = end_time - start_time
                        embedding_dimension = next(
                            (stat["embedding_dimension"] for stat in chunk_stats if stat["success"]), 0
                        )
                        error_messages = [stat["error"] for stat in chunk_stats if not stat["success"]]
                        status = "Success" if successful_chunks == total_chunks else f"Failed: {', '.join(error_messages)}"

                        operation_info = {
                            "timestamp": datetime.now().isoformat(),
                            "file_name": file.name,
                            "data_type": file.type,
                            "text_snippet": first_snippet or "No text",
                            "total_chunks": total_chunks,
                            "successful_chunks": successful_chunks,
                            "status": status,
                            "total_duration": total_duration,
                            "avg_duration": avg_duration,
                            "embedding_dimension": embedding_dimension,
                        }
                        if st.session_state.debug_mode:
                            logger.info(f"File embedding report: {operation_info}")

                    except Exception as e:
                        errors.append(f"Error processing {file.name}: {str(e)}")
                        logger.error(f"File processing error: {e}")
                    finally:
                        if 'file_path' in locals():
                            try:
                                os.remove(file_path)
                            except Exception as e:
                                errors.append(f"Error deleting temporary file: {e}")
                                logger.error(f"Temp file deletion error: {e}")
                    return messages, errors, operation_info

                all_messages = []
                all_errors = []
                max_workers = min(10, multiprocessing.cpu_count())
                with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                    futures = [executor.submit(add_file, file, st.session_state.app) for file in uploaded_files]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            messages, errors, _ = future.result()
                            all_messages.extend(messages)
                            all_errors.extend(errors)
                        except Exception as e:
                            all_errors.append(f"Thread execution error: {str(e)}")
                            logger.error(f"Thread execution error: {e}")

                for msg in all_messages:
                    st.write(msg)
                for err in all_errors:
                    st.error(err)

                if not all_errors:
                    st.success("✅ All files processed successfully")
                else:
                    st.warning("Some files processed with errors.")

        st.divider()
        st.subheader("📄 File Preview")
        display_file(uploaded_files[0] if uploaded_files else None)

    st.divider()
    st.checkbox(
        "🐞 Enable Debug Mode", value=st.session_state.debug_mode, key="debug_mode"
    )

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
                if st.session_state.debug_mode:
                    logger.info(f"Processing chat prompt: '{prompt[:50]}...'")
                if st.session_state.last_uploaded_image is not None:
                    img_file = st.session_state.last_uploaded_image
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(img_file.name)[1]
                    ) as tmp_img:
                        tmp_img.write(img_file.getvalue())
                        img_path = tmp_img.name
                    response = st.session_state.app.chat(prompt, image=img_path)
                    os.remove(img_path)
                else:
                    response = st.session_state.app.chat(prompt)
                filtered_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
                st.session_state.messages.append({"role": "assistant", "content": filtered_response})
                message(filtered_response)
                if st.session_state.debug_mode:
                    logger.info("Chat response processed and displayed")
        except Exception as e:
            st.error(f"Response error: {str(e)}")
            logger.error(f"Chat processing error: {str(e)}")

def get_debug_info():
    if hasattr(st.session_state, 'app'):
        return st.session_state.app.get_debug_info()
    return {}

if st.session_state.debug_mode:
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
                    f"- [{op['timestamp']}] File: {op['file_name']} (Type: {op['data_type']}), "
                    f"Status: {op['status']}, "
                    f"Chunks: {op['successful_chunks']}/{op['total_chunks']}, "
                    f"Embedding Dimension: {op['embedding_dimension']}, "
                    f"Total Duration: {op['total_duration']:.2f}s, "
                    f"Avg Duration: {op['avg_duration']:.2f}s, "
                    f"Snippet: {op['text_snippet']}"
                )
        else:
            st.write("No file embedding reports recorded.")

    with debug_tab3:
        st.write("**Vector DB Stats:**")
        try:
            if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, "db"):
                if st.session_state.debug_mode:
                    logger.info("Accessing DB stats")
                count = st.session_state.app.db.count()
                st.write(f"📈 Documents in DB: {count}")
                if st.session_state.debug_mode:
                    logger.info(f"Current DB document count: {count}")
        except Exception as e:
            st.error(f"Error accessing vector DB: {str(e)}")
            logger.error(f"Vector DB access error: {str(e)}")

with st.sidebar.expander("🧪 RAG Component Testing", expanded=False):
    st.write("Test individual RAG components for troubleshooting.")
    test_text = st.text_area("Enter text to test embedding:", value="The whole Duty of man..\r Fear God, keep his commands.")
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