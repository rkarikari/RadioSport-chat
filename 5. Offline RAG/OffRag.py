import os
import shutil
import tempfile
import base64
import re
import gc
import time
import logging
from datetime import datetime

import streamlit as st
from embedchain import App
from streamlit_chat import message
from PIL import Image
import pytesseract
from io import BytesIO
import numpy as np
import concurrent.futures
import pdfplumber

# === IMPORTANT ===
# If Tesseract is NOT in your system PATH, uncomment and update the line below:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

os.environ.pop("OPENAI_API_KEY", None)  # Enforce offline mode

st.set_page_config(
    page_title="Multimodal Chat Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Setup logging for debugging and troubleshooting
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("rag_assistant")

# --- Custom Debuggable App Class to track RAG pipeline operations ---
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

    @property
    def embedder(self):
        """Access the embedder, initializing it if needed."""
        if hasattr(self, '_embedder'):
            return self._embedder

        try:
            if hasattr(self, 'db') and hasattr(self.db, 'embedder'):
                self._embedder = self.db.embedder
                logger.info("Using embedder from self.db.embedder")
            elif hasattr(self, 'config') and isinstance(self.config, dict) and 'embedder' in self.config:
                from embedchain.embedder.ollama import OllamaEmbedder
                embedder_config = self.config['embedder'].get('config', {})
                self._embedder = OllamaEmbedder(**embedder_config)
                logger.info("Created embedder from config")
            else:
                from embedchain.embedder.ollama import OllamaEmbedder
                self._embedder = OllamaEmbedder(
                    model="nomic-embed-text:latest",
                    base_url="http://localhost:11434"
                )
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
        logger.info(f"Adding document (type={data_type}): snippet='{text_snippet}'")

        try:
            if not args or not args[0].strip():
                raise ValueError("Empty text provided to embed")
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
            }
            self.debug_info["add_operations"].append(operation_info)
            raise
        else:
            operation_info = {
                "timestamp": datetime.now().isoformat(),
                "data_type": data_type,
                "text_snippet": text_snippet,
                "duration": time.time() - start_time,
                "success": True,
            }
            self.debug_info["add_operations"].append(operation_info)
            logger.info(f"Add operation completed: {operation_info}")
            return result

    def chat(self, prompt, **kwargs):
        self.debug_info["current_session"] = {
            "prompt": prompt,
            "response": "",
            "retrieved_docs": [],
            "contexts": [],
            "prompt": "",
            "response": "",
        }
        start_time = time.time()
        logger.info(f"Processing chat query: '{prompt[:50]}...'")

        response = super().chat(prompt, **kwargs)

        self.debug_info["current_session"]["response"] = response
        duration = time.time() - start_time
        logger.info(f"Chat response generated in {duration:.2f}s")

        return response

    def test_embedding(self, text):
        try:
            start_time = time.time()
            embeddings = self.embedder.to_embeddings([text])

            if isinstance(embeddings, list):
                embedding = embeddings[0]
            elif isinstance(embeddings, np.ndarray):
                embedding = embeddings
            else:
                raise TypeError(f"Unexpected embedding type: {type(embeddings)}")

            if isinstance(embedding, np.float32):
                raise TypeError("Embedding is a single float value, not a sequence.")

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
        logger.info(f"Creating app with db_path: {db_path}")
        app = embedchain_bot(db_path=db_path)
        return app
    except Exception as e:
        logger.error(f"Failed to create app in get_app: {str(e)}")
        raise

# Initialize app and db_dir at the start
def initialize_app():
    try:
        logger.info("Attempting to initialize app")
        # Check if app is already initialized and valid
        if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, 'db'):
            try:
                # Test DB access to ensure it's valid
                st.session_state.app.db.count()
                logger.info("App already initialized and valid, skipping initialization")
                return
            except Exception as e:
                logger.warning(f"Existing app is invalid: {str(e)}, reinitializing")

        # Use a fixed DB path to persist data
        db_path = "./chroma_db"
        os.makedirs(db_path, exist_ok=True)
        st.session_state.db_dir = db_path
        st.session_state.app = get_app(db_path)
        logger.info("Successfully initialized st.session_state.app and st.session_state.db_dir")
    except Exception as e:
        st.error(f"Failed to initialize app: {str(e)}")
        logger.error(f"App initialization error: {str(e)}")
        raise

# Run initialization immediately
initialize_app()

# Initialize session state defaults
if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_uploaded_image" not in st.session_state:
    st.session_state.last_uploaded_image = None

if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

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
                # Ensure app is initialized
                if not hasattr(st.session_state, 'app'):
                    try:
                        logger.warning("st.session_state.app missing before file processing, reinitializing")
                        initialize_app()
                        logger.info("Reinitialized st.session_state.app for file processing")
                    except Exception as e:
                        st.error(f"Failed to reinitialize app for file processing: {str(e)}")
                        logger.error(f"App reinitialization error: {str(e)}")
                        st.stop()

                def add_file(file, app):
                    messages = []  # Collect messages for main thread
                    errors = []  # Collect errors for main thread
                    try:
                        with tempfile.NamedTemporaryFile(
                            delete=False, suffix=os.path.splitext(file.name)[1]
                        ) as f:
                            f.write(file.getvalue())
                            file_path = f.name

                        if file.type == "application/pdf":
                            with pdfplumber.open(file_path) as pdf:
                                pages_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
                            if not pages_text:
                                errors.append(f"No text extracted from {file.name}")
                                logger.warning(f"No text extracted from PDF: {file.name}")
                                return messages, errors
                            messages.append(f"Extracted text from {len(pages_text)} pages in {file.name}")
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                futures = [
                                    executor.submit(app.add, text, data_type="text")
                                    for text in pages_text if text.strip()
                                ]
                                for future in concurrent.futures.as_completed(futures):
                                    try:
                                        future.result()
                                    except Exception as e:
                                        errors.append(f"Error adding PDF page: {e}")
                                        logger.error(f"ThreadPoolExecutor error (PDF): {e}")
                        elif file.type.startswith("image/"):
                            img = Image.open(file_path)
                            ocr_text = pytesseract.image_to_string(img).strip()
                            if not ocr_text:
                                errors.append(f"No text extracted from {file.name}")
                                logger.warning(f"No OCR text extracted from image: {file.name}")
                                return messages, errors
                            messages.append(f"Extracted OCR text from {file.name}")
                            app.add(ocr_text, data_type="text")
                        elif file.type == "text/plain":
                            text_content = file.read().decode("utf-8")
                            if not text_content.strip():
                                errors.append(f"Empty text file: {file.name}")
                                logger.warning(f"Empty text file: {file.name}")
                                return messages, errors
                            chunk_size = 1000
                            chunks = [text_content[i:i + chunk_size] for i in range(0, len(text_content), chunk_size)]
                            messages.append(f"Split {file.name} into {len(chunks)} chunks")
                            with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                                futures = [
                                    executor.submit(app.add, chunk, data_type="text")
                                    for chunk in chunks if chunk.strip()
                                ]
                                for future in concurrent.futures.as_completed(futures):
                                    try:
                                        future.result()
                                    except Exception as e:
                                        errors.append(f"Error adding text chunk: {e}")
                                        logger.error(f"ThreadPoolExecutor error (text): {e}")
                        else:
                            errors.append(f"Unsupported file type: {file.type}")
                            logger.warning(f"Unsupported file type: {file.type}")
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
                    return messages, errors

                # Process files and collect messages/errors
                all_messages = []
                all_errors = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
                    futures = [executor.submit(add_file, file, st.session_state.app) for file in uploaded_files]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            messages, errors = future.result()
                            all_messages.extend(messages)
                            all_errors.extend(errors)
                        except Exception as e:
                            all_errors.append(f"Thread execution error: {str(e)}")
                            logger.error(f"Thread execution error: {e}")

                # Display messages and errors in the main thread
                for msg in all_messages:
                    st.write(msg)
                for err in all_errors:
                    st.error(err)

                if not all_errors:
                    st.success("✅ All files processed successfully")
                else:
                    st.warning("Some files processed with errors. Check logs for details.")

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
        st.rerun()

with col2:
    if st.button("🗑️ Flush RAG Cache"):
        st.session_state.messages = []
        db_dir = None
        if "app" in st.session_state:
            try:
                if hasattr(st.session_state.app, 'db') and hasattr(st.session_state.app.db, 'client'):
                    st.session_state.app.db.client.close()
                    time.sleep(2)
            except Exception as e:
                st.error(f"Error closing ChromaDB client: {e}")
            del st.session_state.app
        if "db_dir" in st.session_state:
            db_dir = st.session_state.db_dir
            del st.session_state.db_dir

        gc.collect()
        st.cache_data.clear()
        st.cache_resource.clear()

        try:
            if db_dir:
                shutil.rmtree(db_dir, ignore_errors=True)
                logger.info(f"Deleted DB directory: {db_dir}")
        except Exception as e:
            st.error(f"Error deleting vector DB directory: {e}")

        # Reinitialize app and db_dir
        try:
            initialize_app()
            st.success("RAG cache and chat history have been flushed.")
            logger.info("Reinitialized st.session_state.app and st.session_state.db_dir after cache flush")
        except Exception as e:
            st.error(f"Failed to reinitialize app: {str(e)}")
            logger.error(f"App reinitialization error: {str(e)}")
            st.stop()
        st.rerun()

prompt = st.chat_input("Ask about your files or images...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    with st.spinner("🔍 Analyzing..."):
        try:
            if not hasattr(st.session_state, 'app'):
                st.error("st.session_state.app is not initialized. Please restart the app.")
                logger.error("st.session_state.app is not initialized for chat")
            else:
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
        except Exception as e:
            st.error(f"Response error: {str(e)}")

@st.cache_data(show_spinner=False)
def get_cached_debug_info():
    if hasattr(st.session_state, 'app'):
        return st.session_state.app.get_debug_info()
    return {}

if st.session_state.debug_mode:
    st.divider()
    st.subheader("🔍 RAG Pipeline Debug Information")

    debug_info = get_cached_debug_info()

    debug_tab1, debug_tab2, debug_tab3 = st.tabs(
        ["Current Query", "Add Operations", "DB Stats"]
    )

    with debug_tab1:
        current_session = debug_info.get("current_session", {})
        st.write("**Last Query Prompt:**")
        st.write(current_session.get("prompt", "No prompt available"))

        st.write("**Retrieved Documents:**")
        st.write("Not available due to Embedchain API limitations.")

        st.write("**Response:**")
        st.write(current_session.get("response", "No response available"))

    with debug_tab2:
        st.write("**Document Add Operations:**")
        if debug_info.get("add_operations", []):
            for op in debug_info["add_operations"]:
                status = "Success" if op["success"] else f"Failed: {op.get('error', 'Unknown error')}"
                st.write(
                    f"- [{op['timestamp']}] Type: {op['data_type']}, "
                    f"Status: {status}, Duration: {op['duration']:.2f}s, "
                    f"Snippet: {op['text_snippet']}"
                )
        else:
            st.write("No add operations recorded.")

    with debug_tab3:
        st.write("**Vector DB Stats:**")
        try:
            if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, "db"):
                logger.info("Accessing DB stats")
                try:
                    count = st.session_state.app.db.count()
                    st.write(f"📈 Documents in DB: {count}")
                    logger.info(f"Current DB document count: {count}")
                except Exception as e:
                    st.warning(f"Could not retrieve DB stats: {str(e)}")
                    logger.error(f"DB stats error: {str(e)}")
            else:
                st.write("No DB instance found.")
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
                    logger.error("st.session_state.app is not initialized for test embedding")
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