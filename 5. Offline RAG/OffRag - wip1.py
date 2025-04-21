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

    def add(self, *args, **kwargs):
        start_time = time.time()
        data_type = kwargs.get("data_type", "unknown")
        text_snippet = str(args[0])[:100] + "..." if args else "No text"
        logger.info(f"Adding document (type={data_type}): snippet='{text_snippet}'")

        result = super().add(*args, **kwargs)

        operation_info = {
            "timestamp": datetime.now().isoformat(),
            "data_type": data_type,
            "text_snippet": text_snippet,
            "duration": time.time() - start_time,
            "success": result is not None,
        }
        self.debug_info["add_operations"].append(operation_info)
        logger.info(f"Add operation completed: {operation_info}")
        return result

    def chat(self, prompt, **kwargs):
        self.debug_info["current_session"] = {
            "prompt": prompt,
            "response": "",
            "retrieved_docs": [],
        }
        start_time = time.time()
        logger.info(f"Processing chat query: '{prompt[:50]}...'")

        response = super().chat(prompt, **kwargs)

        self.debug_info["current_session"]["response"] = response
        duration = time.time() - start_time
        logger.info(f"Chat response generated in {duration:.2f}s")

        # If you want to capture retrieved docs, check if response or other attributes expose them.
        # Otherwise, leave retrieved_docs empty or implement alternative retrieval logging.

        return response

    def test_embedding(self, text):
        try:
            start_time = time.time()
            embedding = self.embedder.embed_query(text)
            duration = time.time() - start_time
            return {
                "success": True,
                "embedding_dimension": len(embedding),
                "embedding_sample": embedding[:5],
                "duration": duration,
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def get_debug_info(self):
        return self.debug_info


def embedchain_bot(db_path):
    # Use DebugApp instead of App for enhanced debugging
    return DebugApp.from_config(
        config={
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "granite3.3:2b",  # Correct vision model name
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


@st.cache_resource(show_spinner=False)
def get_app():
    db_dir = tempfile.mkdtemp()
    return embedchain_bot(db_dir), db_dir


if "app" not in st.session_state or "db_dir" not in st.session_state:
    st.session_state.app, st.session_state.db_dir = get_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_uploaded_image" not in st.session_state:
    st.session_state.last_uploaded_image = None

# Debug mode toggle
if "debug_mode" not in st.session_state:
    st.session_state.debug_mode = False

with st.sidebar:
    st.title("🗂️ File Management")
    st.header("Upload Your Files")

    uploaded_file = st.file_uploader(
        "Select files",
        type=["pdf", "png", "jpg", "jpeg", "txt"],
        accept_multiple_files=False,
        key="file_uploader",
    )

    if uploaded_file:
        if uploaded_file.type.startswith("image/"):
            st.session_state.last_uploaded_image = uploaded_file

        if st.button("🚀 Add to Knowledge Base", type="primary"):
            with st.spinner("Processing..."):
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=os.path.splitext(uploaded_file.name)[1]
                    ) as f:
                        f.write(uploaded_file.getvalue())
                        file_path = f.name

                    if uploaded_file.type == "application/pdf":
                        data_type = "pdf_file"
                        st.session_state.app.add(file_path, data_type=data_type)
                        st.success(f"✅ Added {uploaded_file.name}")

                    elif uploaded_file.type.startswith("image/"):
                        try:
                            img = Image.open(file_path)
                            ocr_text = pytesseract.image_to_string(img).strip()
                            if not ocr_text:
                                st.error("OCR did not extract any text from the image.")
                            else:
                                st.session_state.app.add(ocr_text, data_type="text")
                                st.success(f"✅ Added {uploaded_file.name} (via OCR text)")
                        except Exception as e:
                            st.error(f"OCR Error: {str(e)}")

                    elif uploaded_file.type.startswith("audio/"):
                        data_type = "audio_file"
                        st.session_state.app.add(file_path, data_type=data_type)
                        st.success(f"✅ Added {uploaded_file.name}")

                    elif uploaded_file.type.startswith("video/"):
                        data_type = "video_file"
                        st.session_state.app.add(file_path, data_type=data_type)
                        st.success(f"✅ Added {uploaded_file.name}")

                    elif uploaded_file.type == "text/plain":
                        data_type = "text"
                        text_content = uploaded_file.read().decode("utf-8")
                        st.session_state.app.add(text_content, data_type=data_type)
                        st.success(f"✅ Added {uploaded_file.name}")

                    else:
                        st.error(f"Unsupported file type: {uploaded_file.type}")

                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                    if "file_path" in locals() and file_path:
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            st.error(f"Error deleting temporary file: {e}")

        st.divider()
        st.subheader("📄 File Preview")
        display_file(uploaded_file)

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
            del st.session_state.app
        if "db_dir" in st.session_state:
            db_dir = st.session_state.db_dir
            del st.session_state.db_dir

        gc.collect()  # Force garbage collection to close file handles

        st.cache_data.clear()
        st.cache_resource.clear()

        try:
            if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, 'db') and hasattr(st.session_state.app.db, 'client'):
                st.session_state.app.db.client.close()
                time.sleep(2)
        except Exception as e:
            st.error(f"Error closing ChromaDB client: {e}")

        try:
            if db_dir:
                shutil.rmtree(db_dir, ignore_errors=True)
        except Exception as e:
            st.error(f"Error deleting vector DB directory: {e}")

        st.session_state.app, st.session_state.db_dir = get_app()
        st.success("RAG cache and chat history have been flushed.")
        st.rerun()

prompt = st.chat_input("Ask about your files or images...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    with st.spinner("🔍 Analyzing..."):
        try:
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

# --- Display Debug Information if Debug Mode is Enabled ---
if st.session_state.debug_mode:
    st.divider()
    st.subheader("🔍 RAG Pipeline Debug Information")

    debug_info = st.session_state.app.get_debug_info()

    debug_tab1, debug_tab2, debug_tab3 = st.tabs(
        ["Current Query", "Add Operations", "DB Stats"]
    )

    with debug_tab1:
        current_session = debug_info["current_session"]
        st.write("**Last Query Prompt:**")
        st.write(current_session["prompt"])

        st.write("**Retrieved Documents:**")
        st.write("Not available due to Embedchain API limitations.")

        st.write("**Response:**")
        st.write(current_session["response"])

    with debug_tab2:
        st.write("**Document Add Operations:**")
        if debug_info["add_operations"]:
            for op in debug_info["add_operations"]:
                st.write(
                    f"- [{op['timestamp']}] Type: {op['data_type']}, "
                    f"Success: {op['success']}, Duration: {op['duration']:.2f}s, "
                    f"Snippet: {op['text_snippet']}"
                )
        else:
            st.write("No add operations recorded.")

    with debug_tab3:
        st.write("**Vector DB Stats:**")
        try:
            if hasattr(st.session_state.app, "db"):
                try:
                    count = st.session_state.app.db.count()
                    st.write(f"📈 Documents in DB: {count}")
                    logger.info(f"Current DB document count: {count}")
                except Exception as e:
                    st.warning(f"Could not retrieve DB stats: {str(e)}")
                    logger.warning(f"Error getting DB stats: {str(e)}")
            else:
                st.write("No DB instance found.")
        except Exception as e:
            st.error(f"Error accessing vector DB: {str(e)}")
            logger.error(f"Error accessing vector DB: {str(e)}")


# --- Optional: Add a sidebar section for component testing ---
with st.sidebar.expander("🧪 RAG Component Testing", expanded=False):
    st.write("Test individual RAG components for troubleshooting.")

    test_text = st.text_area("Enter text to test embedding:", value="This is a test of the embedding model.")
    if st.button("Test Embedding", key="test_embed_btn"):
        with st.spinner("Testing embedding..."):
            try:
                result = st.session_state.app.test_embedding(test_text)
                if result["success"]:
                    st.success(f"✅ Embedding successful - Dimension: {result['embedding_dimension']}")
                    st.write("Sample of embedding vector:", result["embedding_sample"])
                    st.write(f"Process took {result['duration']:.4f} seconds")
                else:
                    st.error(f"❌ Embedding failed: {result['error']}")
            except Exception as e:
                st.error(f"Test failed with error: {str(e)}")
