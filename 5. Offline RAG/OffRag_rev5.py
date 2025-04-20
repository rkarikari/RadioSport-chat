import os
import shutil
import tempfile
import base64
import re
import gc
import time

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

def embedchain_bot(db_path):
    return App.from_config(
        config={
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "gemma3:4b",  # Correct vision model name
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
    except Exception as e:
        st.error(f"Preview error: {str(e)}")

@st.cache_resource
def get_app():
    db_dir = tempfile.mkdtemp()
    return embedchain_bot(db_dir), db_dir

if "app" not in st.session_state or "db_dir" not in st.session_state:
    st.session_state.app, st.session_state.db_dir = get_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Store last uploaded image for multimodal querying
if "last_uploaded_image" not in st.session_state:
    st.session_state.last_uploaded_image = None

with st.sidebar:
    st.title("🗂️ File Management")
    st.header("Upload Your Files")

    uploaded_file = st.file_uploader(
        "Select files",
        type=["pdf", "png", "jpg", "jpeg", "mp3", "wav", "mp4", "txt"],
        accept_multiple_files=False,
        key="file_uploader",
    )

    if uploaded_file:
        # Save last uploaded image for vision queries
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
                        st.session_state.app.add(file_path, data_type=data_type)
                        st.success(f"✅ Added {uploaded_file.name}")


                    else:
                        st.error(f"Unsupported file type: {uploaded_file.type}")


                except Exception as e:
                    st.error(f"Error: {str(e)}")
                finally:
                     if 'file_path' in locals() and file_path:
                        try:
                            os.remove(file_path)
                        except Exception as e:
                            st.error(f"Error deleting temporary file: {e}")

        st.divider()
        st.subheader("📄 File Preview")
        display_file(uploaded_file)

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

        # Attempt to close ChromaDB client and wait before deleting the directory
        try:
            if hasattr(st.session_state, 'app') and hasattr(st.session_state.app, 'db') and hasattr(st.session_state.app.db, 'client'):
                st.session_state.app.db.client.close()
                time.sleep(2)  # Wait for 2 seconds
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
            # If an image was uploaded, pass it along for multimodal query
            if st.session_state.last_uploaded_image is not None:
                img_file = st.session_state.last_uploaded_image
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(img_file.name)[1]) as tmp_img:
                    tmp_img.write(img_file.getvalue())
                    img_path = tmp_img.name

                # Pass image path with prompt to the vision model
                response = st.session_state.app.chat(
                    prompt,
                    image=img_path  # Adjust this param if Embedchain API differs
                )
                os.remove(img_path)
            else:
                response = st.session_state.app.chat(prompt)

            filtered_response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
            st.session_state.messages.append({"role": "assistant", "content": filtered_response})
            message(filtered_response)
        except Exception as e:
            st.error(f"Response error: {str(e)}")
