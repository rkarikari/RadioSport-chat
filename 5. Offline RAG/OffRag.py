import os
import shutil
import tempfile
import base64
import re
import gc

import streamlit as st
from embedchain import App
from streamlit_chat import message
from PIL import Image
import pytesseract
from io import BytesIO

# === IMPORTANT ===
# If Tesseract is NOT in your system PATH, uncomment and update the line below:
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Enforce offline mode by removing unnecessary API keys
os.environ.pop("OPENAI_API_KEY", None)

# Configure Streamlit page settings
st.set_page_config(
    page_title="Multimodal Chat Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

def create_embedchain_app(db_path):
    """
    Creates and returns an Embedchain application instance configured for gemma3:4b vision model.
    """
    return App.from_config(
        config={
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "gemma3:4b",
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

def render_file_preview(uploaded_file):
    """
    Renders a preview of the uploaded file based on its MIME type.
    """
    if not uploaded_file:
        return
    try:
        uploaded_file.seek(0)
        mime_type = uploaded_file.type
        if mime_type == "application/pdf":
            base64_pdf = base64.b64encode(uploaded_file.read()).decode("utf-8")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                'width="100%" height="600px" frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True,
            )
        elif mime_type.startswith("image/"):
            st.image(uploaded_file, use_container_width=True)
        elif mime_type.startswith("audio/"):
            st.audio(uploaded_file, format=mime_type)
        elif mime_type.startswith("video/"):
            st.video(uploaded_file)
        else:
            st.warning(f"Unsupported file type: {mime_type}")
    except Exception as error:
        st.error(f"Error previewing file: {error}")

@st.cache_resource
def initialize_app():
    """
    Initializes the Embedchain application and database directory.
    """
    db_dir = tempfile.mkdtemp()
    return create_embedchain_app(db_dir), db_dir

# Initialize Streamlit session state
if "app" not in st.session_state or "db_dir" not in st.session_state:
    st.session_state.app, st.session_state.db_dir = initialize_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_uploaded_image" not in st.session_state:
    st.session_state.last_uploaded_image = None

# Sidebar for file management
with st.sidebar:
    st.title("🗂️ File Management")
    st.header("Upload Your Files")

    uploaded_file = st.file_uploader(
        "Select files to upload",
        type=["pdf", "png", "jpg", "jpeg", "mp3", "wav", "mp4", "txt"],
        accept_multiple_files=False,
    )

    if uploaded_file:
        # Cache the last uploaded image for multimodal querying
        if uploaded_file.type.startswith("image/"):
            st.session_state.last_uploaded_image = uploaded_file

        if st.button("🚀 Add to Knowledge Base"):
            with st.spinner("Processing..."):
                try:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as temp_file:
                        temp_file.write(uploaded_file.getvalue())
                        file_path = temp_file.name

                    if uploaded_file.type == "application/pdf":
                        st.session_state.app.add(file_path, data_type="pdf_file")
                        st.success(f"✅ Added {uploaded_file.name}")
                    elif uploaded_file.type.startswith("image/"):
                        try:
                            image = Image.open(file_path)
                            extracted_text = pytesseract.image_to_string(image).strip()
                            if not extracted_text:
                                st.error("No text extracted via OCR.")
                            else:
                                st.session_state.app.add(extracted_text, data_type="text")
                                st.success(f"✅ Added text from {uploaded_file.name}")
                        except Exception as ocr_error:
                            st.error(f"OCR Error: {ocr_error}")
                    elif uploaded_file.type.startswith("audio/"):
                        st.session_state.app.add(file_path, data_type="audio_file")
                        st.success(f"✅ Added {uploaded_file.name}")
                    elif uploaded_file.type.startswith("video/"):
                        st.session_state.app.add(file_path, data_type="video_file")
                        st.success(f"✅ Added {uploaded_file.name}")
                    elif uploaded_file.type == "text/plain":
                        st.session_state.app.add(file_path, data_type="text")
                        st.success(f"✅ Added {uploaded_file.name}")
                    else:
                        st.warning(f"Unsupported file type: {uploaded_file.type}")

                    os.remove(file_path)
                except Exception as error:
                    st.error(f"Error adding file: {error}")

        st.divider()
        st.subheader("📄 File Preview")
        render_file_preview(uploaded_file)

st.title("🌐 Multimodal Chat Assistant")
st.caption("Interact with documents, images, audio, and video using gemma3:4b vision model")

# Display chat history
for idx, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(idx))

# Options to clear chat history or flush cache
col1, col2 = st.columns(2)
with col1:
    if st.button("🧹 Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

with col2:
    if st.button("🗑️ Flush Cache"):
        st.session_state.messages = []
        db_dir = st.session_state.pop("db_dir", None)
        st.session_state.pop("app", None)
        gc.collect()

        st.cache_resource.clear()
        st.cache_data.clear()

        if db_dir:
            try:
                shutil.rmtree(db_dir)
            except Exception as flush_error:
                st.error(f"Error cleaning up cache directory: {flush_error}")

        st.session_state.app, st.session_state.db_dir = initialize_app()
        st.success("Cache flushed successfully!")
        st.rerun()

# Handle user inputs
user_prompt = st.chat_input("Ask questions about your uploaded files or images...")
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    message(user_prompt, is_user=True)

    with st.spinner("🔍 Generating Response..."):
        try:
            if st.session_state.last_uploaded_image:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image:
                    temp_image.write(st.session_state.last_uploaded_image.getvalue())
                    image_path = temp_image.name

                assistant_response = st.session_state.app.chat(user_prompt, image=image_path)
                os.remove(image_path)
            else:
                assistant_response = st.session_state.app.chat(user_prompt)

            processed_response = re.sub(r"<think>.*?</think>", "", assistant_response, flags=re.DOTALL)
            st.session_state.messages.append({"role": "assistant", "content": processed_response})
            message(processed_response)
        except Exception as error:
            st.error(f"Error generating response: {error}")
