import os
import tempfile
import streamlit as st
from embedchain import App
import base64
from streamlit_chat import message
import re
import requests

# Set OpenAI API key to an empty string to disable it
os.environ["OPENAI_API_KEY"] = ""

# Page config
st.set_page_config(
    page_title="Multimodal Chat Assistant",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Define embedchain app creation with Ollama only (offline mode)
def embedchain_bot(db_path):
    return App.from_config(
        config={
            "llm": {
                "provider": "ollama",
                "config": {
                    "model": "granite3.3:2b",
                    "max_tokens": 1000,
                    "temperature": 0.3,
                    "stream": True,
                    "base_url": "http://localhost:11434"
                }
            },
            "vectordb": {
                "provider": "chroma",
                "config": {"dir": db_path}
            },
            "embedder": {
                "provider": "ollama",
                "config": {
                    "model": "nomic-embed-text:latest",
                    "base_url": "http://localhost:11434",
                }
            }
        }
    )

# Test the connection to the Ollama server
def test_ollama_connection():
    try:
        # Send a request to the base URL to confirm it's running
        response = requests.get("http://localhost:11434")
        if response.status_code == 200 and "Ollama is running" in response.text:
            st.sidebar.success("Ollama server is online! 🟢")
        else:
            st.sidebar.error(f"Ollama server is offline. Status code: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"Connection Error: {e}")

# Check connection on app start
test_ollama_connection()

def display_file(file):
    try:
        file.seek(0)
        mime_type = file.type
        if mime_type == "application/pdf":
            base64_pdf = base64.b64encode(file.read()).decode("utf-8")
            st.markdown(
                f'<iframe src="data:application/pdf;base64,{base64_pdf}" '
                'width="100%" height="600px" frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )
        elif mime_type.startswith("image/"):
            st.image(file, use_container_width=True)
        elif mime_type.startswith("audio/"):
            st.audio(file, format=mime_type)
        elif mime_type.startswith("video/"):
            st.video(file)
    except Exception as e:
        st.error(f"Preview error: {str(e)}")

# Cache app instance once per session
@st.cache_resource
def get_app():
    db_dir = tempfile.mkdtemp()
    return embedchain_bot(db_dir)

# Initialize app and messages in session state
if "app" not in st.session_state:
    st.session_state.app = get_app()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for file upload and adding to knowledge base
with st.sidebar:
    st.title("🗂️ File Management")
    st.header("Upload Your Files")

    uploaded_file = st.file_uploader(
        "Select files",
        type=["pdf", "png", "jpg", "jpeg", "mp3", "wav", "mp4", "txt"],
        accept_multiple_files=False,
        key="file_uploader"
    )

    if uploaded_file:
        if st.button("🚀 Add to Knowledge Base", type="primary"):
            with st.spinner("Processing..."):
                try:
                    # Save uploaded file to a temp file
                    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as f:
                        f.write(uploaded_file.getvalue())
                        # Explicitly define valid data_type for embedchain add()
                        if uploaded_file.type == "application/pdf":
                            data_type = "pdf"  # Correct Embedchain-supported data type for PDFs
                        elif uploaded_file.type.startswith("image/"):
                            data_type = "image"
                        elif uploaded_file.type.startswith("audio/"):
                            data_type = "audio"
                        elif uploaded_file.type.startswith("video/"):
                            data_type = "video"
                        elif uploaded_file.type == "text/plain":
                            data_type = "text"
                        else:
                            data_type = None

                        if data_type is None:
                            st.error(f"Unsupported file type: {uploaded_file.type}")
                        else:
                            st.session_state.app.add(f.name, data_type=data_type)
                            st.success(f"✅ Added {uploaded_file.name}")
                    os.remove(f.name)
                except Exception as e:
                    st.error(f"Error: {str(e)}")

        st.divider()
        st.subheader("📄 File Preview")
        display_file(uploaded_file)

# Main chat interface
st.title("🌐 Multimodal Chat Assistant")
st.caption("Chat with documents, images, audio, and video using Gemma3:12b")

# Display chat messages
for i, msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user", key=str(i))

if prompt := st.chat_input("Ask about your files..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)

    with st.spinner("🔍 Analyzing..."):
        try:
            response = st.session_state.app.chat(prompt)
            # Remove any <think>...</think> tags from response
            filtered_response = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
            st.session_state.messages.append({"role": "assistant", "content": filtered_response})
            message(filtered_response)
        except Exception as e:
            st.error(f"Response error: {str(e)}")

if st.button("🧹 Clear Chat History"):
    st.session_state.messages = []
    st.experimental_rerun()
