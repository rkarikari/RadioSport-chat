import os
import streamlit as st
from config import APP_VERSION
from gui import render_gui, safe_message, format_latex_content
from utils import initialize_app, save_chat_history, load_chat_history, save_config, load_config, ollama_raw_stream, logger
from file_processing import estimate_total_chunks, extract_chunks, process_chunk
import concurrent.futures
import multiprocessing
import time
from rag_pipeline import get_app

# Session State Initialization
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
    "streaming_session_id": 0,
    "streaming_active": False,
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

# Main GUI Rendering
render_gui()

# Handle User Input
prompt = st.chat_input("Ask about your files or images...")
if prompt:
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
            if 'streaming_active' not in st.session_state:
                st.session_state.streaming_active = False
                logger.warning("Initialized missing st.session_state.streaming_active to False")
            if st.session_state.streaming_active:
                st.session_state.messages.append({"role": "assistant", "content": "A streaming session is already active. Please wait for it to complete."})
                save_chat_history()
                st.rerun()
            else:
                st.session_state.streaming_active = True
                if st.session_state.use_streaming:
                    st.session_state.streaming_session_id += 1
                    session_id = st.session_state.streaming_session_id
                    response_chunks = ollama_raw_stream(prompt, debug_mode=st.session_state.debug_enabled)
                    text = ""
                    progress_bar = st.progress(0)
                    chunk_count = 0
                    total_chunks_estimated = 10
                    placeholder = st.empty()
                    try:
                        for chunk in st.write_stream(response_chunks):
                            chunk_count += 1
                            text += chunk
                            formatted_text = format_latex_content(text)
                            placeholder.markdown(f"**Assistant:** {formatted_text}", unsafe_allow_html=True)
                            logger.debug(
                                f"Streamed chunk {chunk_count} at {time.time():.2f}s: "
                                f"'{chunk[:50]}...' (len={len(chunk)}, total_len={len(text)}, session_id={session_id})"
                            )
                            progress_bar.progress(min(chunk_count / total_chunks_estimated, 1.0))
                        formatted_text = format_latex_content(text)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_text})
                        save_chat_history()
                        progress_bar.progress(1.0)
                        st.session_state.streaming_active = False
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Rendering error (streaming): {str(e)}")
                        formatted_error = format_latex_content(f"Rendering error: {str(e)}")
                        placeholder.markdown(f"**Assistant:** {formatted_error}", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_error})
                        save_chat_history()
                        progress_bar.progress(1.0)
                        st.session_state.streaming_active = False
                        st.rerun()
                else:
                    response_chunks = st.session_state.app.chat(
                        prompt, debug_mode=st.session_state.debug_enabled
                    )
                    text = ""
                    progress_bar = st.progress(0)
                    chunk_count = 0
                    total_chunks_estimated = 10
                    placeholder = st.empty()
                    st.session_state.streaming_session_id += 1
                    session_id = st.session_state.streaming_session_id
                    try:
                        for chunk in st.write_stream(response_chunks):
                            chunk_count += 1
                            text += chunk
                            formatted_text = format_latex_content(text)
                            placeholder.markdown(f"**Assistant:** {formatted_text}", unsafe_allow_html=True)
                            logger.debug(
                                f"Streamed chunk {chunk_count} at {time.time():.2f}s: "
                                f"'{chunk[:50]}...' (len={len(chunk)}, total_len={len(text)}, session_id={session_id})"
                            )
                            progress_bar.progress(min(chunk_count / total_chunks_estimated, 1.0))
                        formatted_text = format_latex_content(text)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_text})
                        save_chat_history()
                        progress_bar.progress(1.0)
                        st.session_state.streaming_active = False
                        st.rerun()
                    except Exception as e:
                        logger.error(f"Rendering error (streaming): {str(e)}")
                        formatted_error = format_latex_content(f"Rendering error: {str(e)}")
                        placeholder.markdown(f"**Assistant:** {formatted_error}", unsafe_allow_html=True)
                        st.session_state.messages.append({"role": "assistant", "content": formatted_error})
                        save_chat_history()
                        progress_bar.progress(1.0)
                        st.session_state.streaming_active = False
                        st.rerun()
        except Exception as e:
            st.error(f"Chat error: {str(e)}")
            logger.error(f"Chat error: {str(e)}")
            formatted_error = format_latex_content(f"Chat error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": formatted_error})
            save_chat_history()
            st.session_state.streaming_active = False
            st.rerun()