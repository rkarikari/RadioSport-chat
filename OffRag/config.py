import streamlit as st
# Version tracking
APP_VERSION = "v2.0.0.5"

# Environment settings
import os
os.environ.pop("OPENAI_API_KEY", None)  # Enforce offline mode


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

