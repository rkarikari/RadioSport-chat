import streamlit as st
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pdfplumber
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from io import BytesIO, StringIO
import base64
import streamlit.components.v1 as components
import subprocess
import re
import hashlib
import logging
import warnings
import sys
import os

# Instructions for setting up the local virtual environment:
# 1. Ensure a Python installation is available (e.g., portable Python in ./python or system Python).
# 2. Create a virtual environment in src/radiosportchat/venv:
#    .\python\python.exe -m venv src\radiosportchat\venv  # Windows
#    ./python/bin/python3 -m venv src/radiosportchat/venv  # Linux/macOS
# 3. Activate the venv:
#    src\radiosportchat\venv\Scripts\activate  # Windows
#    source src/radiosportchat/venv/bin/activate  # Linux/macOS
# 4. Install dependencies:
#    pip install -r requirements.txt
# 5. Install Ollama CLI from https://ollama.com/ and ensure it's running (ollama serve).
# 6. Run the app from the project root:
#    src\radiosportchat\venv\Scripts\python -m streamlit run src\radiosportchat\rag.py  # Windows
#    src/radiosportchat/venv/bin/python -m streamlit run src/radiosportchat/rag.py     # Linux/macOS

# Set page configuration
st.set_page_config(
    page_title="RadioSport Chat",
    page_icon="🧟",
    layout="centered",
    menu_items={
        'Report a Bug': "https://github.com/rkarikari/RadioSport-chat",
        'About': "Copyright © RNK, 2025 RadioSport. All rights reserved."
    }
)

# Suppress pdfplumber logs and warnings
logging.getLogger("pdfplumber").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="pdfplumber")

# Custom stderr filter to suppress CropBox warnings
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        self.buffer = StringIO()

    def write(self, text):
        if "CropBox missing from /Page, defaulting to MediaBox" not in text:
            self.original_stderr.write(text)

    def flush(self):
        self.original_stderr.flush()

# Version and Changelog
VERSION = "v2.5.4"
CHANGELOG = """
Changelog:
- v2.5.4 (2025-05-15): Modified model fetching to use global 'ollama list' command, added error handling for missing Ollama CLI, and improved venv path validation.
- v2.5.3 (2025-05-15): Updated venv path to src/radiosportchat/venv, adjusted Python executable paths in subprocess calls.
- v2.5.2 (2025-05-15): Fixed SyntaxError in 3D Scatter section due to unterminated string literal in colors input.
- v2.5.1 (2025-05-14): Updated to use local virtual environment (venv) for all Python dependencies, eliminating system-wide Python requirements.
- v2.5.0 (2025-05-14): Added st.set_page_config with custom page title, zombie icon, centered layout, and menu items for bug reporting and about info.
- v2.4.14 (2025-05-14): Added custom CSS to reduce top padding, shifting the title "RadioSport Chat" higher on the screen.
- v2.4.13 (2025-05-14): Removed all unnecessary logging, including pdfplumber logging level print and PDF processing debug prints, to streamline console output and improve performance.
- v2.4.12 (2025-05-14): Replaced deprecated `run` with `invoke` for RAG mode to address LangChain deprecation warning, moved pdfplumber logging to function to run once, added file deduplication to prevent repeated PDF processing.
- v2.4.11 (2025-05-14): Added stderr filtering to suppress CropBox warnings during PDF processing, added debug logging for PDF page count.
- v2.4.10 (2025-05-14): Added warnings.filterwarnings to suppress pdfplumber UserWarning for CropBox issue, in addition to logging suppression.
- v2.4.9 (2025-05-14): Added debug print to verify pdfplumber logging level and commented alternative warnings suppression for CropBox warning.
- v2.4.8 (2025-05-14): Suppressed pdfplumber warning "Cropbox missing from /Page, defaulting to Mediabox" by setting logging level to ERROR.
- v2.4.7 (2025-05-14): Fixed warning "Please upload a CSV file" by displaying it only when CSV File data source is selected and no file is uploaded. Corrected typo in Stacked Bar warning message ("honour" to "a").
- v2.4.6 (2025-05-14): Optimized for 200% performance boost:
  - Replaced PyPDF2 with pdfplumber for faster PDF text extraction.
  - Cached qa_chain creation in RAG mode using st.cache_resource with file hashes.
  - Cached list of available models using st.cache_data.
- v2.4.5 (2025-05-13): Implemented periodic scrolling for the floating reasoning window using setInterval to ensure it scrolls to the bottom during streaming.
- v2.4.4 (2025-05-13): Added setTimeout to the scrolling script in the floating reasoning window to ensure it scrolls to the bottom after content updates.
- v2.4.3 (2025-05-13): Enabled automatic scrolling for the floating reasoning window to keep the last line visible. Added backdrop blur to the floating window for better text visibility.
- v2.4.2 (2025-05-12): Added support for Bar, Column, Stacked Bar, Pie Chart, Histogram, Line Chart, Area Chart, 3D Scatter, 3D Line, Waterfall Chart, Radar Chart, Box Plot, Violin Plot, and Error Bar graph types, with CSV support where applicable.
- v2.4.1 (2025-05-12): Added CSV file upload support for Scatter graph type, allowing users to select columns for X, Y, sizes, and colors.
- v2.4.0 (2025-05-12): Added support for Polar, Scatter, 3D Surface, and Contour graph types in the Graphing Controls section.
- v2.3.11 (2025-05-11): Prevented session state updates in RAG mode and added UI note to clarify fixed models (nomic-embed-text:latest and granite3.3:2b).
- v2.3.10 (2025-05-11): Fixed IndentationError in stream_response function by removing erroneous '- \' line.
- v2.3.9 (2025-05-09): Updated RAG mode to display nomic-embed-text:latest and granite3.3:2b in disabled dropdowns to accurately reflect the models used.
- v2.3.8 (2025-05-09): Disabled model selection dropdowns in RAG mode to prevent user confusion.
- v2.3.7 (2025-05-09): Restricted RAG mode to use nomic-embed-text:latest for embedding model and granite3.3:2b for LLM model.
- v2.3.6 (2025-05-09): Updated default LaTeX expression to \\frac{\\sum_{i=1}^{77}i}{\\sum_{i=1}^7i}.
- v2.3.5 (2025-05-09): Updated default x parametric equation to sin(t)*(np.exp(cos(t))-2*cos(4*t)-sin(t/12)**5), y parametric equation to cos(t)*(np.exp(cos(t))-2*cos(4*t)-sin(t/12)**5), default embedder to nomic-embed-text:latest, and default language model to granite3.3:2b.
- v2.3.4 (2025-05-09): Reduced floating window maximum width to 400px.
- v2.3.3 (2025-05-09): Reduced floating window minimum width to 400px.
- v2.3.2 (2025-05-09): Fixed SyntaxError in stream_response function by removing erroneous 'GRID' from cleaned_response assignment.
- v2.3.1 (2025-05-09): Updated floating window CSS to auto-scroll to last line, transparent background, blue text, minimum width 500px, maximum height 300px.
- v2.3.0 (2025-05-09): Implemented separate floating chat window for <think>...</think> text, cleared after </think>, with non-reasoning text streaming to main chat window.
- v2.2.5 (2025-05-09): Fixed SyntaxError, but floating window was not a separate chat window and text routing was incorrect.
- v2.2.4 (2025-05-09): Attempted to fix floating window and streaming, but introduced SyntaxError with 'yield' outside function.
- v2.2.3 (2025-05-09): Fixed floating window to be single-line, horizontal, but window did not show and streaming was broken.
- v2.2.2 (2025-05-09): Attempted <think>...</think> filtering with floating window (incorrectly displayed as column).
- v2.2.1 (2025-05-09): Added version tracking and version number display under title.
- v2.2.0 (2025-05-09): Base version with model selection, LaTeX in expander, separate LLM/embedding models, and chat history context.
"""

# Function to compute file hash
def compute_file_hash(file):
    file.seek(0)
    hash_obj = hashlib.md5()
    while chunk := file.read(8192):
        hash_obj.update(chunk)
    file.seek(0)
    return hash_obj.hexdigest()

# Cached function to get list of available Ollama language models (excluding embedding models)
@st.cache_data
def get_ollama_llm_models():
    try:
        # Use global ollama CLI command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        models = []
        for line in lines[1:]:
            model_name = line.split()[0].strip()
            if not (model_name.startswith('all-minilm') or model_name.startswith('nomic-embed')):
                if model_name and model_name not in models:
                    models.append(model_name)
        return models if models else ["granite3.3:2b"]
    except FileNotFoundError:
        st.error("Ollama CLI not found. Please install Ollama from https://ollama.com/ and ensure it's accessible in your system PATH.")
        return ["granite3.3:2b"]
    except subprocess.CalledProcessError as e:
        st.error(f"Error running 'ollama list': {str(e)}. Ensure Ollama is running (ollama serve).")
        return ["granite3.3:2b"]
    except Exception as e:
        st.error(f"Unexpected error fetching Ollama LLM models: {str(e)}")
        return ["granite3.3:2b"]

# Cached function to get list of available Ollama embedding models
@st.cache_data
def get_ollama_embedding_models():
    try:
        # Use global ollama CLI command
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True)
        lines = result.stdout.splitlines()
        models = []
        for line in lines[1:]:
            model_name = line.split()[0].strip()
            if model_name.startswith('all-minilm') or model_name.startswith('nomic-embed'):
                if model_name and model_name not in models:
                    models.append(model_name)
        return models if models else ["nomic-embed-text:latest"]
    except FileNotFoundError:
        st.error("Ollama CLI not found. Please install Ollama from https://ollama.com/ and ensure it's accessible in your system PATH.")
        return ["nomic-embed-text:latest"]
    except subprocess.CalledProcessError as e:
        st.error(f"Error running 'ollama list': {str(e)}. Ensure Ollama is running (ollama serve).")
        return ["nomic-embed-text:latest"]
    except Exception as e:
        st.error(f"Unexpected error fetching Ollama embedding models: {str(e)}")
        return ["nomic-embed-text:latest"]

# Cached function to create qa_chain in RAG mode
@st.cache_resource
def create_qa_chain(_file_hashes, uploaded_files):
    # Deduplicate files based on hashes
    unique_files = []
    seen_hashes = set()
    for file, file_hash in zip(uploaded_files, _file_hashes):
        if file_hash not in seen_hashes:
            unique_files.append(file)
            seen_hashes.add(file_hash)
    
    documents = []
    # Redirect stderr to filter CropBox warnings
    original_stderr = sys.stderr
    sys.stderr = StderrFilter(original_stderr)
    try:
        for file in unique_files:
            with pdfplumber.open(file) as pdf:
                text = ""
                for page in pdf.pages:
                    extracted_text = page.extract_text()
                    if extracted_text:
                        text += extracted_text
                documents.append(text)
    finally:
        # Restore original stderr
        sys.stderr = original_stderr
    if documents:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = []
        for doc in documents:
            if doc:
                split_texts = text_splitter.create_documents([doc])
                texts.extend(split_texts)
        if texts:
            embeddings = OllamaEmbeddings(model="nomic-embed-text:latest", base_url="http://localhost:11434")
            vector_store = FAISS.from_documents(texts, embeddings)
            llm = ChatOllama(model="granite3.3:2b", base_url="http://localhost:11434")
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vector_store.as_retriever()
            )
            return qa_chain
    return None

# Function to process RAG response (non-streaming)
def process_response(response):
    think_pattern = r'<think>(.*?)</think>'
    cleaned_response = re.sub(think_pattern, '', response).strip()
    return cleaned_response

# Generator function for streaming response
def stream_response(llm, prompt):
    accumulated_text = ""
    reasoning_text = []
    in_think_block = False
    think_start = "<think>"
    think_end = "</think>"
    reasoning_window = st.session_state.reasoning_window
    
    for chunk in llm.stream(prompt):
        chunk_text = chunk.content
        accumulated_text += chunk_text
        
        # Process chunk character by character to detect tags
        i = 0
        while i < len(chunk_text):
            if not in_think_block:
                # Look for <think>
                if accumulated_text.endswith(think_start):
                    in_think_block = True
                    accumulated_text = accumulated_text[:-len(think_start)]
                    i += len(think_start)
                    continue
                # Yield non-think text
                if accumulated_text:
                    cleaned_chunk = accumulated_text
                    accumulated_text = ""
                    if cleaned_chunk:
                        yield cleaned_chunk
                        # Clear reasoning window when non-think text starts
                        reasoning_window.empty()
            else:
                # Look for </think>
                if accumulated_text.endswith(think_end):
                    in_think_block = False
                    reasoning_content = accumulated_text[:-len(think_end)]
                    if reasoning_content:
                        reasoning_text.append(reasoning_content)
                        content = f'<div id="thinking-window" class="thinking-window">{" ".join(reasoning_text)}</div>'
                        reasoning_window.markdown(content, unsafe_allow_html=True)
                    accumulated_text = ""
                    i += len(think_end)
                    continue
                # Accumulate reasoning text
                if accumulated_text:
                    reasoning_text.append(accumulated_text)
                    content = f'<div id="thinking-window" class="thinking-window">{" ".join(reasoning_text)}</div>'
                    reasoning_window.markdown(content, unsafe_allow_html=True)
                    accumulated_text = ""
            i += 1
    
    # Handle any remaining non-think text
    if accumulated_text and not in_think_block:
        yield accumulated_text
        reasoning_window.empty()
    
    # Return cleaned response for chat history
    cleaned_response = process_response(" ".join(reasoning_text) + accumulated_text)
    return cleaned_response

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'graph_image' not in st.session_state:
    st.session_state.graph_image = None
if 'latex_expressions' not in st.session_state:
    st.session_state.latex_expressions = []
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "granite3.3:2b"
if 'selected_embedding_model' not in st.session_state:
    st.session_state.selected_embedding_model = "nomic-embed-text:latest"
if 'reasoning_window' not in st.session_state:
    st.session_state.reasoning_window = None

# Sidebar for mode selection, document loading, LaTeX, model selection, and graphing controls
with st.sidebar:
    st.header("Chat Mode")
    chat_mode = st.selectbox("Select chat mode", ["RAG Mode", "Direct Model Mode"])
    
    st.header("Document Loader")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    # Progress bar for document loading
    if uploaded_files:
        progress_bar = st.progress(0)
        total_files = len(uploaded_files)
        file_hashes = [compute_file_hash(file) for file in uploaded_files]
        
        st.session_state.qa_chain = create_qa_chain(file_hashes, uploaded_files)
        for i in range(total_files):
            progress_bar.progress((i + 1) / total_files)
        
        if st.session_state.qa_chain:
            st.success("Documents processed successfully!")
        else:
            st.error("No valid text extracted from uploaded documents.")
    
    # Math and Model Controls expander
    with st.expander("Math and Model Controls", expanded=False):
        st.subheader("Model Selection")
        available_llm_models = get_ollama_llm_models()
        available_embedding_models = get_ollama_embedding_models()
        
        # Set fixed models for RAG mode, otherwise use session state
        display_llm_model = "granite3.3:2b" if chat_mode == "RAG Mode" else st.session_state.selected_model
        display_embedding_model = "nomic-embed-text:latest" if chat_mode == "RAG Mode" else st.session_state.selected_embedding_model
        
        # Render dropdowns
        llm_index = available_llm_models.index(display_llm_model) if display_llm_model in available_llm_models else 0
        embedding_index = available_embedding_models.index(display_embedding_model) if display_embedding_model in available_embedding_models else 0
        
        selected_llm = st.selectbox(
            "Select LLM model",
            available_llm_models,
            index=llm_index,
            disabled=(chat_mode == "RAG Mode")
        )
        selected_embedding = st.selectbox(
            "Select embedding model",
            available_embedding_models,
            index=embedding_index,
            disabled=(chat_mode == "RAG Mode")
        )
        
        # Update session state only in Direct Model Mode
        if chat_mode == "Direct Model Mode":
            st.session_state.selected_model = selected_llm
            st.session_state.selected_embedding_model = selected_embedding
        
        # Add note for RAG mode
        if chat_mode == "RAG Mode":
            st.markdown("*Note: RAG Mode uses fixed models (LLM: granite3.3:2b, Embedding: nomic-embed-text:latest).*")
        
        st.subheader("LaTeX Expression")
        latex_expr = st.text_input("Enter LaTeX expression (e.g., \\frac{1}{2})", "\\frac{\\sum_{i=1}^{77}i}{\\sum_{i=1}^7i}")
        if st.button("Render LaTeX Expression"):
            if latex_expr:
                st.session_state.latex_expressions.append(latex_expr)
            else:
                st.error("Please enter a valid LaTeX expression.")
    
    # Graphing controls in a collapsed expander
    with st.expander("Graphing Controls", expanded=False):
        graph_type = st.selectbox("Select graph type", [
            "Parametric", "Function", "Polar", "Scatter", "3D Surface", "3D Scatter", "3D Line", 
            "Contour", "Bar", "Column", "Stacked Bar", "Pie Chart", "Histogram", "Line Chart", 
            "Area Chart", "Waterfall Chart", "Radar Chart", "Box Plot", "Violin Plot", "Error Bar"
        ])
        
        if graph_type == "Parametric":
            x_eq = st.text_input("X parametric equation (use t)", "sin(t)*(np.exp(cos(t))-2*cos(4*t)-sin(t/12)**5)")
            y_eq = st.text_input("Y parametric equation (use t)", "cos(t)*(np.exp(cos(t))-2*cos(4*t)-sin(t/12)**5)")
            t_min = st.number_input("t min", value=-10.0)
            t_max = st.number_input("t max", value=10.0)
        elif graph_type == "Function":
            func = st.text_input("Function (use x)", "x**2")
            x_min = st.number_input("x min", value=-10.0)
            x_max = st.number_input("x max", value=10.0)
        elif graph_type == "Polar":
            r_eq = st.text_input("Radius equation (use theta)", "sin(3*theta)")
            theta_min = st.number_input("theta min (radians)", value=0.0)
            theta_max = st.number_input("theta max (radians)", value=2*np.pi)
        elif graph_type == "Scatter":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                x_points = st.text_input("X coordinates (comma-separated numbers)", "1,2,3,4,5")
                y_points = st.text_input("Y coordinates (comma-separated numbers)", "2,4,3,5,1")
                sizes = st.text_input("Point sizes (comma-separated numbers, optional)", "")
                colors = st.text_input("Colors (comma-separated matplotlib colors, optional)", "")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="scatter_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        x_col = st.selectbox("Select X column", columns, key="scatter_x")
                        y_col = st.selectbox("Select Y column", columns, key="scatter_y")
                        size_col = st.selectbox("Select size column (optional)", ["None"] + columns, key="scatter_size")
                        color_col = st.selectbox("Select color column (optional)", ["None"] + columns, key="scatter_color")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "3D Surface":
            z_eq = st.text_input("Z equation (use x, y)", "np.sin(np.sqrt(x**2 + y**2))")
            x_min = st.number_input("x min", value=-5.0)
            x_max = st.number_input("x max", value=5.0)
            y_min = st.number_input("y min", value=-5.0)
            y_max = st.number_input("y max", value=5.0)
        elif graph_type == "3D Scatter":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                x_points = st.text_input("X coordinates (comma-separated numbers)", "1,2,3,4,5")
                y_points = st.text_input("Y coordinates (comma-separated numbers)", "2,4,3,5,1")
                z_points = st.text_input("Z coordinates (comma-separated numbers)", "1,3,2,4,5")
                sizes = st.text_input("Point sizes (comma-separated numbers, optional)", "")
                colors = st.text_input("Colors (comma-separated matplotlib colors, optional)", "")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="3d_scatter_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        x_col = st.selectbox("Select X column", columns, key="3d_scatter_x")
                        y_col = st.selectbox("Select Y column", columns, key="3d_scatter_y")
                        z_col = st.selectbox("Select Z column", columns, key="3d_scatter_z")
                        size_col = st.selectbox("Select size column (optional)", ["None"] + columns, key="3d_scatter_size")
                        color_col = st.selectbox("Select color column (optional)", ["None"] + columns, key="3d_scatter_color")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "3D Line":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                x_points = st.text_input("X coordinates (comma-separated numbers)", "1,2,3,4,5")
                y_points = st.text_input("Y coordinates (comma-separated numbers)", "2,4,3,5,1")
                z_points = st.text_input("Z coordinates (comma-separated numbers)", "1,3,2,4,5")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="3d_line_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        x_col = st.selectbox("Select X column", columns, key="3d_line_x")
                        y_col = st.selectbox("Select Y column", columns, key="3d_line_y")
                        z_col = st.selectbox("Select Z column", columns, key="3d_line_z")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "Contour":
            z_eq = st.text_input("Z equation (use x, y)", "x**2 + y**2")
            x_min = st.number_input("x min", value=-5.0)
            x_max = st.number_input("x max", value=5.0)
            y_min = st.number_input("y min", value=-5.0)
            y_max = st.number_input("y max", value=5.0)
            levels = st.number_input("Number of contour levels", value=10, min_value=1)
        elif graph_type in ["Bar", "Column"]:
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                categories = st.text_input("Categories (comma-separated)", "A,B,C,D")
                values = st.text_input("Values (comma-separated numbers)", "10,20,15,25")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="bar_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        category_col = st.selectbox("Select category column", columns, key="bar_category")
                        value_col = st.selectbox("Select value column", columns, key="bar_value")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "Stacked Bar":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                categories = st.text_input("Categories (comma-separated)", "A,B,C")
                series = st.text_area("Series values (one series per line, comma-separated numbers)", "10,20,15\n5,10,25")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="stacked_bar_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        category_col = st.selectbox("Select category column", columns, key="stacked_bar_category")
                        value_cols = st.multiselect("Select value columns (one per series)", columns, key="stacked_bar_values")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "Pie Chart":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                labels = st.text_input("Labels (comma-separated)", "A,B,C,D")
                values = st.text_input("Values (comma-separated numbers)", "30,20,25,25")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="pie_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        label_col = st.selectbox("Select label column", columns, key="pie_label")
                        value_col = st.selectbox("Select value column", columns, key="pie_value")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "Histogram":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                data = st.text_input("Data (comma-separated numbers)", "1,2,2,3,3,3,4,4,5")
                bins = st.number_input("Number of bins", value=10, min_value=1)
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="hist_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        data_col = st.selectbox("Select data column", columns, key="hist_data")
                        bins = st.number_input("Number of bins", value=10, min_value=1)
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "Line Chart":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                x_points = st.text_input("X coordinates (comma-separated numbers)", "1,2,3,4,5")
                y_points = st.text_input("Y coordinates (comma-separated numbers)", "2,4,3,5,1")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="line_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        x_col = st.selectbox("Select X column", columns, key="line_x")
                        y_col = st.selectbox("Select Y column", columns, key="line_y")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "Area Chart":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                x_points = st.text_input("X coordinates (comma-separated numbers)", "1,2,3,4,5")
                y_points = st.text_input("Y coordinates (comma-separated numbers)", "2,4,3,5,1")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="area_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        x_col = st.selectbox("Select X column", columns, key="area_x")
                        y_col = st.selectbox("Select Y column", columns, key="area_y")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "Waterfall Chart":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                categories = st.text_input("Categories (comma-separated)", "Start,Income,Expenses,End")
                changes = st.text_input("Value changes (comma-separated numbers)", "100,50,-30,-20")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="waterfall_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        category_col = st.selectbox("Select category column", columns, key="waterfall_category")
                        change_col = st.selectbox("Select change column", columns, key="waterfall_change")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "Radar Chart":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                categories = st.text_input("Categories (comma-separated)", "A,B,C,D,E")
                series = st.text_area("Series values (one series per line, comma-separated numbers)", "4,3,2,5,4\n2,4,5,3,2")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="radar_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        category_col = st.selectbox("Select category column", columns, key="radar_category")
                        value_cols = st.multiselect("Select value columns (one per series)", columns, key="radar_values")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        elif graph_type == "Box Plot":
            csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="box_csv")
            if csv_file:
                try:
                    df = pd.read_csv(csv_file)
                    columns = df.columns.tolist()
                    data_cols = st.multiselect("Select data columns", columns, key="box_data")
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                df = None
            else:
                df = None
                st.warning("Please upload a CSV file.")
        elif graph_type == "Violin Plot":
            csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="violin_csv")
            if csv_file:
                try:
                    df = pd.read_csv(csv_file)
                    columns = df.columns.tolist()
                    data_cols = st.multiselect("Select data columns", columns, key="violin_data")
                except Exception as e:
                    st.error(f"Error reading CSV file: {str(e)}")
                df = None
            else:
                df = None
                st.warning("Please upload a CSV file.")
        elif graph_type == "Error Bar":
            data_source = st.radio("Data source", ["Manual Input", "CSV File"])
            if data_source == "Manual Input":
                x_points = st.text_input("X coordinates (comma-separated numbers)", "1,2,3,4,5")
                y_points = st.text_input("Y coordinates (comma-separated numbers)", "2,4,3,5,1")
                y_err = st.text_input("Y errors (comma-separated numbers)", "0.5,0.4,0.3,0.5,0.2")
            else:
                csv_file = st.file_uploader("Upload CSV file", type=["csv"], key="error_csv")
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        columns = df.columns.tolist()
                        x_col = st.selectbox("Select X column", columns, key="error_x")
                        y_col = st.selectbox("Select Y column", columns, key="error_y")
                        err_col = st.selectbox("Select error column", columns, key="error_err")
                    except Exception as e:
                        st.error(f"Error reading CSV file: {str(e)}")
                        df = None
                else:
                    df = None
                    if data_source == "CSV File":
                        st.warning("Please upload a CSV file.")
        
        if st.button("Generate Graph"):
            try:
                if graph_type in ["Parametric", "Function", "Scatter", "Contour", "Bar", "Column", "Stacked Bar", "Pie Chart", "Histogram", "Line Chart", "Area Chart", "Waterfall Chart", "Box Plot", "Violin Plot", "Error Bar"]:
                    fig, ax = plt.subplots()
                elif graph_type == "Polar":
                    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                elif graph_type == "Radar Chart":
                    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
                elif graph_type in ["3D Surface", "3D Scatter", "3D Line"]:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                
                if graph_type == "Parametric":
                    t = np.linspace(t_min, t_max, 1000)
                    x = eval(x_eq, {"t": t, "np": np, "cos": np.cos, "sin": np.sin, "exp": np.exp})
                    y = eval(y_eq, {"t": t, "np": np, "cos": np.cos, "sin": np.sin, "exp": np.exp})
                    ax.plot(x, y)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(True)
                elif graph_type == "Function":
                    x = np.linspace(x_min, x_max, 1000)
                    y = eval(func, {"x": x, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                    ax.plot(x, y)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(True)
                elif graph_type == "Polar":
                    theta = np.linspace(theta_min, theta_max, 1000)
                    r = eval(r_eq, {"theta": theta, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                    ax.plot(theta, r)
                    ax.set_theta_zero_location("E")
                elif graph_type == "Scatter":
                    if data_source == "Manual Input":
                        x = [float(x) for x in x_points.split(",") if x.strip()]
                        y = [float(y) for y in y_points.split(",") if y.strip()]
                        if len(x) != len(y):
                            raise ValueError("X and Y coordinates must have the same number of points")
                        sizes = [float(s) for s in sizes.split(",") if s.strip()] if sizes else None
                        colors = colors.split(",") if colors else None
                        if sizes and len(sizes) != len(x):
                            raise ValueError("Sizes must match the number of points")
                        if colors and len(colors) != len(x):
                            raise ValueError("Colors must match the number of points")
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        x = df[x_col].dropna().astype(float).tolist()
                        y = df[y_col].dropna().astype(float).tolist()
                        if len(x) != len(y):
                            raise ValueError("Selected X and Y columns must have the same number of non-null values")
                        sizes = df[size_col].dropna().astype(float).tolist() if size_col != "None" else None
                        colors = df[color_col].tolist() if color_col != "None" else None
                        if sizes and len(sizes) != len(x):
                            raise ValueError("Size column must match the number of points")
                        if colors and len(colors) != len(x):
                            raise ValueError("Color column must match the number of points")
                    ax.scatter(x, y, s=sizes, c=colors)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(True)
                elif graph_type == "3D Surface":
                    x = np.linspace(x_min, x_max, 100)
                    y = np.linspace(y_min, y_max, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = eval(z_eq, {"x": X, "y": Y, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                    ax.plot_surface(X, Y, Z, cmap='viridis')
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                elif graph_type == "3D Scatter":
                    if data_source == "Manual Input":
                        x = [float(x) for x in x_points.split(",") if x.strip()]
                        y = [float(y) for y in y_points.split(",") if y.strip()]
                        z = [float(z) for z in z_points.split(",") if z.strip()]
                        if not (len(x) == len(y) == len(z)):
                            raise ValueError("X, Y, and Z coordinates must have the same number of points")
                        sizes = [float(s) for s in sizes.split(",") if s.strip()] if sizes else None
                        colors = colors.split(",") if colors else None
                        if sizes and len(sizes) != len(x):
                            raise ValueError("Sizes must match the number of points")
                        if colors and len(colors) != len(x):
                            raise ValueError("Colors must match the number of points")
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        x = df[x_col].dropna().astype(float).tolist()
                        y = df[y_col].dropna().astype(float).tolist()
                        z = df[z_col].dropna().astype(float).tolist()
                        if not (len(x) == len(y) == len(z)):
                            raise ValueError("Selected X, Y, and Z columns must have the same number of non-null values")
                        sizes = df[size_col].dropna().astype(float).tolist() if size_col != "None" else None
                        colors = df[color_col].tolist() if color_col != "None" else None
                        if sizes and len(sizes) != len(x):
                            raise ValueError("Size column must match the number of points")
                        if colors and len(colors) != len(x):
                            raise ValueError("Color column must match the number of points")
                    ax.scatter(x, y, z, s=sizes, c=colors)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                elif graph_type == "3D Line":
                    if data_source == "Manual Input":
                        x = [float(x) for x in x_points.split(",") if x.strip()]
                        y = [float(y) for y in y_points.split(",") if y.strip()]
                        z = [float(z) for z in z_points.split(",") if z.strip()]
                        if not (len(x) == len(y) == len(z)):
                            raise ValueError("X, Y, and Z coordinates must have the same number of points")
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        x = df[x_col].dropna().astype(float).tolist()
                        y = df[y_col].dropna().astype(float).tolist()
                        z = df[z_col].dropna().astype(float).tolist()
                        if not (len(x) == len(y) == len(z)):
                            raise ValueError("Selected X, Y, and Z columns must have the same number of non-null values")
                    ax.plot(x, y, z)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.set_zlabel("z")
                elif graph_type == "Contour":
                    x = np.linspace(x_min, x_max, 100)
                    y = np.linspace(y_min, y_max, 100)
                    X, Y = np.meshgrid(x, y)
                    Z = eval(z_eq, {"x": X, "y": Y, "np": np, "sin": np.sin, "cos": np.cos, "exp": np.exp})
                    ax.contour(X, Y, Z, levels=levels, cmap='viridis')
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(True)
                elif graph_type in ["Bar", "Column"]:
                    if data_source == "Manual Input":
                        categories = [c.strip() for c in categories.split(",") if c.strip()]
                        values = [float(v) for v in values.split(",") if v.strip()]
                        if len(categories) != len(values):
                            raise ValueError("Number of categories must match number of values")
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        categories = df[category_col].dropna().astype(str).tolist()
                        values = df[value_col].dropna().astype(float).tolist()
                        if len(categories) != len(values):
                            raise ValueError("Category and value columns must have the same number of non-null values")
                    ax.bar(categories, values)
                    ax.set_xlabel("Categories")
                    ax.set_ylabel("Values")
                    ax.grid(True, axis='y')
                elif graph_type == "Stacked Bar":
                    if data_source == "Manual Input":
                        categories = [c.strip() for c in categories.split(",") if c.strip()]
                        series_values = [[float(v) for v in line.split(",") if v.strip()] for line in series.split("\n") if line.strip()]
                        if not all(len(s) == len(categories) for s in series_values):
                            raise ValueError("Each series must have the same number of values as categories")
                    else:
                        if not df or not value_cols:
                            raise ValueError("No CSV data or value columns selected")
                        categories = df[category_col].dropna().astype(str).tolist()
                        series_values = [df[col].dropna().astype(float).tolist() for col in value_cols]
                        if not all(len(s) == len(categories) for s in series_values):
                            raise ValueError("Each value column must have the same number of non-null values as categories")
                    bottom = np.zeros(len(categories))
                    for i, series in enumerate(series_values):
                        ax.bar(categories, series, bottom=bottom, label=f"Series {i+1}")
                        bottom += np.array(series)
                    ax.set_xlabel("Categories")
                    ax.set_ylabel("Values")
                    ax.legend()
                    ax.grid(True, axis='y')
                elif graph_type == "Pie Chart":
                    if data_source == "Manual Input":
                        labels = [l.strip() for l in labels.split(",") if l.strip()]
                        values = [float(v) for v in values.split(",") if v.strip()]
                        if len(labels) != len(values):
                            raise ValueError("Number of labels must match number of values")
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        labels = df[label_col].dropna().astype(str).tolist()
                        values = df[value_col].dropna().astype(float).tolist()
                        if len(labels) != len(values):
                            raise ValueError("Label and value columns must have the same number of non-null values")
                    ax.pie(values, labels=labels, autopct='%1.1f%%')
                elif graph_type == "Histogram":
                    if data_source == "Manual Input":
                        data = [float(d) for d in data.split(",") if d.strip()]
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        data = df[data_col].dropna().astype(float).tolist()
                    ax.hist(data, bins=bins)
                    ax.set_xlabel("Value")
                    ax.set_ylabel("Frequency")
                    ax.grid(True, axis='y')
                elif graph_type == "Line Chart":
                    if data_source == "Manual Input":
                        x = [float(x) for x in x_points.split(",") if x.strip()]
                        y = [float(y) for y in y_points.split(",") if y.strip()]
                        if len(x) != len(y):
                            raise ValueError("X and Y coordinates must have the same number of points")
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        x = df[x_col].dropna().astype(float).tolist()
                        y = df[y_col].dropna().astype(float).tolist()
                        if len(x) != len(y):
                            raise ValueError("Selected X and Y columns must have the same number of non-null values")
                    ax.plot(x, y)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(True)
                elif graph_type == "Area Chart":
                    if data_source == "Manual Input":
                        x = [float(x) for x in x_points.split(",") if x.strip()]
                        y = [float(y) for y in y_points.split(",") if y.strip()]
                        if len(x) != len(y):
                            raise ValueError("X and Y coordinates must have the same number of points")
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        x = df[x_col].dropna().astype(float).tolist()
                        y = df[y_col].dropna().astype(float).tolist()
                        if len(x) != len(y):
                            raise ValueError("Selected X and Y columns must have the same number of non-null values")
                    ax.fill_between(x, y)
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(True)
                elif graph_type == "Waterfall Chart":
                    if data_source == "Manual Input":
                        categories = [c.strip() for c in categories.split(",") if c.strip()]
                        changes = [float(c) for c in changes.split(",") if c.strip()]
                        if len(categories) != len(changes):
                            raise ValueError("Number of categories must match number of changes")
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        categories = df[category_col].dropna().astype(str).tolist()
                        changes = df[change_col].dropna().astype(float).tolist()
                        if len(categories) != len(changes):
                            raise ValueError("Category and change columns must have the same number of non-null values")
                    values = np.cumsum([0] + changes[:-1])
                    for i, (val, change) in enumerate(zip(values, changes)):
                        color = 'green' if change >= 0 else 'red'
                        ax.bar(categories[i], change, bottom=val, color=color)
                    ax.set_xlabel("Categories")
                    ax.set_ylabel("Value")
                    ax.grid(True, axis='y')
                elif graph_type == "Radar Chart":
                    if data_source == "Manual Input":
                        categories = [c.strip() for c in categories.split(",") if c.strip()]
                        series_values = [[float(v) for v in line.split(",") if v.strip()] for line in series.split("\n") if line.strip()]
                        if not all(len(s) == len(categories) for s in series_values):
                            raise ValueError("Each series must have the same number of values as categories")
                    else:
                        if not df or not value_cols:
                            raise ValueError("No CSV data or value columns selected")
                        categories = df[category_col].dropna().astype(str).tolist()
                        series_values = [df[col].dropna().astype(float).tolist() for col in value_cols]
                        if not all(len(s) == len(categories) for s in series_values):
                            raise ValueError("Each value column must have the same number of non-null values as categories")
                    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
                    angles += angles[:1]
                    for i, series in enumerate(series_values):
                        values = series + series[:1]
                        ax.plot(angles, values, label=f"Series {i+1}")
                        ax.fill(angles, values, alpha=0.1)
                    ax.set_xticks(angles[:-1])
                    ax.set_xticklabels(categories)
                    ax.legend()
                elif graph_type == "Box Plot":
                    if not df or not data_cols:
                        raise ValueError("No CSV data or data columns selected")
                    data = [df[col].dropna().astype(float).tolist() for col in data_cols]
                    ax.boxplot(data, labels=data_cols)
                    ax.set_ylabel("Values")
                    ax.grid(True, axis='y')
                elif graph_type == "Violin Plot":
                    if not df or not data_cols:
                        raise ValueError("No CSV data or data columns selected")
                    data = [df[col].dropna().astype(float).tolist() for col in data_cols]
                    ax.violinplot(data)
                    ax.set_xticks(range(1, len(data_cols) + 1))
                    ax.set_xticklabels(data_cols)
                    ax.set_ylabel("Values")
                    ax.grid(True, axis='y')
                elif graph_type == "Error Bar":
                    if data_source == "Manual Input":
                        x = [float(x) for x in x_points.split(",") if x.strip()]
                        y = [float(y) for y in y_points.split(",") if y.strip()]
                        y_err = [float(e) for e in y_err.split(",") if e.strip()]
                        if not (len(x) == len(y) == len(y_err)):
                            raise ValueError("X, Y, and error values must have the same number of points")
                    else:
                        if not df:
                            raise ValueError("No CSV data available")
                        x = df[x_col].dropna().astype(float).tolist()
                        y = df[y_col].dropna().astype(float).tolist()
                        y_err = df[err_col].dropna().astype(float).tolist()
                        if not (len(x) == len(y) == len(y_err)):
                            raise ValueError("Selected X, Y, and error columns must have the same number of non-null values")
                    ax.errorbar(x, y, yerr=y_err, fmt='o')
                    ax.set_xlabel("x")
                    ax.set_ylabel("y")
                    ax.grid(True)
                
                buf = BytesIO()
                plt.savefig(buf, format="png", bbox_inches='tight')
                buf.seek(0)
                img_str = base64.b64encode(buf.getvalue()).decode()
                st.session_state.graph_image = f"data:image/png;base64,{img_str}"
                plt.close()
            except Exception as e:
                st.error(f"Error generating graph: {str(e)}")

# Inject MathJax for LaTeX rendering
mathjax_script = """
<script src="/static/mathjax/es5/tex-chtml.js" id="MathJax-script" async></script>
<script>
    MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']]
        },
        chtml: {
            scale: 1.1
        }
    };
</script>
"""
components.html(mathjax_script, height=0)

# Inject CSS for floating reasoning window
reasoning_window_css = """
<style>
.thinking-window {
    position: fixed;
    top: 10px;
    right: 10px;
    background-color: rgba(255, 255, 255, 0.5);
    padding: 10px;
    border-radius: 8px;
    border: 1.1px solid #ccc;
    max-width: 400px;
    max-height: 300px;
    overflow-y: auto;
    font-size: 14px;
    color: blue;
    z-index: 1000;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    -webkit-backdrop-filter: blur(5px);
    backdrop-filter: blur(5px);
}
.thinking-window::-webkit-scrollbar {
    width: 8px;
}
.thinking-window::-webkit-scrollbar-thumb {
    background: #888;
    border-radius: 4px;
}
.thinking-window::-webkit-scrollbar-track {
    background: transparent;
}
.thinking-window {
    scrollbar-width: thin;
    scrollbar-color: #888 transparent;
}
.thinking-window > * {
    margin: 0;
}
.thinking-window:empty {
    display: none;
}
</style>
"""
st.markdown(reasoning_window_css, unsafe_allow_html=True)

# Inject CSS to reduce top padding for title
title_position_css = """
<style>
.stApp {
    padding-top: 0px !important;
}
</style>
"""
st.markdown(title_position_css, unsafe_allow_html=True)

# Main chat interface
st.title("RadioSport Chat")
st.markdown(f"Version {VERSION}")

# Floating window placeholder (separate from chat container)
st.session_state.reasoning_window = st.empty()

# Display chat history
chat_container = st.container()
with chat_container:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if st.session_state.graph_image:
        st.image(st.session_state.graph_image)
    
    for expr in st.session_state.latex_expressions:
        st.markdown(f"${expr}$")

# Chat input at bottom
user_input = st.chat_input("Ask a question...")

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.session_state.graph_image = None
    st.session_state.latex_expressions = []
    st.session_state.reasoning_window.empty()  # Clear reasoning window
    st.rerun()

# Process user query
if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)
    
    st.session_state.chat_history.append({"role": "user", "content": user_input})
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            if chat_mode == "RAG Mode":
                if st.session_state.qa_chain:
                    response = st.session_state.qa_chain.invoke({"query": user_input})["result"]
                    cleaned_response = process_response(response)
                    st.markdown(cleaned_response)
                    st.session_state.chat_history.append({"role": "assistant", "content": cleaned_response})
                else:
                    st.warning("Please upload and process documents for RAG Mode.")
            else:  # Direct Model Mode
                try:
                    llm = ChatOllama(model=st.session_state.selected_model, base_url="http://localhost:11434", streaming=True)
                    prompt = "You are an assistant. Use the following conversation history to provide accurate and context-aware responses. Pay attention to any corrections or clarifications provided and avoid repeating errors.\n\n"
                    for msg in st.session_state.chat_history[:-1]:
                        if msg["role"] == "user":
                            prompt += f"User: {msg['content']}\n"
                        else:
                            prompt += f"Assistant: {msg['content']}\n"
                    prompt += f"User: {user_input}\nAssistant: "
                    
                    # Stream response
                    cleaned_response = st.write_stream(stream_response(llm, prompt))
                    if cleaned_response:
                        st.session_state.chat_history.append({"role": "assistant", "content": cleaned_response})
                    
                except Exception as e:
                    st.error(f"Error connecting to Ollama in Direct Model Mode: {str(e)}")
                    st.session_state.reasoning_window.empty()

# Add script for periodic scrolling of the reasoning window
st.markdown("""
<script>
if (!window.scrollInterval) {
  window.scrollInterval = setInterval(function() {
    const targetNode = document.getElementById('thinking-window');
    if (targetNode) {
      targetNode.scrollTop = targetNode.scrollHeight;
    }
  }, 100);
}
</script>
""", unsafe_allow_html=True)