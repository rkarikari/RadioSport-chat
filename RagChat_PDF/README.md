# RadioSport Chat (rag.py)

**Version**: v2.5.0 (2025-05-14)

RadioSport Chat is a Streamlit-based web application that provides an interactive chat interface powered by Ollama models. It supports Retrieval-Augmented Generation (RAG) for answering questions based on uploaded PDF documents and a direct model mode for general queries. The app also includes advanced graphing capabilities, LaTeX rendering, and a clean, efficient interface optimized for performance.

## Features

- **Dual Chat Modes**:
  - **RAG Mode**: Answers queries using content from uploaded PDFs with fixed models (`granite3.3:2b` for LLM, `nomic-embed-text:latest` for embeddings).
  - **Direct Model Mode**: General-purpose chat with customizable Ollama models.
- **PDF Document Processing**:
  - Upload multiple PDFs for text extraction using `pdfplumber`.
  - File deduplication to prevent redundant processing.
  - Progress bar for document loading.
- **Graphing Capabilities**:
  - Supports 19 graph types: Parametric, Function, Polar, Scatter, 3D Surface, 3D Scatter, 3D Line, Contour, Bar, Column, Stacked Bar, Pie Chart, Histogram, Line Chart, Area Chart, Waterfall Chart, Radar Chart, Box Plot, Violin Plot, Error Bar.
  - Manual input or CSV file upload for data.
  - Customizable parameters (e.g., equations, ranges, bins).
- **LaTeX Rendering**:
  - Input and render mathematical expressions using MathJax.
  - Store and display multiple expressions.
- **Model Selection**:
  - Choose from available Ollama LLM and embedding models in Direct Model Mode.
  - Fixed models in RAG Mode for consistency.
- **Floating Reasoning Window**:
  - Displays `<think>...</think>` reasoning text in a separate, auto-scrolling window during streaming responses.
- **Performance Optimizations**:
  - Cached QA chain creation and model listing.
  - Suppressed unnecessary logging for a clean console.
  - Efficient PDF processing with deduplication.

## Functionality

### Chat Interface
- **Input**: Enter queries via a chat input box at the bottom of the page.
- **History**: View conversation history with user and assistant messages.
- **Clear History**: Reset chat history, graphs, and LaTeX expressions with a single button.
- **Streaming Responses**: Direct Model Mode streams responses with reasoning text in a floating window.

### RAG Mode
- Upload PDFs in the sidebar under "Document Loader".
- The app processes PDFs, extracts text, and creates a FAISS vector store for retrieval.
- Queries are answered based on document content using `granite3.3:2b` and `nomic-embed-text:latest`.
- A progress bar shows processing status, with success/error messages.

### Direct Model Mode
- Queries are sent directly to the selected Ollama model without document context.
- Supports streaming responses with reasoning text displayed in a floating window.
- Model selection is available in the "Math and Model Controls" expander.

### Graphing Controls
- Located in the sidebar under "Graphing Controls" (collapsed by default).
- Select from 19 graph types and input data manually or via CSV upload.
- Customize parameters (e.g., equations for Parametric graphs, column selections for CSV-based graphs).
- Graphs are displayed below the chat history after generation.

### LaTeX Rendering
- Enter LaTeX expressions in the "Math and Model Controls" expander.
- Rendered expressions are displayed below the chat history using MathJax.
- Multiple expressions can be stored and displayed.

### UI Elements
- **Sidebar**: Contains mode selection, document uploader, model controls, LaTeX input, and graphing controls.
- **Main Page**: Shows the chat interface, version number, chat history, graphs, and LaTeX expressions.
- **Floating Window**: Displays reasoning text during streaming, with auto-scrolling and a transparent, blurred background.

## Installation Notes

### Prerequisites
- **Python**: Version 3.11 or later.
  ```bash
  python --version
  ```
- **Ollama**: Install and run Ollama locally with models `granite3.3:2b` and `nomic-embed-text:latest`.
  - Download from [Ollama's website](https://ollama.ai/).
  - Start Ollama:
    ```bash
    ollama serve
    ```
  - Pull required models:
    ```bash
    ollama pull granite3.3:2b
    ollama pull nomic-embed-text:latest
    ```

### Setup
1. **Clone or Download**:
   - Place `rag.py` and `requirements.txt` in a directory (e.g., `C:\test`).

2. **Create a Virtual Environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   The `requirements.txt` includes:
   ```
   streamlit>=1.31.0
   langchain-ollama>=0.1.0
   langchain-community>=0.2.0
   pdfplumber>=0.11.0
   matplotlib>=3.7.0
   numpy>=1.24.0
   pandas>=2.0.0
   faiss-cpu>=1.8.0
   ```

4. **Verify Ollama**:
   - Ensure Ollama is running (`http://localhost:11434`).
   - Check available models:
     ```bash
     ollama list
     ```

### Running the Application
1. **Start the Streamlit App**:
   ```bash
   streamlit run rag.py
   ```
   - The app will open in your default browser (e.g., `http://localhost:8501`).

2. **Usage**:
   - Select a chat mode (RAG or Direct Model) in the sidebar.
   - For RAG Mode, upload PDFs and wait for processing confirmation.
   - Enter queries in the chat input box.
   - Use the "Math and Model Controls" expander for model selection or LaTeX input.
   - Use the "Graphing Controls" expander to generate graphs.
   - Clear chat history as needed.

## Additional Notes

### Troubleshooting
- **Ollama Connection Errors**:
  - Ensure Ollama is running and models are pulled.
  - Verify the Ollama URL (`http://localhost:11434`) is accessible.
- **PDF Processing Issues**:
  - Check that PDFs are valid and not corrupted.
  - Clear Streamlit cache if issues persist:
    ```bash
    streamlit cache clear
    ```
- **Graphing Errors**:
  - Ensure equations or CSV data are correctly formatted.
  - Check error messages in the UI for specific issues.
- **Console Output**:
  - The console should be silent unless errors occur.
  - If unexpected logs appear, verify `requirements.txt` dependencies are installed correctly.

### Usage Tips
- **RAG Mode**: Upload relevant PDFs before querying. Ensure documents contain text (not scanned images).
- **Direct Model Mode**: Experiment with different Ollama models for varied response styles.
- **Graphing**: Use CSV uploads for large datasets to simplify input.
- **LaTeX**: Test expressions in a LaTeX editor if rendering issues occur.

### Changelog
See the `CHANGELOG` section in `rag.py` for a detailed version history. Key updates in v2.4.13:
- Removed unnecessary logging for a cleaner console.
- Optimized PDF processing with deduplication.
- Fixed LangChain deprecation warnings.

For issues or contributions, please report bugs or submit pull requests via the repository (if hosted) or contact the maintainer.