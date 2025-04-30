import os
import tempfile
import base64
import logging
import pdfplumber
from PIL import Image
import pytesseract
import streamlit as st
from config import APP_VERSION

logger = logging.getLogger("rag_assistant")

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

def estimate_total_chunks(uploaded_files):
    total_chunks = 0
    for file in uploaded_files:
        try:
            file.seek(0)
            if file.type == "application/pdf":
                with pdfplumber.open(file) as pdf:
                    pages = [page for page in pdf.pages if page.extract_text()]
                    total_chunks += len(pages)
            elif file.type.startswith("image/"):
                total_chunks += 1
            elif file.type == "text/plain":
                try:
                    text_content = file.read().decode("utf-8")
                except UnicodeDecodeError:
                    text_content = file.read().decode("latin1", errors="ignore")
                if text_content.strip():
                    chunk_size = 2000
                    chunks = [text_content[i:i + chunk_size] for i in range(0, len(text_content), chunk_size)]
                    total_chunks += len([chunk for chunk in chunks if chunk.strip()])
                else:
                    total_chunks += 1
            else:
                total_chunks += 1
            file.seek(0)
        except Exception as e:
            logger.warning(f"Error estimating chunks for {file.name}: {str(e)}")
            total_chunks += 1
    return max(total_chunks, 1)

def extract_chunks(uploaded_files, debug_mode=False):
    chunks = []
    messages = []
    errors = []
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_files = []
        for file in uploaded_files:
            try:
                temp_path = os.path.join(temp_dir, file.name)
                with open(temp_path, "wb") as f:
                    f.write(file.getvalue())
                temp_files.append(temp_path)
                if file.type == "application/pdf":
                    try:
                        with pdfplumber.open(temp_path) as pdf:
                            pages_text = [page.extract_text() for page in pdf.pages if page.extract_text()]
                    except Exception as e:
                        errors.append(f"Error extracting text from PDF {file.name}: {str(e)}")
                        logger.error(f"PDF text extraction error: {e}")
                        chunks.append((file.name, "", "text"))
                        messages.append(f"No text extracted from {file.name}")
                        continue
                    if not pages_text:
                        errors.append(f"No text extracted from {file.name}")
                        logger.warning(f"No text extracted from PDF: {file.name}")
                        chunks.append((file.name, "", "text"))
                        messages.append(f"No text extracted from {file.name}")
                        continue
                    messages.append(f"Extracted text from {len(pages_text)} pages in {file.name}")
                    for text in pages_text:
                        if text.strip():
                            chunks.append((file.name, text, "text"))
                elif file.type.startswith("image/"):
                    img = Image.open(temp_path)
                    max_size = 1024
                    if img.width > max_size or img.height > max_size:
                        img.thumbnail((max_size, max_size))
                        resized_path = os.path.join(temp_dir, f"resized_{file.name}")
                        img.save(resized_path)
                        temp_files.append(resized_path)
                        temp_path = resized_path
                    ocr_text = pytesseract.image_to_string(img).strip()
                    if not ocr_text:
                        errors.append(f"No text extracted from {file.name}")
                        logger.warning(f"No OCR text extracted from image: {file.name}")
                        chunks.append((file.name, "", "text"))
                        messages.append(f"No text extracted from {file.name}")
                        continue
                    messages.append(f"Extracted OCR text from {file.name}")
                    chunks.append((file.name, ocr_text, "text"))
                elif file.type == "text/plain":
                    try:
                        with open(temp_path, "r", encoding="utf-8") as f:
                            text_content = f.read()
                    except UnicodeDecodeError:
                        with open(temp_path, "r", encoding="latin1") as f:
                            text_content = f.read()
                    if not text_content.strip():
                        errors.append(f"Empty text file: {file.name}")
                        logger.warning(f"Empty text file: {file.name}")
                        chunks.append((file.name, "", "text"))
                        messages.append(f"No text extracted from {file.name}")
                        continue
                    chunk_size = 2000
                    text_chunks = [text_content[i:i + chunk_size] for i in range(0, len(text_content), chunk_size)]
                    messages.append(f"Split {file.name} into {len(text_chunks)} chunks")
                    for chunk in text_chunks:
                        if chunk.strip():
                            chunks.append((file.name, chunk, "text"))
                else:
                    errors.append(f"Unsupported file type: {file.type}")
                    logger.warning(f"Unsupported file type: {file.type}")
                    chunks.append((file.name, "", "text"))
                    messages.append(f"Unsupported file type for {file.name}")
            except Exception as e:
                errors.append(f"Error processing {file.name}: {str(e)}")
                logger.error(f"File processing error: {e}")
                chunks.append((file.name, "", "text"))
                messages.append(f"Error processing {file.name}")
        return chunks, messages, errors, temp_files

def process_chunk(app, file_name, chunk_text, data_type, debug_mode, completed_chunks, total_chunks):
    try:
        if not chunk_text.strip():
            raise ValueError("Empty chunk text")
        stat = app.add(chunk_text, data_type=data_type, debug_mode=debug_mode, file_name=file_name)
        with completed_chunks.get_lock():
            completed_chunks.value += 1
        if debug_mode:
            logger.debug(f"Incremented completed_chunks to {completed_chunks.value} for chunk in {file_name}")
        status_message = f"Processed chunk {completed_chunks.value}/{total_chunks} for {file_name}: {stat['status']}"
        return file_name, stat, status_message
    except Exception as e:
        if debug_mode:
            logger.error(f"Error processing chunk for {file_name}: {str(e)}")
        with completed_chunks.get_lock():
            completed_chunks.value += 1
        if debug_mode:
            logger.debug(f"Incremented completed_chunks to {completed_chunks.value} for failed chunk in {file_name}")
        status_message = f"Processed chunk {completed_chunks.value}/{total_chunks} for {file_name}: Failed ({str(e)})"
        return file_name, {
            "success": False,
            "error": str(e),
            "duration": 0,
            "embedding_dimension": 0,
            "status": "Failed",
            "total_chunks": 1,
            "successful_chunks": 0,
            "total_duration": 0,
            "avg_duration": 0,
        }, status_message