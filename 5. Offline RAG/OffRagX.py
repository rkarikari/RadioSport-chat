import os
import warnings
from dotenv import load_dotenv
from docling.document_converter import DocumentConverter
from pathlib import Path
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain_ollama import OllamaEmbeddings
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

import streamlit as st
import pyttsx3

# Environment setup
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
warnings.filterwarnings("ignore")
load_dotenv()

def load_and_convert_document(file_path):
    """
    Converts a document to markdown format.
    
    Parameters:
    - file_path: Path to the document file.
    
    Returns:
    - Markdown content of the document.
    """
    converter = DocumentConverter()
    result = converter.convert(file_path)
    return result.document.export_to_markdown()

def get_markdown_splits(markdown_content):
    """
    Splits markdown content into chunks based on headers.
    
    Parameters:
    - markdown_content: Markdown content to be split.
    
    Returns:
    - List of markdown chunks.
    """
    headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3"), ("####", "Header 4"), ("#####", "Header 5"), ("######", "Header 6")]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
    return markdown_splitter.split_text(markdown_content)

def setup_vector_store(chunks):
    """
    Creates a vector store from document chunks.
    
    Parameters:
    - chunks: List of document chunks.
    
    Returns:
    - Vector store containing the document chunks.
    """
    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
    single_vector = embeddings.embed_query("this is some text data")
    index = faiss.IndexFlatL2(len(single_vector))
    vector_store = FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={}
    )
    vector_store.add_documents(documents=chunks)
    return vector_store

def format_docs(docs):
    """
    Formats a list of documents into a single string.
    
    Parameters:
    - docs: List of documents.
    
    Returns:
    - Formatted string of documents.
    """
    return "\n\n".join([doc.page_content for doc in docs])

def create_rag_chain(retriever):
    """
    Creates a RAG chain for question answering.
    
    Parameters:
    - retriever: Document retriever.
    
    Returns:
    - RAG chain.
    """
    prompt = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know.
        Answer in bullet points. Make sure your answer is relevant to the question and it is answered from the context only.
        Do not include any internal reasoning or text within in your answer. Provide only the final answer.
        Question: {question} 
        Context: {context} 
        Answer:
    """
    model = ChatOllama(model="deepseek-r1:1.5b", base_url="http://localhost:11434")
    prompt_template = ChatPromptTemplate.from_template(prompt)
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt_template
        | model
        | StrOutputParser()
    )

def process_selected_file(file_path):
    """
    Processes a selected document and returns the RAG chain and markdown content.
    
    Parameters:
    - file_path: Path to the document file.
    
    Returns:
    - RAG chain and markdown content.
    """
    @st.cache_resource(show_spinner="Processing document...")
    def cached_processing(_file_path):
        markdown_content = load_and_convert_document(_file_path)
        chunks = get_markdown_splits(markdown_content)
        vector_store = setup_vector_store(chunks)
        return vector_store, markdown_content
    
    vector_store, markdown_content = cached_processing(file_path)
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 3})
    return create_rag_chain(retriever), markdown_content

# Main execution logic for Streamlit app
if __name__ == "__main__":
    st.title("🧠 RCS- Chatbot")
    st.markdown("GET MORE INFO @ [RCS](https://rcs.edu.gh)")

    # List available PDF files
    file_directory = 'rag-dataset/'
    pdf_files = [f for f in os.listdir(file_directory) if f.endswith('.pdf')]
    
    selected_file = st.selectbox("Select Document", [""] + pdf_files)
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'current_file' not in st.session_state:
        st.session_state['current_file'] = None
    if 'enable_voice' not in st.session_state:
        st.session_state['enable_voice'] = False

    # Voice toggle
    st.write("##### Voice Output")
    st.session_state['enable_voice'] = st.checkbox("Enable Voice Output", value=st.session_state['enable_voice'])

    # Only process file when it changes
    if selected_file and selected_file != st.session_state.current_file:
        with st.spinner("Processing document..."):
            rag_chain, markdown_content = process_selected_file(os.path.join(file_directory, selected_file))
            st.session_state.current_file = selected_file
            st.session_state.rag_chain = rag_chain
    elif not selected_file:
        st.session_state.current_file = None
        st.session_state.rag_chain = None

    with st.form("rag-form"):
        text = st.text_area("Enter your question:")
        submit = st.form_submit_button("Submit")

    if submit and text and st.session_state.rag_chain is not None:
        with st.spinner("Generating response..."):
            response = st.session_state.rag_chain.invoke(text)
            st.session_state['chat_history'].append({"user": text, "rag_response": response})
            
            # Voice output if enabled
            if st.session_state['enable_voice']:
                try:
                    response_engine = pyttsx3.init()
                    response_engine.say(response)
                    response_engine.runAndWait()
                except RuntimeError as e:
                    print(f"Voice error: {e}")
                    if hasattr(response_engine, '_inLoop'):  # pylint: disable=protected-access
                        response_engine.endLoop()
                    response_engine.stop()
                    # Retry with fresh instance
                    response_engine = pyttsx3.init()
                    response_engine.say(response)
                    response_engine.runAndWait()
                finally:
                    response_engine.stop()
                    del response_engine

    # Display chat history
    st.write("## Chat History")
    for chat in reversed(st.session_state['chat_history']):
        st.write(f"**🧑 User**: {chat['user']}")
        st.write(f"**🧠 Assistant**: {chat['rag_response']}")
        st.write("---")

    if selected_file:
        st.write(f"Selected Document: {selected_file}")
