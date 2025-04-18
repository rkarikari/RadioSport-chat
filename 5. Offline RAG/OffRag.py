

import os
import tempfile
import streamlit as st
from embedchain import App
import base64
from streamlit_chat import message


#configure embedchain App- we are using ollama specifically llama3.2
def embedchain_bot(db_path):
    return App.from_config(
        config = {
            "llm" : {"provider":"ollama", "config" : {"model":"deepseek-r1:1.5b", "max_tokens": 250,
                                                      "temperature":0.5, "stream": True, "base_url": 'http://localhost:11434'}},
            "vectordb": {"provider": "chroma", "config":{"dir": db_path}},
            "embedder": {"provider": "ollama", "config": {"model": "all-minilm:latest", "base_url": 'http://localhost:11434'}}
        }
    )


# add function to display PDF
def display_pdf(pdf_file):
    base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
    pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600px" frameborder="0" allowfullscreen></iframe>'
    st.markdown(pdf_display, unsafe_allow_html=True)



#streamlit Title and Prep.
st.set_page_config(page_title="Chat with your PDF", page_icon=":books:", layout="wide")
st.title("Chat with your PDF's")
st.caption("Upload a PDF and ask questions about it.")

db_path = tempfile.mkdtemp()    #db to store pdf temporarily

if "app" not in st.session_state:
    st.session_state.app = embedchain_bot(db_path)
if "messages" not in st.session_state:
    st.session_state.messages = [] 


#sidebar for pdf upload
with st.sidebar:
    st.header("PDF Upload")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"])

    if pdf_file:
        st.subheader("Preview PDF")
        display_pdf(pdf_file)
         # add PDF to knowledge base
        if st.button("Submit PDF"):
            with st.spinner("Adding PDF to knowledge base..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                    f.write(pdf_file.getvalue())
                    st.session_state.app.add(f.name, data_type="pdf_file")
                os.remove(f.name)
            st.success(f"Added {pdf_file.name} to knowledge base!")




# set up chat interface
for i,msg in enumerate(st.session_state.messages):
    message(msg["content"], is_user=msg["role"] == "user",key=str(i))

if prompt := st.chat_input("Ask a question about the PDF"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    message(prompt, is_user=True)


#user query and display response
    with st.spinner("Thinking..."):
        response = st.session_state.app.chat(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        message(response)

if st.button("Clear Chat"):
    st.session_state.messages = []