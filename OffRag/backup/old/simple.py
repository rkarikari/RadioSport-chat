import streamlit as st

st.title("Local Chat App")

# Initialize session state for storing messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Input text area for the user
text_input = st.text_area(label='Type your message here:', height=250)

# Send button logic
if st.button('Send') and text_input:
    # Append the new message to the session state
    st.session_state.messages.append({'text': text_input})
    # Clear the text input
    text_input = ""

# Display chat history
if st.session_state.messages:
    st.write("Received:")
    for message in st.session_state.messages:
        st.write("- " + message['text'])


    