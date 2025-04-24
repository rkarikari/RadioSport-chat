import streamlit as st
from streamlit import caching


st.title("Local Chat App")
text_input = st.text_area(label='Type your message here:', height=250, value='')
send_button = st.button('Send')

if send_button:
    # Store chat history (optional)
    @st.cache
    def store_message(message):
        return messages

    messages = []  # Initialize an empty list to hold messages
    messages = store_message(text_input)
    
    text_input.clear()

if len(messages):
    st.write("Received:")
    for message in messages:
        st.write("- " + message['text'])

if send_button:

    new_message = {'text': text_input}  # Create a dictionary for the new message
    messages.append(new_message)
    st.write("- " + new_message['text'])
    text_input.clear()

    