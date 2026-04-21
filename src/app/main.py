"""
This is a demo Streamlit app.
It shows how to use Streamlit to create a simple web app that interacts with an API.
"""
import requests

import streamlit as st


API_URL = 'https://my-shelves-image-151819310613.europe-west1.run.app/read'


st.title("Hello World! 👋🌎")
st.markdown(
   """
   This is a demo Streamlit app.
   Enter your name in the text box below and press a button to see some fun features in Streamlit.
   """
)


name = st.text_input("Enter your name:")

# Use columns to create buttons side by side
col1, col2 = st.columns(2)

with col1:
    if st.button("Send balloons! 🎈"):
        st.balloons()
        st.write(f"Time to celebrate {name}! 🥳")
        st.write("You deployed a Streamlit app! 👏")

with col2:
    if st.button("Send snow! ❄️"):
        st.snow()
        st.write(f"Let it snow {name}! 🌨️")
        st.write("You deployed a Streamlit app! 👏")


book_id = st.text_input("Enter book ID:")
params = {
        "book_id": book_id
    }

response = requests.get(API_URL, params=params, timeout=10)
response_json = response.json()
st.markdown(response_json['description'])
