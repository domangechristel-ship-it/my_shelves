"""
Streamlit Application - My Shelves 📚

This module defines the main entry point of the Streamlit web application
for the "My Shelves" project. The app allows users to search and explore
book information retrieved from a backend API connected to a BigQuery dataset.

Behavior
--------
- If a `book_id` is provided in the URL query parameters, the app
  automatically fetches and displays the corresponding book details.
- If no `book_id` is provided, the user is prompted to enter one.
- Once a valid `book_id` is available, a request is sent to the API
  and the response is rendered in the UI.

Dependencies
------------
- streamlit
- requests

Environment Variables
---------------------
- API_URL : str
    Base URL of the backend API used to fetch book data.

Example
-------
Access with query parameter:
    ?book_id=22077083

Notes
-----
This module focuses on UI logic and API interaction. Data processing
and formatting should be handled in dedicated utility modules when possible.
"""
import requests

import streamlit as st
from book_detail import show_book_details

API_URL = 'https://my-shelves-image-151819310613.europe-west1.run.app/read'
# API_URL = 'http://127.0.0.1:8000/read'

query_params = st.query_params
book_id = query_params.get("book_id")

# Si book_id dans l'URL → pas d'input
if not book_id:
    # Sinon → input utilisateur
    book_id = st.text_input("Enter book ID:")

# Si on a un book_id (URL ou input)
if book_id:
    params = {"book_id": book_id}

    response = requests.get(API_URL, params=params, timeout=10)
    response_json = response.json()

    show_book_details(response_json)
