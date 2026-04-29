
"""Similarity page."""


import requests
import streamlit as st

from params import API_URL_BOOKS, API_URL_BOOK_IDS_SIMILAR,API_URL_BOOK
from helpers import book_spinner, show_books_table
# from helpers import (
#     get_book_id_from_query_or_input,
# )


def show_similar_books() -> None:
    """
    Display similar books for a selected book ID.
    """

    col1, col2 = st.columns(2)

    # with col1:
    #     model_name = st.selectbox(
    #         "Choose a model:",
    #         ("knn_tf", "knn_sk", "sota_torch", "sota_tf", "sota_mpnet"),
    #     )
    model_name = "knn_tf"

    # with col2:
    #     query_params = st.query_params
    #     book_id_from_url = query_params.get("book_id", "1")
    #     book_id = st.text_input("Enter book ID:", value=book_id_from_url)

    query_params = st.query_params
    book_id = query_params.get("book_id", "1")

    if not book_id:
        return

    loader = st.empty()

    try:
        # STEP 1: get selected book title
        with loader:
            book_spinner("Retrieving selected book...")

        response = requests.get(
            API_URL_BOOK,
            params={"book_id": book_id},
            timeout=30
        )
        response.raise_for_status()

        book = response.json()
        title = book.get("title", book_id)

        loader.empty()

        st.markdown(
            f"#### 📚 Similar books based on **{title}**",
            unsafe_allow_html=True
        )

        # STEP 2: get similar book IDs
        with loader:
            book_spinner("Finding similar books...")

        response_ids = requests.get(
            API_URL_BOOK_IDS_SIMILAR,
            params={
                "book_id": book_id,
                "model_name": model_name
            },
            headers={"accept": "application/json"},
            timeout=20
        )

        if response_ids.status_code == 404:
            loader.empty()
            st.warning(
                f"(404) No similar books found for {book_id} with model {model_name}."
            )
            return
        elif response_ids.status_code != 200:
            loader.empty()
            st.error("Error while retrieving book IDs.")
            return

        book_ids = response_ids.json()

        if not book_ids:
            loader.empty()
            st.warning(f"No books found for {book_id} with model {model_name}.")
            return

        params = [("book_id_list", bid) for bid in book_ids]

        # STEP 3: get book details
        with loader:
            book_spinner("Retrieving book details...")

        response_books = requests.post(
            API_URL_BOOKS,
            json={"book_id_list": book_ids}
        )

        loader.empty()

        if response_books.status_code == 404:
            st.warning(f"No books details found for {book_id}.")
        elif response_books.status_code != 200:
            st.error("Error while retrieving books details.")
        else:
            show_books_table(response_books.json())

    except requests.RequestException as exc:
        loader.empty()
        st.error(f"Request error: {exc}")
