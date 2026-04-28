
"""Similarity page."""


import requests
import streamlit as st

from params import API_URL_BOOKS, API_URL_BOOK_IDS_SIMILAR
from show_pages import show_books_table
# from helpers import (
#     get_book_id_from_query_or_input,
# )


def show_similar_books() -> None:
    """
    Display the books list page for the currently selected country.
    """
    # --------------------------------------------------
    # Charge uniqument si tab actif
    # --------------------------------------------------
    if "tab_similar" not in st.session_state:
        st.session_state.tab_country_loaded = True
    else:
        return

    # st.subheader("📚 Books list")

    col1, col2 = st.columns(2)

    with col1:
        model_name = st.selectbox(
            "Choose a model:",
            ("knn_tf", "knn_sk", "sota_torch", "sota_tf", "sota_mpnet"),
        )

        # st.write("You selected:", model_name)
    # with col2:
    #     training_dataset = st.selectbox(
    #         "Choose a training dataset:",
    #         ("10k", "20k", "50k", "100k", "150k", "200k", "all"),
    #     )

        # st.write("You selected:", training_dataset)

    with col2:
        # book_id = get_book_id_from_query_or_input()
        query_params = st.query_params
        book_id_from_url = query_params.get("book_id", "1")
        book_id = st.text_input("Enter book ID:", value=book_id_from_url)


    if not book_id:
        return

    if book_id is None:
        st.warning("Choose a book ID first.")

    else:
        st.write(f"Books for **{book_id}**:")

        try:
            response_ids = requests.get(
                API_URL_BOOK_IDS_SIMILAR,
                params={"book_id": book_id,
                        "model_name": model_name},
                headers={"accept": "application/json"},
                timeout=10
            )

            if response_ids.status_code == 404:
                st.warning(f"(404) No similar books found for {book_id} with model {model_name}.")
            elif response_ids.status_code != 200:
                st.error("Error while retrieving book IDs.")
            else:
                book_ids = response_ids.json()
                # st.write(book_ids)
                if not book_ids:
                    st.warning(f"No books found for {book_id} with model {model_name}.")
                else:
                    params = [("book_id_list", book_id) for book_id in book_ids]

                    response_books = requests.get(
                        API_URL_BOOKS,
                        params=params,
                        timeout=20
                    )

                    if response_books.status_code == 404:
                        st.warning(f"No books details found for {book_id}.")
                    elif response_books.status_code != 200:
                        st.error("Error while retrieving books details.")
                    else:
                        response_books_json = response_books.json()
                        show_books_table(response_books_json)

        except requests.RequestException as exc:
            st.error(f"Request error: {exc}")
