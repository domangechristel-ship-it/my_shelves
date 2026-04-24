
import requests
import streamlit as st

from params import API_URL_BOOKS, API_URL_BOOK_IDS_SIMILAR
from show_pages import show_books_table
from helpers import (
    get_book_id_from_query_or_input,
)


def show_similar_books() -> None:
    """
    Display the books list page for the currently selected country.
    """

    st.subheader("📚 Books list")
    # book_id = get_book_id_from_query_or_input()
    book_id = st.text_input("Enter book ID:", "220708")
    if not book_id:
        return

    if book_id is None:
        st.warning("Choose a book ID first.")

    else:
        st.write(f"Books for **{book_id}**:")

        try:
            response_ids = requests.get(
                API_URL_BOOK_IDS_SIMILAR,
                params={"book_id": book_id},
                timeout=10
            )

            if response_ids.status_code == 404:
                st.warning(f"(404) No similar books found for {book_id}.")
            elif response_ids.status_code != 200:
                st.error("Error while retrieving book IDs.")
            else:
                book_ids = response_ids.json()
                st.write(book_ids)
                if not book_ids:
                    st.warning(f"No books found for {book_id}.")
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
