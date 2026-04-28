import requests
import streamlit as st

from show_pages import show_books_table
from params import API_URL_BOOKS
from helpers import get_chat_respons


def show_chatbot() -> None:
    """
    Display the semantic book search chatbot page.

    The user enters a natural language query, the app calls the API to search
    similar books, retrieves the matching book details, and displays them.
    """

    query = st.text_area(
        " ",
        placeholder="A fantasy story with dragons and adventure...",
        key="chatbot_query",
        height=150
    )

    top_k = st.slider(
        "Number of books",
        min_value=1,
        max_value=20,
        value=3,
        key="chatbot_top_k",
    )

    if st.button("🔎 Search", key="chatbot_search_button"):
        if not query.strip():
            st.warning("Please enter a search query.")
            return

        try:
            with st.spinner("Searching similar books..."):
                response_search = get_chat_respons(query,top_k)

            if response_search.status_code != 200:
                st.error("Error while searching similar books.")
                return

            search_json = response_search.json()
            book_ids = search_json.get("book_ids", [])

            if not book_ids:
                st.warning("No similar books found.")
                return

            params = [("book_id_list", book_id) for book_id in book_ids]

            with st.spinner("Retrieving book details..."):
                response_books = requests.get(
                    API_URL_BOOKS,
                    params=params
                    # ,timeout=20
                )

            if response_books.status_code == 200:
                st.subheader("Results")
                show_books_table(response_books.json())
            else:
                st.error(f"Error while retrieving books.{response_books.status_code}")
                st.warning(params)
        except requests.exceptions.RequestException as exc:
            st.error(f"API request failed: {exc}")
