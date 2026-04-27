import requests
import streamlit as st

from show_pages import show_books_table
from vector_search import search_similar_books
from params import API_URL_BOOKS


def show_chatbot() -> None:
    """
    Display the semantic book search chatbot page.

    The user enters a natural language query, the app searches for similar books
    using the vector search endpoint, retrieves the matching book details from
    the API, and displays them in a table.
    """

    st.title("🤖 Book Search")

    query = st.text_input(
        "Search for a book",
        placeholder="A fantasy story with dragons and adventure...",
        key="chatbot_query",
    )

    top_k = st.slider(
        "Number of results",
        min_value=1,
        max_value=20,
        value=5,
        key="chatbot_top_k",
    )

    if st.button("Search", key="chatbot_search_button"):
        if not query.strip():
            st.warning("Please enter a search query.")
            return

        with st.spinner("Searching similar books..."):
            results = search_similar_books(query, top_k=top_k)

        if not results or not results[0]:
            st.warning("No similar books found.")
            return

        neighbors = results[0]
        book_ids = [neighbor.id for neighbor in neighbors]

        params = [("book_id_list", book_id) for book_id in book_ids]

        try:
            response_books = requests.get(
                API_URL_BOOKS,
                params=params,
                timeout=20,
            )

            if response_books.status_code == 200:
                st.subheader("Results")
                show_books_table(response_books.json())
            else:
                st.error("Error while retrieving books.")

        except requests.exceptions.RequestException as exc:
            st.error(f"API request failed: {exc}")
