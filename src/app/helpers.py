"""
Helper functions for building the book details page in the Streamlit app.

This module provides reusable utilities to support the
`show_book_details()` workflow, including:

- Retrieving the book ID from URL query parameters or user input
- Fetching book data from the backend API
- Normalizing and cleaning API response data for display
- Rendering UI components such as metric boxes and book detail sections

Purpose
-------
To separate data retrieval, processing, and UI rendering logic from the main
Streamlit page, improving code readability, maintainability, and testability.

Typical Flow
------------
1. Get book ID (URL or input)
2. Call API to retrieve book details
3. Normalize/clean the data
4. Render the book detail UI

Usage
-----
Imported and used by `show_pages.py`:

    import helpers

    book = helpers.fetch_book_details(book_id)
    helpers.render_book_details(book)

Notes
-----
- All functions are designed to be lightweight and reusable
- UI functions rely on Streamlit (`st`)
- API functions handle errors gracefully and return `None` when needed
"""
import requests
import streamlit as st
from params import  (
    API_URL_BOOK,
    API_URL_COUNTRY,
    API_URL_BOOK_IDS_BY_COUNTRY,
    API_URL_SIMILAR_BOOKS,
    API_URL_READ_TITLE
    )

BOOK_DETAILS_CSS = """
<style>
.book-card {
    background-color: #ffffff;
    padding: 2rem;
    border-radius: 18px;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
    margin-bottom: 1.5rem;
}
.book-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
    color: #1f1f1f;
}
.book-subtitle {
    font-size: 0.95rem;
    color: #6c757d;
    margin-bottom: 1rem;
}
.metric-box {
    background-color: #f8f9fa;
    padding: 0.8rem 1rem;
    border-radius: 12px;
    text-align: center;
    border: 1px solid #e9ecef;
}
.metric-label {
    font-size: 0.85rem;
    color: #6c757d;
}
.metric-value {
    font-size: 1.1rem;
    font-weight: 600;
    color: #212529;
}
.description-box {
    background-color: #fcfcfc;
    padding: 1rem;
    border-radius: 12px;
    border: 1px solid #eeeeee;
    line-height: 1.6;
    color: #333333;
}
</style>
"""

def clean_numeric_display(value: str | int | float | None) -> str:
    """Format numeric-like values for display."""
    if value in (None, ""):
        return ""
    return str(value).replace(".0", "")


def get_book_id_from_query_or_input() -> str:
    """Get the book id from URL query params or from a text input."""
    book_id = st.query_params.get("book_id")

    if isinstance(book_id, list):
        book_id = book_id[0]

    if not book_id:
        book_id = st.text_input("Enter book ID:")

    return str(book_id).strip() if book_id else ""

def get_search_value_from_query_or_input() -> str:
    """Get search value from query params or persistent input."""

    # 1. Initialize session_state if not exists
    if "search_value" not in st.session_state:
        query_value = st.query_params.get("book_id")

        if isinstance(query_value, list):
            query_value = query_value[0]

        st.session_state.search_value = query_value or ""

    # 2. Text input bound to session_state
    search_value = st.text_input(
        " ",
        key="search_value",
        placeholder="Example: 1885731 or Harry Potter",
    )

    return search_value.strip()

def is_book_id(search_value: str) -> bool:
    """Return True if the search value is an integer book_id."""
    return search_value.strip().isdigit()

def fetch_book_details(book_id: str) -> dict | None:
    """
    Fetch book details from the API.

    Returns
    -------
    dict | None
        Book JSON if found, otherwise None.
    """
    try:
        response = get_book_by_id(book_id)

        if response.status_code == 404:
            st.warning("📕Book not found.")
            return None

        if response.status_code != 200:
            st.error(f"⚠️ Error API (code {response.status_code})")
            return None

        if not response.text.strip():
            st.warning("📕Book not found.")
            return None

        return response.json()

    except requests.exceptions.JSONDecodeError:
        st.warning("📕Book not found.")
        return None
    except requests.exceptions.RequestException as exc:
        st.error(f"🚨 Error connexion API : {exc}")
        return None


def normalize_book_data(response_json: dict) -> dict:
    """Extract and clean fields used in the UI."""
    return {
        "title": response_json.get("title", "Unknown title"),
        "book_id": response_json.get("book_id", ""),
        "description": response_json.get("description", "No description available."),
        "publication_year": clean_numeric_display(response_json.get("publication_year", "")),
        "image_url": response_json.get("image_url", ""),
        "goodreads_url": response_json.get("url", ""),
        "average_rating": response_json.get("average_rating", ""),
        "num_pages": clean_numeric_display(response_json.get("num_pages", "")),
        "series": response_json.get("series", ""),
    }


def render_metric_box(label: str, value: str) -> None:
    """Render one metric box."""
    st.markdown(
        f"""
        <div class="metric-box">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_book_details(book: dict) -> None:
    """Render the book details page."""
    st.markdown(BOOK_DETAILS_CSS, unsafe_allow_html=True)

    st.markdown("#### 📗 Book Details")

    col1, col2 = st.columns([1, 2])

    with col1:
        if book["image_url"]:
            st.markdown(
                f"""
                <div style="text-align:center;">
                    <img src="{book['image_url']}" style="max-height:400px; width:auto;">
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.write("No cover available")

    with col2:
        st.markdown(f'<div class="book-title">{book["title"]}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="book-subtitle">Book ID: {book["book_id"]}</div>',
            unsafe_allow_html=True
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            render_metric_box("⭐ Rating", str(book["average_rating"]))
        with m2:
            render_metric_box("📄 Pages", str(book["num_pages"]))
        with m3:
            render_metric_box("📅 Year", str(book["publication_year"]))

        if book["series"]:
            st.markdown(f'**Series**: {book["series"]}')

        if book["goodreads_url"]:
            st.link_button("🔗 Open on Goodreads", book["goodreads_url"])

    st.markdown("### Description")
    st.markdown(
        f'<div class="description-box">{book["description"]}</div>',
        unsafe_allow_html=True
    )

@st.cache_data(ttl=3600)
def get_country_data():
    response = requests.get(API_URL_COUNTRY, timeout=30)
    response.raise_for_status()
    return response.json()

@st.cache_data(ttl=3600)
def get_by_country(selected_country):
    response_ids = requests.get(
                API_URL_BOOK_IDS_BY_COUNTRY,
                params={"country": selected_country}
                # ,timeout=30
            )
    response_ids.raise_for_status()
    return response_ids

@st.cache_data(ttl=3600)
def get_chat_respons(query,top_k):
    response_search = requests.get(
                    API_URL_SIMILAR_BOOKS,
                    params={
                        "query": query,
                        "top_k": top_k,
                    }
                    # ,timeout=30,
                )
    return response_search

@st.cache_data(ttl=3600)
def get_books_by_title(title):
    response = requests.get(
            API_URL_READ_TITLE,
            params={"title": title}
            # ,timeout=10
        )
    return response

@st.cache_data(ttl=3600)
def get_book_by_id(book_id):
    response = requests.get(
            API_URL_BOOK,
            params={"book_id": book_id}
            # ,timeout=30
        )
    return response
