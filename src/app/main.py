"""
Main entry point for the Streamlit Books Explorer application.

This module defines the overall layout and navigation of the app, including:
- Tab-based interface (e.g., country view, book details)
- Session state management for page navigation and selected country
- Integration with UI components from `show_pages.py`

Features
--------
- Interactive world map displaying number of books per country
- Clickable countries to explore associated books
- Dynamic navigation between map and book list views
- Support for URL query parameters (deep linking)

Architecture
------------
- `show_pages.py` contains reusable UI rendering functions
- This module orchestrates page routing and user interaction flow

Usage
-----
Run the app with:

    streamlit run main.py

Requirements
------------
- Streamlit
- Requests
- Pandas
- Folium (via streamlit-folium)

"""
import streamlit as st
from show_pages import show_book_details, show_map, show_books_by_country
from similarity_page import show_similar_books
from show_chatbot import show_chatbot


st.set_page_config(page_title="Books App", layout="wide")

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "country_page" not in st.session_state:
    st.session_state.country_page = "Map"

if "selected_country" not in st.session_state:
    st.session_state.selected_country = None

# --------------------------------------------------
# Main title
# --------------------------------------------------
st.title("📚 Book Shelves")

tab_find_book, tab_country, tab_similar, tab_chatbot = st.tabs(["Find book",
                                                   "Country",
                                                   "Similar books",
                                                   "Chatbot"])

# ==================================================
# TAB 1 - FIND BOOK
# ==================================================
with tab_find_book:
    st.subheader("🔎 find book by Id")
    show_book_details()

# ==================================================
# TAB 2 - COUNTRY
# ==================================================
with tab_country:

    # ----------------------------------------------
    # Page 1 : Map
    # ----------------------------------------------
    if st.session_state.country_page == "Map":
        show_map()

    # # ----------------------------------------------
    # # Page 2 : Books
    # # ----------------------------------------------
    elif st.session_state.country_page == "Books":
        show_books_by_country()


# ==================================================
# TAB 3 - SIMILAR
# ==================================================
with tab_similar:

    st.subheader("🔎 find a similar book by Id")
    show_similar_books()

# ==================================================
# TAB 4 - CHATBOT
# ==================================================
with tab_chatbot:
    show_chatbot()
