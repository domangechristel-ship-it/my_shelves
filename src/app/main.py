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
from show_pages import show_find_book, show_map, show_books_by_country
from show_features import show_book_by_filters
from similarity_page import show_similar_books
from show_chatbot import show_chatbot

st.set_page_config(
    page_title="Book Shelves",
    page_icon="📚",
    layout="wide",
)

# --------------------------------------------------
# Custom CSS
# --------------------------------------------------
st.markdown(
    """
    <style>
        /* ============================= */
        /* GLOBAL TYPO */
        /* ============================= */
        .main-title {
            font-size: 42px;
            font-weight: 900;
            margin-bottom: 0px;
            color: #0f4c5c;
        }

        .subtitle {
            font-size: 18px;
            color: #5f6f73;
            margin-top: 0px;
            margin-bottom: 25px;
        }

        /* ============================= */
        /* HERO BOX (TOP BLOCK) */
        /* ============================= */
        .hero-box {
            padding: 28px 32px;
            border-radius: 18px;
            background: linear-gradient(135deg, #e6f4f1 0%, #f4fbfa 100%);
            border: 1px solid #d1e7e4;
            margin-bottom: 25px;
            box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        }

        /* ============================= */
        /* SECTIONS */
        /* ============================= */
        .section-title {
            font-size: 24px;
            font-weight: 800;
            margin-bottom: 4px;
            color: #0f4c5c;
        }

        .section-caption {
            color: #6b7c80;
            font-size: 15px;
            margin-bottom: 20px;
        }

        /* ============================= */
        /* TABS */
        /* ============================= */

        /* Remove default red line */
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            background-color: transparent !important;
        }

        /* Base tab style */
        div[data-testid="stTabs"] button {
            font-size: 16px;
            font-weight: 600;
            color: #5f6f73;
            padding: 10px 18px;
            border: none;
            border-bottom: 2px solid transparent;
            transition: all 0.25s ease;
            border-radius: 8px 8px 0 0;
        }

        /* Hover */
        div[data-testid="stTabs"] button:hover {
            background-color: #e6f4f1;
            color: #0f4c5c;
        }

        /* Active tab */
        div[data-testid="stTabs"] button[aria-selected="true"] {
            color: #0f4c5c;
            border-bottom: 3px solid #2a9d8f;
            background: rgba(42,157,143,0.08);
        }

    </style>
    """,
    unsafe_allow_html=True,
)
# --------------------------------------------------
# Session state
# --------------------------------------------------
if "country_page" not in st.session_state:
    st.session_state.country_page = "Map"

if "selected_country" not in st.session_state:
    st.session_state.selected_country = None

# --------------------------------------------------
# Header
# --------------------------------------------------
st.markdown(
    """
    <div style="
        padding: 20px 30px;
        border-radius: 20px;
        background: linear-gradient(135deg, #0f4c5c 0%, #1a6f7a 100%);
        color: white;
        margin-bottom: 20px;
        text-align: center;
        box-shadow: 0 8px 20px rgba(0,0,0,0.2);
    ">
        <div style="font-size:38px; font-weight:900; margin-bottom:5px;">
            <span style="color:#ffd60a;">My Shelves</span>
        </div>
        <div style="font-size:16px; opacity:0.9;">
            Your intelligent book discovery platform
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
# --------------------------------------------------
# Tabs
# --------------------------------------------------
tab_find_book, tab_country, tab_features, tab_similar, tab_chatbot = st.tabs(
    [
        "🔎 Find book",
        "🌍 Country",
        "🎛️ Features",
        "📚 Similar books",
        "💬 Chatbot",
    ]
)

# --------------------------------------------------
# TAB 1 - FIND BOOK
# --------------------------------------------------
with tab_find_book:
    st.markdown('<div class="section-title">🔎 Find a book</div>', unsafe_allow_html=True)

    show_find_book()

# --------------------------------------------------
# TAB 2 - COUNTRY
# --------------------------------------------------
with tab_country:
    st.markdown('<div class="section-title">🌍 Explore books by country</div>', unsafe_allow_html=True)

    if st.session_state.country_page == "Map":
        show_map()
    elif st.session_state.country_page == "Books":
        show_books_by_country()

# --------------------------------------------------
# TAB 3 - FEATURES
# --------------------------------------------------
with tab_features:
    st.markdown('<div class="section-title">🎛️ Filter books by features</div>', unsafe_allow_html=True)

    show_book_by_filters()

# --------------------------------------------------
# TAB 4 - SIMILAR BOOKS
# --------------------------------------------------
with tab_similar:
    st.markdown('<div class="section-title">📚 Similar books</div>', unsafe_allow_html=True)

    show_similar_books()

# --------------------------------------------------
# TAB 5 - CHATBOT
# --------------------------------------------------
with tab_chatbot:
    st.markdown('<div class="section-title">💬 Describe the book you are looking for</div>', unsafe_allow_html=True)

    show_chatbot()
