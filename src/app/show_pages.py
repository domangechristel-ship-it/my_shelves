"""
Streamlit Pages Module - My Shelves 📚

This module centralizes the definition of all page-related functions used
in the Streamlit application. Each function corresponds to a specific page
or view in the app and is responsible for rendering its UI components and
handling user interactions.

Purpose
-------
- Organize the application into reusable and modular page functions
- Separate UI logic from the main application entry point (`main.py`)
- Improve readability, maintainability, and scalability of the project

Structure
---------
Each page is implemented as a function, typically following this pattern:

    def show_<page_name>(...):
        \"\"\"Render the <page_name> page.\"\"\"
        # Streamlit UI elements
        # Data retrieval (API, BigQuery, etc.)
        # Display results

Examples of pages
----------------
- show_book_details(book_data) :
    Display detailed information about a selected book
- show_books_table(book_list,nbr_rows):
    Display a table of books
- show_map():
    Display a map

Dependencies
------------
- streamlit
- requests (for API calls)
- Custom utility modules (data access, formatting, etc.)

Notes
-----
- This module should not contain application entry logic (no `st.run()` or routing).
- Navigation between pages should be handled in `main.py`.
- Keep each page function focused on UI rendering and delegate heavy
  processing to dedicated service or utility modules when possible.
"""
import ast
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests
from helpers import (
    get_book_id_from_query_or_input,
    fetch_book_details,
    normalize_book_data,
    render_book_details,
    is_book_id,
    get_search_value_from_query_or_input,
    get_country_data,
    get_by_country,
    get_books_by_title,
    book_spinner
)
from params import API_URL_BOOKS, API_URL_READ_TITLE

def show_books_table(response_json: dict | list[dict]) -> None:
    """
    Display a books table with clickable cover images.

    When clicking the image, it updates the URL with ?book_id=...
    """

    if not response_json:
        st.info("No books to display.")
        return

    # Ensure list
    data = [response_json] if isinstance(response_json, dict) else response_json
    df = pd.DataFrame(data)

    # Keep only needed columns
    df = df[["image_url", "book_id", "title", "average_rating"]]
    st.markdown(
        """
        <div style="
            display: grid;
            grid-template-columns: 1fr 1fr 4fr 1fr;
            align-items: center;
            padding: 12px 18px;
            border-radius: 14px;
            background: #f8fafc;
            border: 1px solid #dbeafe;
            margin-bottom: 12px;
            color: #1e3a8a;
            font-weight: 700;
            font-size: 15px;
        ">
            <div>📖 Cover</div>
            <div>🆔 Book ID</div>
            <div>📘 Title</div>
            <div>⭐ Rating</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Rows
    for _, row in df.iterrows():
        col1, col2, col3, col4 = st.columns([1, 1, 4, 1])

        book_id = row["book_id"]
        link = f"?book_id={book_id}"

        col1.markdown(
            f"""
            <a href="{link}">
                <img src="{row['image_url']}" style="height:90px;border-radius:6px;">
            </a>
            """,
            unsafe_allow_html=True
        )

        col2.write(book_id)
        col3.write(row["title"])
        col4.write(f"⭐ {row['average_rating']}")

        # 👉 subtle line between rows
        st.markdown(
            "<hr style='margin:10px 0; border: none; border-top: 1px solid #dbeafe;'>",
            unsafe_allow_html=True
        )

def show_map() -> None:
    """
    Display an interactive world map with book counts per country.

    Features
    --------
    - Color-coded markers based on number of books (binned)
    - Click on a country to:
        • store it in session_state
        • navigate to "Books" page
        • update URL query params
    """

    # --------------------------------------------------
    # Charge uniqument si tab actif
    # --------------------------------------------------
    if "tab_country" not in st.session_state:
        st.session_state.tab_country_loaded = True
    else:
        return
    # --------------------------------------------------
    # Data
    # --------------------------------------------------

    books_count_country = pd.DataFrame(get_country_data())
    books_count_country["latlng"] = books_count_country["capital_latlng"].apply(ast.literal_eval)
    books_count_country = books_count_country[["country", "latlng", "count_books"]].copy()
    books_count_country = books_count_country[
        books_count_country["latlng"].notna() &
        (books_count_country["latlng"].apply(len) == 2)
    ]

    # -----------------------------
    # Binning
    # -----------------------------
    bins = [0, 5, 10, 50, 100, 500, 1000, 100000]
    labels = ["1-5", "6-10", "11-50", "51-100", "101-500", "501-1000", "1001+"]

    df = books_count_country.copy()

    df["count_bin"] = pd.cut(
        df["count_books"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    # -----------------------------
    # Colors (cool-warm)
    # -----------------------------
    bin_colors = {
        "1-5": "#2c7bb6",
        "6-10": "#66c2a5",
        "11-50": "#abdda4",
        "51-100": "#ffffbf",
        "101-500": "#fdae61",
        "501-1000": "#f46d43",
        "1001+": "#d73027"
    }

    # -----------------------------
    # Map
    # -----------------------------
    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    for _, row in df.iterrows():
        color = bin_colors.get(str(row["count_bin"]), "#cccccc")

        folium.CircleMarker(
            location=row["latlng"],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{row['country']}: {row['count_books']} books",
            tooltip=f"{row['country']} ({row['count_books']} books)"
        ).add_to(m)

    map_data = st_folium(
        m,
        width=1200,
        height=500,
        returned_objects=["last_object_clicked_popup"]
    )

    # -----------------------------
    # Handle click
    # -----------------------------
    clicked_popup = map_data.get("last_object_clicked_popup")

    if clicked_popup:
        country_clicked = clicked_popup.split(":")[0].strip()

        st.session_state.selected_country = country_clicked
        st.session_state.country_page = "Books"

        # update URL
        st.query_params["country_page"] = "Books"
        st.query_params["country"] = country_clicked

        st.success(f"Country selected: {country_clicked}")
        st.rerun()

    # -----------------------------
    # Info
    # -----------------------------
    if st.session_state.get("selected_country"):
        st.info(f"Current selected country: {st.session_state.selected_country}")

def show_books_by_country() -> None:
    """
    Display the books list page for the currently selected country.
    """

    if st.button("⬅ Back to map"):
        st.session_state.country_page = "Map"
        st.session_state.selected_country = None
        st.query_params["country_page"] = "Map"
        st.query_params.pop("country", None)
        st.rerun()

    selected_country = st.session_state.get("selected_country")

    if selected_country is None:
        st.warning("No country selected yet. Go to the map and click on a marker.")
        return

    st.write(f"Books for **{selected_country}**:")

    loader = st.empty()

    try:
        # ---------------------------
        # STEP 1: Get book IDs
        # ---------------------------
        with loader:
            book_spinner("Finding books for this country...")

        response_ids = get_by_country(selected_country)

        if response_ids.status_code == 404:
            loader.empty()
            st.warning(f"No books found for {selected_country}.")
            return
        elif response_ids.status_code != 200:
            loader.empty()
            st.error(f"Error while retrieving book IDs {response_ids.status_code}.")
            return

        book_ids = response_ids.json()

        if not book_ids:
            loader.empty()
            st.warning(f"No books found for {selected_country}.")
            return

        params = [("book_id_list", book_id) for book_id in book_ids]

        # ---------------------------
        # STEP 2: Get book details
        # ---------------------------
        with loader:
            book_spinner("Retrieving book details...")

        response_books = requests.get(
            API_URL_BOOKS,
            params=params
        )

        # 👉 remove loader
        loader.empty()

        if response_books.status_code == 404:
            st.warning(f"No books details found for {selected_country}.")
        elif response_books.status_code != 200:
            st.error("Error while retrieving books details.")
        else:
            show_books_table(response_books.json())

    except requests.RequestException as exc:
        loader.empty()
        st.error(f"Request error: {exc}")

def show_book_details() -> None:
    """Get a book id, fetch data from the API, and display the result."""
    book_id = get_book_id_from_query_or_input()
    if not book_id:
        return

    response_json = fetch_book_details(book_id)
    if response_json is None:
        return

    book = normalize_book_data(response_json)
    render_book_details(book)

def show_books_by_title(title: str) -> list[dict] | None:
    """Fetch books matching a title from the API."""

    try:
        response = get_books_by_title(title)

        if response.status_code == 404:
            st.warning("📕 No books found for this title.")
            return None

        if response.status_code != 200:
            st.error("⚠️ Error while searching books by title.")
            return None

        return response.json()

    except requests.RequestException as e:
        st.error(f"API error: {e}")
        return None

def show_find_book() -> None:
    """
    Search a book by ID or by title.

    - If the input is only digits: fetch one book by book_id
    - Otherwise: search books by title and show a table
    """

    search_value = get_search_value_from_query_or_input()

    if not search_value:
        return

    loader = st.empty()

    try:
        if is_book_id(search_value):
            with loader:
                book_spinner("Retrieving book details...")

            response_json = fetch_book_details(search_value)

            loader.empty()

            if response_json is None:
                return

            book = normalize_book_data(response_json)
            render_book_details(book)

        else:
            with loader:
                book_spinner("Searching books by title...")

            response_json = show_books_by_title(search_value)

            loader.empty()

            if response_json is None:
                return

            show_books_table(response_json)

    except Exception as exc:
        loader.empty()
        st.error(f"Error while searching book: {exc}")
