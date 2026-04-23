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
import streamlit as st
import pandas as pd

def show_book_details(response_json: dict) -> None:
    """
    Display a book detail page from a JSON response.

    Parameters
    ----------
    response_json : dict
        Dictionary containing book information.

    Features
    --------
    - Retrieve a book by its `book_id` via:
        • URL query parameter (e.g., ?book_id=22077083)
        • Manual input through a text field
    - Display detailed book information (title, description, cover, metadata)
    - Seamless integration with an external API endpoint
    - Dynamic UI behavior depending on user input or URL state

    """

    # Récupération sécurisée des champs
    title = response_json.get("title", "Unknown title")
    book_id = response_json.get("book_id", "")
    description = response_json.get("description", "No description available.")
    publication_year = response_json.get("publication_year", "")
    image_url = response_json.get("image_url", "")
    goodreads_url = response_json.get("url", "")
    average_rating = response_json.get("average_rating", "")
    num_pages = response_json.get("num_pages", "")
    series = response_json.get("series", "")

    # Petit nettoyage des valeurs
    if publication_year:
        publication_year = str(publication_year).replace(".0", "")
    if num_pages:
        num_pages = str(num_pages).replace(".0", "")

    st.markdown(
        """
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
        """,
        unsafe_allow_html=True
    )


    st.title("📚 Book Details")

    col1, col2 = st.columns([1, 2])

    with col1:
        if image_url:
            st.image(image_url, use_container_width=True)
        else:
            st.write("No cover available")

    with col2:
        st.markdown(f'<div class="book-title">{title}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="book-subtitle">Book ID: {book_id}</div>', unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)

        with m1:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">⭐ Rating</div>
                    <div class="metric-value">{average_rating}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m2:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">📄 Pages</div>
                    <div class="metric-value">{num_pages}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with m3:
            st.markdown(
                f"""
                <div class="metric-box">
                    <div class="metric-label">📅 Year</div>
                    <div class="metric-value">{publication_year}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        if series:
            st.markdown(f"**Series**: {series}")

        if goodreads_url:
            st.link_button("🔗 Open on Goodreads", goodreads_url)

    st.markdown("### Description")
    st.markdown(f'<div class="description-box">{description}</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

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

    # Header
    col1, col2, col3, col4 = st.columns([1, 1, 4, 1])
    col1.markdown("**Cover**")
    col2.markdown("**Book ID**")
    col3.markdown("**Title**")
    col4.markdown("**Rating**")

    st.markdown("---")

    # Rows
    for _, row in df.iterrows():
        col1, col2, col3, col4 = st.columns([1, 1, 4, 1])

        book_id = row["book_id"]

        # 🔗 clickable image → updates URL
        link = f"?book_id={book_id}"

        col1.markdown(
            f"""
            <a href="{link}">
                <img src="{row['image_url']}" style="height:100px;border-radius:5px;">
            </a>
            """,
            unsafe_allow_html=True
        )

        col2.write(book_id)
        col3.write(row["title"])
        col4.write(row["average_rating"])
