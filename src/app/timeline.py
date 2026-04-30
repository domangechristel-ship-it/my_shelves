import json
import html
from pathlib import Path

import pandas as pd
import streamlit as st


@st.cache_data
def load_timeline_data(eras_path: str, books_path: str) -> pd.DataFrame:
    with open(eras_path, encoding="utf-8") as f:
        eras = json.load(f)

    with open(books_path, encoding="utf-8") as f:
        books = json.load(f)


    df_eras = pd.DataFrame(eras)
    df_books = pd.DataFrame(books)
    # st.write("df_books columns:", df_books.columns.tolist())
    # st.write("df_books sample:", df_books.head())



    # Safety check
    if "era_order" not in df_books.columns:
        st.error("❌ 'era_order' column missing in books dataset")
        st.write(df_books.head())
        return pd.DataFrame()

    # Convert safely
    df_books["era_order"] = pd.to_numeric(df_books["era_order"], errors="coerce")
    df_eras["era_order"] = pd.to_numeric(df_eras["era_order"], errors="coerce")

    # Drop NaN (important)
    df_books = df_books.dropna(subset=["era_order"])
    df_eras = df_eras.dropna(subset=["era_order"])

    books_by_era = (
        df_books
        .groupby("era_order")
        .apply(lambda g: g.to_dict("records"))
        .reset_index(name="book_cards")
    )


    # df_eras["era_order"] = df_eras["era_order"].astype(int)
    # df_books["era_order"] = df_books["era_order"].astype(int)

    # books_by_era = (
    #     df_books
    #     .groupby("era_order")
    #     .apply(lambda g: g.drop(columns="era_order").to_dict("records"))
    #     .reset_index()
    #     .rename(columns={0: "book_cards"})
    # )

    df = df_eras.merge(books_by_era, on="era_order", how="left")
    df["book_cards"] = df["book_cards"].apply(
        lambda v: v if isinstance(v, list) else []
    )

    return df.sort_values("era_order").reset_index(drop=True)


def clean_year(value) -> str:
    if value is None:
        return ""

    value = str(value).replace("[", "").replace("]", "").replace("'", "").strip()

    return value


def fmt_century(c: int) -> str:
    return f"{abs(c)}th c. BCE" if c < 0 else f"{c}th c. CE"


def render_books(book_cards: list) -> str:
    if not book_cards:
        return '<span class="no-books">No book covers linked to this era yet.</span>'

    html_cards = []

    for book in book_cards:
        title = html.escape(str(book.get("title", "Unknown title")))
        year = html.escape(clean_year(book.get("year", "")))
        image_url = html.escape(str(book.get("image_url", "")))
        book_id = html.escape(str(book.get("book_id", "")))
        bg = html.escape(str(book.get("bg", "#FFFFFF")))
        text_color = html.escape(str(book.get("text", "#222222")))

        if image_url:
            cover_html = f'<img src="{image_url}" alt="{title}" loading="lazy">'
        else:
            cover_html = '<div class="cover-placeholder">No cover</div>'

        card_html = (
            f'<a class="book-card2" href="?book_id={book_id}" '
            f'style="background:{bg};" title="{title}">'
            f'<div class="book-cover2">{cover_html}</div>'
            f'<div class="book-info2">'
            f'<div class="book-title2" style="color:{text_color};">{title}</div>'
            f'<div class="book-year2" style="color:{text_color};">{year}</div>'
            f'</div>'
            f'</a>'
        )

        html_cards.append(card_html)

    return "".join(html_cards)


def show_timeline() -> None:
    ERAS_FILE = Path("data/eras.json")
    BOOKS_FILE = Path("data/book.json")

    df = load_timeline_data(str(ERAS_FILE), str(BOOKS_FILE))

    st.markdown("""
    <style>
      .timeline-title {
        font-size: 28px;
        font-weight: 800;
        margin-bottom: 4px;
      }

      .timeline-subtitle {
        color: #6c757d;
        font-size: 15px;
        margin-bottom: 24px;
      }

      .era-card {
        border-radius: 14px;
        border-width: 1.5px;
        border-style: solid;
        padding: 1.2rem 1.5rem;
        margin: 0.4rem 0;
      }

      .era-header-row {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 0.9rem;
        flex-wrap: wrap;
      }

      .era-icon {
        font-size: 1.5rem;
      }

      .era-title {
        font-size: 1.45rem;
        font-weight: 700;
        margin: 0;
        flex: 1;
      }

      .era-pill {
        font-size: 0.75rem;
        font-weight: 600;
        padding: 3px 12px;
        border-radius: 20px;
        white-space: nowrap;
        border-width: 1px;
        border-style: solid;
      }

      .events-list {
        margin: 0 0 1rem;
        padding: 0;
        list-style: none;
      }

      .events-list li {
        font-size: 0.84rem;
        color: #444;
        padding: 3px 0 3px 18px;
        position: relative;
        line-height: 1.5;
      }

      .events-list li .dot {
        position: absolute;
        left: 0;
        top: 8px;
        width: 6px;
        height: 6px;
        border-radius: 50%;
      }

      .books-label {
        font-size: 0.7rem;
        font-weight: 700;
        letter-spacing: 0.09em;
        text-transform: uppercase;
        margin-bottom: 8px;
      }

      .books-row {
        display: flex;
        gap: 14px;
        flex-wrap: wrap;
      }

      .book-card2 {
        width: 130px;
        min-height: 220px;
        border-radius: 12px;
        overflow: hidden;
        text-decoration: none !important;
        box-shadow: 0 3px 12px rgba(0,0,0,0.15);
        transition: transform 0.18s, box-shadow 0.18s;
        display: block;
      }

      .book-card2:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.22);
      }

      .book-cover2 img {
        width: 100%;
        height: 170px;
        object-fit: cover;
        display: block;
      }

      .cover-placeholder {
        height: 170px;
        display: flex;
        align-items: center;
        justify-content: center;
        color: #999;
        background: #eee;
        font-size: 0.75rem;
      }

      .book-info2 {
        padding: 8px 10px 10px;
      }

      .book-title2 {
        font-size: 0.82rem;
        font-weight: 700;
        line-height: 1.25;
      }

      .book-year2 {
        font-size: 0.72rem;
        opacity: 0.75;
        margin-top: 4px;
      }

      .no-books {
        font-size: 0.78rem;
        color: #aaa;
        font-style: italic;
      }

      .stat-row {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-top: 1rem;
        padding-top: 0.75rem;
        border-top: 1px solid rgba(0,0,0,0.07);
      }

      .stat-track {
        flex: 1;
        height: 5px;
        background: rgba(0,0,0,0.08);
        border-radius: 3px;
        overflow: hidden;
      }

      .stat-fill {
        height: 100%;
        border-radius: 3px;
      }

      .stat-lbl {
        font-size: 0.7rem;
        color: #888;
        white-space: nowrap;
      }

      .stat-num {
        font-size: 0.78rem;
        font-weight: 600;
        white-space: nowrap;
      }

      .connector {
        display: flex;
        justify-content: center;
        align-items: center;
        height: 32px;
        position: relative;
      }

      .connector::before {
        content: "";
        position: absolute;
        left: 50%;
        transform: translateX(-50%);
        width: 2px;
        height: 100%;
        background: linear-gradient(to bottom, rgba(0,0,0,0.06), rgba(0,0,0,0.12));
      }

      .connector-dot {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        background: #fff;
        border: 1.5px solid #ddd;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.65rem;
        color: #bbb;
        position: relative;
        z-index: 1;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
      }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="timeline-title">📜 Historical Timeline</div>
    """, unsafe_allow_html=True)

    # c1, c2, c3 = st.columns(3)
    # c1.metric("Eras covered", len(df))
    # c2.metric("Books in dataset", int(df["books"].sum()))
    # c3.metric("Centuries covered", int(df["century"].max() - df["century"].min()))

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

    max_books = int(df["books"].max()) if int(df["books"].max()) > 0 else 1

    for idx, row in df.iterrows():
        color = row.get("color", "#555")
        light_bg = row.get("light_bg", "#fafafa")
        border = row.get("border", "#ddd")
        pill_bg = row.get("pill_bg", "#eee")
        icon = row.get("icon", "•")
        events = row.get("events", [])
        books = row.get("book_cards", [])

        events_html = "".join(
            f'<li><span class="dot" style="background:{color};"></span>{html.escape(str(event))}</li>'
            for event in events
        )

        st.markdown(f"""
        <div class="era-card" style="background:{light_bg}; border-color:{border};">
          <div class="era-header-row">
            <span class="era-icon">{icon}</span>
            <span class="era-title" style="color:{color};">{html.escape(str(row["era"]))}</span>
            <span class="era-pill"
                  style="background:{pill_bg}; color:{color}; border-color:{border};">
              {fmt_century(int(row["century"]))}
            </span>
          </div>

          <ul class="events-list">{events_html}</ul>

          <div class="books-label" style="color:{color};">📚 Related books</div>
          <div class="books-row">{render_books(books)}</div>
          <div class="stat-row">
            <span class="stat-lbl">Books in dataset</span>
            <div class="stat-track">
              <div class="stat-fill"
                   style="width:{int(row["books"] / max_books * 100)}%; background:{color};">
              </div>
            </div>
            <span class="stat-num" style="color:{color};">{int(row["books"])}</span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        if idx < len(df) - 1:
            st.markdown(
                '<div class="connector"><div class="connector-dot">▼</div></div>',
                unsafe_allow_html=True,
            )
