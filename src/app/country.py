'''
streamlit country map and book list
'''

import ast
import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import requests


st.set_page_config(page_title="Books by Country", layout="wide")

# --------------------------------------------------
# Example data
# --------------------------------------------------
books_count_country = pd.DataFrame({
    "country": ["Belgium", "France", "Canada", "Italy"],
    "latlng": [[50.5039, 4.4699], [46.2276, 2.2137], [56.1304, -106.3468], [41.8719, 12.5674]],
    "count_books": [12, 25, 18, 9]
})

books_per_country = pd.DataFrame({
    "country": ["Belgium", "Belgium", "France", "France", "France", "Canada", "Italy"],
    "latlng": [
        [50.5039, 4.4699],
        [50.5039, 4.4699],
        [46.2276, 2.2137],
        [46.2276, 2.2137],
        [46.2276, 2.2137],
        [56.1304, -106.3468],
        [41.8719, 12.5674],
    ],
    "book_id": [101, 102, 201, 202, 203, 301, 401]
})
# --------------------------------------------------
# Data
# --------------------------------------------------
API_URL = 'http://127.0.0.1:8000/country'

response = requests.get(API_URL, timeout=10)

books_count_country = pd.DataFrame(response.json())
books_count_country["latlng"] = books_count_country["capital_latlng"].apply(ast.literal_eval)
books_count_country = books_count_country[["country", "latlng", "count_books"]].copy()
books_count_country = books_count_country[
    books_count_country["latlng"].notna() &
    (books_count_country["latlng"].apply(len) == 2)
]
# --------------------------------------------------
# Prepare coordinates
# --------------------------------------------------
books_count_country[["lat", "lon"]] = pd.DataFrame(
    books_count_country["latlng"].tolist(),
    index=books_count_country.index
)

books_per_country[["lat", "lon"]] = pd.DataFrame(
    books_per_country["latlng"].tolist(),
    index=books_per_country.index
)

# --------------------------------------------------
# Session state
# --------------------------------------------------
if "selected_country" not in st.session_state:
    st.session_state.selected_country = None

if "page" not in st.session_state:
    st.session_state.page = "Map"

# --------------------------------------------------
# Sidebar / navigation
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Map", "Books"],
    index=0 if st.session_state.page == "Map" else 1
)
st.session_state.page = page

st.title("🌍 Books by Country")

# --------------------------------------------------
# Page: Map
# --------------------------------------------------

bins = [0, 5, 10, 50, 100, 500, 1000, 100000]
labels = ["1-5", "6-10", "11-50", "51-100", "101-500", "501-1000", "1001+"]

books_count_country["count_bin"] = pd.cut(
    books_count_country["count_books"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

bin_colors = {
    "1-5": "#2c7bb6",
    "6-10": "#66c2a5",
    "11-50": "#abdda4",
    "51-100": "#ffffbf",
    "101-500": "#fdae61",
    "501-1000": "#f46d43",
    "1001+": "#d73027"
}

if st.session_state.page == "Map":
    st.subheader("World map")

    m = folium.Map(location=[20, 0], zoom_start=2, tiles="CartoDB positron")

    for _, row in books_count_country.iterrows():
        color = bin_colors.get(str(row["count_bin"]), "#cccccc")

        folium.CircleMarker(
            location=row["latlng"],
            radius=8,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7,
            popup=f"{row['country']}: {row['count_books']} books"
        ).add_to(m)

    map_data = st_folium(
        m,
        width=1200,
        height=500,
        returned_objects=["last_object_clicked_popup"]
    )

    clicked_popup = map_data.get("last_object_clicked_popup")

    if clicked_popup:
        # popup format = "country|count"
        country_clicked = clicked_popup.split("|")[0]

        # stocker dans session
        st.session_state.selected_country = country_clicked
        st.session_state.page = "Books"

        # 👉 ajouter dans l'URL
        st.query_params["page"] = "Books"
        st.query_params["country"] = country_clicked


        st.success(f"Country selected: {country_clicked}")
        st.rerun()

    if st.session_state.selected_country:
        st.info(f"Current selected country: {st.session_state.selected_country}")

# --------------------------------------------------
# Page: Books
# --------------------------------------------------
elif st.session_state.page == "Books":
    st.subheader("📚 Books list")

    if st.session_state.selected_country is None:
        st.warning("No country selected yet. Go to the map and click on a marker.")
    else:
        selected_country = st.session_state.selected_country

        filtered_books = books_per_country[
            books_per_country["country"] == selected_country
        ][["country", "book_id"]]

        st.write(f"Books for **{selected_country}**:")
        st.dataframe(filtered_books, use_container_width=True)

        if st.button("⬅ Back to map"):
            st.session_state.page = "Map"
            st.rerun()
