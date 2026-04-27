import requests
import streamlit as st

from params import API_URL_BOOKS, API_URL_BOOK_IDS_FILTERS
from show_pages import show_books_table
from my_shelves.ml.classification.dict_features import dict_labels


def show_book_by_filters():

    col_filters, col_results = st.columns([1, 2])

    with col_filters:
        # st.markdown("### 🎛️ Filters")

        params = {}

        for feature, config in dict_labels.items():

            labels = config["labels"]
            is_multi = config["multi_label"]
            display_name = feature.replace("_", " ").capitalize()

            if is_multi:
                selected = st.multiselect(display_name, labels)
                if selected:
                    params[feature] = selected
            else:
                selected = st.selectbox(display_name, [""] + labels)
                if selected:
                    params[feature] = selected

        emotions = st.multiselect(
            "Emotions",
            ["fear", "joy", "neutral", "disgust", "sadness", "surprise", "anger"]
        )
        if emotions:
            params["emotions"] = emotions

        sentiment = st.selectbox(
            "Sentiment",
            ["", "positive", "neutral", "negative"]
        )
        if sentiment:
            params["sentiment"] = sentiment

        #st.write(params)



    with col_results:
        #st.markdown("### 📚 Results")

        apply = st.button("🔍 Apply filters")

        if apply:

            # st.write("### Active filters")
            # st.json(params)

            try:
                response_ids = requests.get(
                    API_URL_BOOK_IDS_FILTERS,
                    params=params,
                    headers={"accept": "application/json"},
                    timeout=10
                )


                if response_ids.status_code != 200:
                    st.error(f"API error: {response_ids.status_code}")
                    st.write(response_ids.text)
                    return

                book_ids = response_ids.json()


                if not book_ids:
                    st.warning("No books found with those filters.")
                    return

                # 🔹 récupérer les livres
                book_params = [("book_id_list", bid) for bid in book_ids]

                response_books = requests.get(
                    API_URL_BOOKS,
                    params=book_params,
                    timeout=20
                )

                if response_books.status_code != 200:
                    st.error(f"Books API error: {response_books.status_code}")
                    st.write(response_books.text)
                    return

                show_books_table(response_books.json())

            except requests.RequestException as exc:
                st.error(f"Request error: {exc}")
