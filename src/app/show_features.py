import requests
import streamlit as st

from params import API_URL_BOOKS, API_URL_BOOK_IDS_FILTERS
from show_pages import show_books_table
from dict_features import dict_labels
from helpers import book_spinner


def show_book_by_filters():
    # --------------------------------------------------
    # Charge uniqument si tab actif
    # --------------------------------------------------
    if "tab_features" not in st.session_state:
        st.session_state.tab_country_loaded = True
    else:
        return

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

        # ------------------
        # Emoji mapping
        # ------------------

        emotion_map = {
            "fear": "😨 ",
            "joy": "😊 ",
            "neutral": "😐 ",
            "disgust": "🤢 ",
            "sadness": "😢 ",
            "surprise": "😲 ",
            "anger": "😡 ",
        }

        # Init session state
        if "selected_emotions" not in st.session_state:
            st.session_state.selected_emotions = []

        st.write("Emotions")

        cols = st.columns(len(emotion_map))

        for i, (emotion, emoji) in enumerate(emotion_map.items()):
            is_selected = emotion in st.session_state.selected_emotions

            if cols[i].button(
                emoji,
                key=emotion,
                help=emotion,
                type="primary" if is_selected else "secondary"
            ):
                if is_selected:
                    st.session_state.selected_emotions.remove(emotion)
                else:
                    st.session_state.selected_emotions.append(emotion)

        # Use result
        emotions = st.session_state.selected_emotions

        if emotions:
            params["emotions"] = emotions

        # ------------------
        # Sentiment slider
        # ------------------
        sentiment_options = {
            " ": None,
            "➖ Negative": "negative",
            "0️⃣ Neutral": "neutral",
            "➕ Positive": "positive",
        }

        selected_label = st.selectbox(
            "Sentiment",
            options=list(sentiment_options.keys()),
            index=0
        )

        sentiment = sentiment_options[selected_label]

        if sentiment is not None:
            params["sentiment"] = sentiment



    with col_results:
        apply = st.button("🔍 Apply filters")

        if apply:
            loader = st.empty()

            try:
                # STEP 1: get matching book IDs
                with loader:
                    book_spinner("Finding matching books...")

                response_ids = requests.get(
                    API_URL_BOOK_IDS_FILTERS,
                    params=params,
                    headers={"accept": "application/json"},
                    timeout=20
                )

                if response_ids.status_code != 200:
                    loader.empty()
                    st.error(f"API error: {response_ids.status_code}")
                    st.write(response_ids.text)
                    return

                book_ids = response_ids.json()

                if not book_ids:
                    loader.empty()
                    st.warning("📕 No books found with those filters.")
                    return

                # STEP 2: get book details
                book_params = [("book_id_list", bid) for bid in book_ids]

                with loader:
                    book_spinner("Retrieving book details...")

                response_books = requests.get(
                    API_URL_BOOKS,
                    params=book_params,
                    timeout=20
                )

                loader.empty()

                if response_books.status_code != 200:
                    st.error(f"⚠️ Books API error: {response_books.status_code}")
                    st.write(response_books.text)
                    return

                show_books_table(response_books.json())

            except requests.RequestException as exc:
                loader.empty()
                st.error(f"Request error: {exc}")
