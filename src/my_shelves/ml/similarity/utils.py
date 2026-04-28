import pandas as pd


from my_shelves.ml.similarity.params import DATASET_ROOT, MODELS_ROOT, N_ROWS_NAMES


NUM_COLS = ["n_votes",
            "read_duration",
            "average_rating",
            "num_pages",
            "ratings_count",
            "total_shelves_count"
            ]

CAT_COLS = ["is_series", "author_names", "top_emotion", "country", "region"]

CLASSIFICATION_COLS = ['book_id', 'emotions', 'content_intensity',
       'romance_heat_level', 'character_type', 'main_themes', 'pace',
       'sentiment']

LOCATION_COLS = ["book_id", "country", "region"]


def prepare_data(n_rows: str = "10k") -> str:
    output_file = f"{DATASET_ROOT}/similarity/extended_ENG_{n_rows}.csv"
    classification_csv = f"{DATASET_ROOT}/classification/classification_merged_features_{n_rows}.csv"
    locations_csv = f"{DATASET_ROOT}/location/locations_{n_rows}.csv"

    # Load datasets
    base = pd.read_csv(f"{DATASET_ROOT}/base_ENG_{n_rows}.csv")
    classification = pd.read_csv(classification_csv,
                           usecols=CLASSIFICATION_COLS)
    locations = pd.read_csv(locations_csv,
                            usecols=LOCATION_COLS)

    # Merge on book_id with left joins
    merged = base.merge(classification, on="book_id", how="left")\
        .merge(locations, on="book_id", how="left")
    # Fill NaN: 0 for numerical, "unknown" for others
    merged[NUM_COLS] = merged[NUM_COLS].fillna(0)
    merged = merged.fillna("unknown")
    # merged = merged.set_index("book_id")

    merged.to_csv(output_file, index=False)
    return output_file
