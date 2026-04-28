"""PYTHONPATH=src python src/my_shelves/ml/similarity/from_llm/copilot/knn_sk.py"""

import os

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer
from my_shelves.ml.similarity.params import DATASET_ROOT, MODELS_ROOT, N_ROWS_NAMES


NUM_COLS = ["n_votes",
            "read_duration",
            "average_rating",
            "num_pages",
            "ratings_count",
            "total_shelves_count"
            ]

CAT_COLS = ["is_series", "author_names"]

CLASSIFICATION_COLS = ['emotions', 'content_intensity',
       'romance_heat_level', 'character_type', 'main_themes', 'pace',
       'sentiment']

LOCATION_COLS = ["country", "region"]

CAT_COLS = CAT_COLS + CLASSIFICATION_COLS + LOCATION_COLS


class SimilarityKNNSK:
    def __init__(self):
        self.pipeline = None
        self.model = None
        self.data = None
        self.book_ids = None

    def prepare_model(self, n_rows: str = "10k"):
        # Load datasets
        self.data = pd.read_csv(f"{DATASET_ROOT}/similarity/extended_ENG_{n_rows}.csv")
        self.book_ids = self.data.index.tolist()

    def encode(self):
        # Preprocessing pipeline with imputers
        preprocessor = ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="mean")),
                    ("scaler", StandardScaler())
                ]), NUM_COLS),
                ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="constant", fill_value="unknown")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]), CAT_COLS)
            ],
            remainder="drop"
        )
        self.pipeline = Pipeline([("preprocessor", preprocessor)])

        # Fit and transform
        X_encoded = self.pipeline.fit_transform(self.data)
        return X_encoded

    def train(self, n_rows: str = "10k"):
        # Load datasets
        self.prepare_model(n_rows=n_rows)
        # Encode
        X_encoded = self.encode()
        # Fit NearestNeighbors
        self.model = NearestNeighbors(n_neighbors=10)
        self.model.fit(X_encoded)

    def save(self, n_rows: str = "10k"):
        joblib.dump(self.pipeline, f"{MODELS_ROOT}/knn_sk/knn_sk_pipeline_{n_rows}.pkl")
        joblib.dump(self.model, f"{MODELS_ROOT}/knn_sk/knn_sk_model_{n_rows}.pkl")
        joblib.dump(self.book_ids, f"{MODELS_ROOT}/knn_sk/knn_sk_book_ids_{n_rows}.pkl")
        joblib.dump(self.data, f"{MODELS_ROOT}/knn_sk/knn_sk_data_{n_rows}.pkl")

    def load(self, n_rows: str = "10k"):
        self.pipeline = joblib.load(f"{MODELS_ROOT}/knn_sk/knn_sk_pipeline_{n_rows}.pkl")
        self.model = joblib.load(f"{MODELS_ROOT}/knn_sk/knn_sk_model_{n_rows}.pkl")
        self.book_ids = joblib.load(f"{MODELS_ROOT}/knn_sk/knn_sk_book_ids_{n_rows}.pkl")
        self.data = joblib.load(f"{MODELS_ROOT}/knn_sk/knn_sk_data_{n_rows}.pkl")

    def is_saved(self, n_rows: str = "10k"):
        required_files = [
            f"{MODELS_ROOT}/knn_sk/knn_sk_pipeline_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_sk/knn_sk_model_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_sk/knn_sk_book_ids_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_sk/knn_sk_data_{n_rows}.pkl",
        ]
        return all(os.path.exists(path) for path in required_files)

    def train_or_load(self, n_rows: str = "10k"):
        if self.is_saved(n_rows=n_rows):
            print(f"Saved knn_sk model with {n_rows} dataset detected, loading from disk.", flush=True)
            self.load(n_rows=n_rows)
        else:
            print(f"No saved knn_sk model with {n_rows} dataset found, training now.", flush=True)
            self.train(n_rows=n_rows)
            self.save(n_rows=n_rows)

    def get_similar(self, book_id):
        print(f"Book {book_id} in data: {book_id in self.book_ids}")
        if book_id not in self.book_ids:
            return []
        # Get the row for the book_id
        row = self.data.loc[[book_id]]
        # Transform
        X_query = self.pipeline.transform(row)
        # Find neighbors
        distances, indices = self.model.kneighbors(X_query)
        similar_indices = indices[0]
        similar_book_ids = [self.book_ids[i] for i in similar_indices]
        return similar_book_ids


if __name__ == "__main__":
    print("Starting", flush=True)
    model = SimilarityKNNSK()
    for n_rows in N_ROWS_NAMES:
        model.train_or_load(n_rows=n_rows)

    # model.train_or_load("all")
    print("Getting similar", flush=True)
    similar_books = model.get_similar(1)
    print("Similar books to book_id 1:", similar_books, flush=True)
