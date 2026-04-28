"""PYTHONPATH=src python src/my_shelves/ml/similarity/from_llm/copilot/knn_sk.py"""

import os

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from sklearn.impute import SimpleImputer

from my_shelves.ml.similarity.params import DATASET_ROOT, MODELS_ROOT, N_ROWS_NAMES, NUM_COLS, CAT_COLS
from my_shelves.ml.similarity.models.base_similarity import SimilarityModel


class BookSimilarityModel(SimilarityModel):
    def __init__(self):
        super().__init__("knn_sk")
        self.pipeline = None
        self.model = None

    def train(self, n_rows: str = "10k"):
        data = self.get_data(n_rows)
        X_encoded = self.preprocess(data)

        # Fit NearestNeighbors
        self.model = NearestNeighbors(n_neighbors=10)
        self.model.fit(X_encoded)

    def preprocess(self, data: pd.DataFrame):
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
        return self.pipeline.fit_transform(data)

    def save(self, n_rows: str = "10k"):
        """Save the pipeline and model artifacts."""
        self.save_common(n_rows)
        os.makedirs(os.path.join(MODELS_ROOT, "knn_sk"), exist_ok=True)
        joblib.dump(
            self.pipeline,
            f"{MODELS_ROOT}/knn_sk/knn_sk_pipeline_{n_rows}.pkl"
        )
        joblib.dump(
            self.model,
            f"{MODELS_ROOT}/knn_sk/knn_sk_model_{n_rows}.pkl"
        )

    def load(self, n_rows: str = "10k"):
        """Load the pipeline and model artifacts."""
        self.load_common(n_rows)
        self.pipeline = joblib.load(
            f"{MODELS_ROOT}/knn_sk/knn_sk_pipeline_{n_rows}.pkl"
        )
        self.model = joblib.load(
            f"{MODELS_ROOT}/knn_sk/knn_sk_model_{n_rows}.pkl"
        )

    def is_saved(self, n_rows: str = "10k"):
        """Check for Sklearn model files."""
        required_files = [
            f"{MODELS_ROOT}/knn_sk/knn_sk_pipeline_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_sk/knn_sk_model_{n_rows}.pkl",
        ]
        return (
            self.is_common_saved(n_rows) and
            all(os.path.exists(path) for path in required_files)
        )

    def get_similar(self, book_id, n_neighbors: int = 10):
        """Find similar books using the fitted pipeline."""
        cached = self._get_cached_similar(book_id, n_neighbors)
        if cached is not None:
            return cached

        if book_id not in self.book_ids:
            return []
        row = self.data.loc[[book_id]]
        X_query = self.pipeline.transform(row)
        _, indices = self.model.kneighbors(X_query, n_neighbors=n_neighbors + 1)
        similar_indices = indices[0]
        similar_book_ids = [
            self.book_ids[i] for i in similar_indices
            if self.book_ids[i] != book_id
        ]
        return similar_book_ids[:n_neighbors]


if __name__ == "__main__":
    print("Starting", flush=True)
    model = BookSimilarityModel()
    for n_rows in N_ROWS_NAMES:
        model.train_or_load(n_rows=n_rows)

    # model.train_or_load("all")
    print("Getting similar", flush=True)
    similar_books = model.get_similar(1)
    print("Similar books to book_id 1:", similar_books, flush=True)
