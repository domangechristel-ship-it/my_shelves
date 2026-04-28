"""
Base class module for similarity models.
"""

import os
import logging
import ast
from abc import ABC, abstractmethod
from typing import Any, List
import pandas as pd
import joblib

from my_shelves.ml.similarity.params import DATASET_ROOT, MODELS_ROOT


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


class SimilarityModel(ABC):
    """
    Abstract base class for all similarity model implementations.

    Provides shared logic for data loading, persistence, common artifacts,
    and incremental caching.
    """
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.data = None
        self.book_ids = None
        self.n_rows = "10k"

    def load_dataset(self, n_rows: str) -> pd.DataFrame:
        """Standard method to load the merged training dataset."""
        data_path = f"{DATASET_ROOT}/similarity/extended_ENG_{n_rows}.csv"
        df = pd.read_csv(data_path)
        df = df.drop_duplicates(subset='book_id')
        df = df.set_index("book_id")
        self.data = df
        self.book_ids = df.index.tolist()
        return self.data

    def get_data(self, n_rows: str) -> pd.DataFrame:
        """Step 5.1: Wrapper for loading data."""
        return self.load_dataset(n_rows)

    @abstractmethod
    def train(self, n_rows: str = "10k"):
        """Train the model using the specified number of rows."""

    @abstractmethod
    def preprocess(self, data: pd.DataFrame) -> Any:
        """Perform feature engineering and scaling on the input data."""

    @abstractmethod
    def save(self, n_rows: str = "10k"):
        """Save the model and associated artifacts to disk."""

    @abstractmethod
    def load(self, n_rows: str = "10k"):
        """Load the model and associated artifacts from disk."""

    @abstractmethod
    def is_saved(self, n_rows: str = "10k") -> bool:
        """Check if the model artifacts already exist on disk."""

    def train_or_load(self, n_rows: str = "10k"):
        """Train the model if it doesn't exist, otherwise load it."""
        self.n_rows = n_rows
        if self.is_saved(n_rows=n_rows):
            logging.info(
                f"Saved {self.model_name} model for {n_rows} "
                "detected, loading..."
            )
            self.load(n_rows=n_rows)
        else:
            logging.info(
                f"No saved {self.model_name} model for {n_rows} "
                "found, training..."
            )
            self.train(n_rows=n_rows)
            self.save(n_rows=n_rows)

    @abstractmethod
    def get_similar(self, book_id: Any, n_neighbors: int = 10) -> list:
        """Retrieve a list of similar book IDs for a single book ID."""

    def _get_cached_similar(self, book_id: Any, n_neighbors: int) -> list:
        """Helper to retrieve similar books from CSV cache if available."""
        cache_path = os.path.join(
            DATASET_ROOT, "similarity",
            f"similarities_{self.model_name}_{self.n_rows}.csv"
        )
        if os.path.exists(cache_path):
            try:
                cache_df = pd.read_csv(cache_path)
                match = cache_df[cache_df['book_id'].astype(str) == str(book_id)]
                if not match.empty:
                    return ast.literal_eval(match['similar_books'].iloc[0])[:n_neighbors]
            except Exception as e:
                logging.debug(f"Cache lookup failed for {book_id}: {e}")
        return None

    def get_similar_batch(self, book_ids: List[Any], n_neighbors: int = 10) -> List[List[Any]]:
        """
        Suggested Improvement: Batch processing.
        Subclasses should override this to use vectorized search.
        """
        return [self.get_similar(bid, n_neighbors) for bid in book_ids]

    def _to_binary(self, series: pd.Series):
        """Utility to convert is_series values to binary."""
        values = series.astype(str).str.strip().str.lower()
        positive = values.isin({"y", "yes", "true", "1", "t"})
        return positive.astype(int).values

    def save_common(self, n_rows: str):
        """Saves data and book_ids to a common location."""
        common_dir = os.path.join(MODELS_ROOT, "common")
        os.makedirs(common_dir, exist_ok=True)
        joblib.dump(self.data, os.path.join(common_dir, f"data_{n_rows}.pkl"))
        joblib.dump(self.book_ids, os.path.join(common_dir, f"book_ids_{n_rows}.pkl"))

    def load_common(self, n_rows: str):
        """Loads data and book_ids from the common location."""
        common_dir = os.path.join(MODELS_ROOT, "common")
        self.data = joblib.load(os.path.join(common_dir, f"data_{n_rows}.pkl"))
        self.book_ids = joblib.load(os.path.join(common_dir, f"book_ids_{n_rows}.pkl"))

    def save_common_scaler(self, n_rows: str, scaler: Any):
        """Saves the shared scaler to the common location."""
        common_dir = os.path.join(MODELS_ROOT, "common")
        os.makedirs(common_dir, exist_ok=True)
        joblib.dump(scaler, os.path.join(common_dir, f"scaler_{n_rows}.pkl"))

    def load_common_scaler(self, n_rows: str) -> Any:
        """Loads the shared scaler from the common location."""
        return joblib.load(os.path.join(MODELS_ROOT, "common", f"scaler_{n_rows}.pkl"))

    def is_common_saved(self, n_rows: str, include_scaler: bool = False) -> bool:
        """Checks if common files exist."""
        common_dir = os.path.join(MODELS_ROOT, "common")
        files = [f"data_{n_rows}.pkl", f"book_ids_{n_rows}.pkl"]
        if include_scaler:
            files.append(f"scaler_{n_rows}.pkl")
        return all(os.path.exists(os.path.join(common_dir, f)) for f in files)

    def save_cache(self, n_rows: str = "50k", batch_size: int = 500):
        """Step 6: Pre-calculate similarities and save incrementally to CSV."""
        cache_dir = os.path.join(DATASET_ROOT, "similarity")
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(
            cache_dir,
            f"similarities_{self.model_name}_{n_rows}.csv"
        )

        processed_ids = set()
        if os.path.exists(cache_path):
            try:
                existing_df = pd.read_csv(cache_path)
                if not existing_df.empty and 'book_id' in existing_df.columns:
                    processed_ids = set(existing_df['book_id'].unique().tolist())
                    logging.info(f"Found existing cache with {len(processed_ids)} entries.")
            except (pd.errors.EmptyDataError, pd.errors.ParserError):
                pass

        ids_to_process = [bid for bid in self.book_ids if bid not in processed_ids]

        if not ids_to_process:
            logging.info(
                f"All books in dataset are already processed in "
                f"cache for {n_rows}."
            )
            return

        logging.info(f"Processing {len(ids_to_process)} remaining books for {self.model_name}...")

        results = []
        for i in range(0, len(ids_to_process), batch_size):
            batch_ids = ids_to_process[i : i + batch_size]
            # Use the batch method for speed
            batch_similar = self.get_similar_batch(batch_ids)

            chunk_results = [
                {"book_id": bid, "similar_books": sim}
                for bid, sim in zip(batch_ids, batch_similar)
            ]

            df_chunk = pd.DataFrame(chunk_results)
            header = (
                not os.path.exists(cache_path) or
                os.stat(cache_path).st_size == 0
            )
            df_chunk.to_csv(cache_path, mode='a', index=False, header=header)

            logging.info(
                f"Saved chunk: {min(i + batch_size, len(ids_to_process))}/"
                f"{len(ids_to_process)} processed."
            )
