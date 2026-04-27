from abc import ABC, abstractmethod
from typing import Any, Self
import pandas as pd
import numpy as np


from my_shelves.ml.similarity.params import DATASET_ROOT, MODELS_ROOT, N_ROWS_NAMES


class SimilarityModel(ABC):
    def __init__(self):
        self.data = None
        self.book_ids = None

    def prepare_model(self, n_rows: str = "10k"):
        # Load datasets
        base = pd.read_csv(f"{DATASET_ROOT}/base_ENG_{n_rows}.csv")
        emotions = pd.read_csv(f"{DATASET_ROOT}/emotions.csv",
                               usecols=["book_id", "emotions", "top_emotion"])
        locations = pd.read_csv(f"{DATASET_ROOT}/locations.csv",
                                usecols=["book_id", "country", "region"])

        # Merge on book_id with left joins
        merged = base.merge(emotions, on="book_id", how="left")\
            .merge(locations, on="book_id", how="left")
        # Fill NaN: 0 for numerical, "unknown" for others
        num_cols = ["n_votes", "read_duration", "average_rating", "num_pages", "ratings_count", "total_shelves_count"]
        merged[num_cols] = merged[num_cols].fillna(0)
        merged = merged.fillna("unknown")
        merged = merged.set_index("book_id")

        self.data = merged
        self.book_ids = merged.index.tolist()

    @abstractmethod
    def train(self, n_rows: str = "10k"):
        pass

    @abstractmethod
    def save(self, n_rows: str = "10k"):
        pass

    @abstractmethod
    def load(self, n_rows: str = "10k"):
        pass

    @abstractmethod
    def is_saved(self, n_rows: str = "10k") -> bool:
        pass

    def train_or_load(self, n_rows: str = "10k"):
        if self.is_saved(n_rows=n_rows):
            print(f"Saved model with {n_rows} dataset detected, loading from disk.", flush=True)
            self.load(n_rows=n_rows)
        else:
            print(f"No saved model with {n_rows} dataset found, training now.", flush=True)
            self.train(n_rows=n_rows)
            self.save(n_rows=n_rows)

    @abstractmethod
    def get_similar(self, book_id: Any, n_neighbors: int = 10) -> list:
        pass


class SimilarityKNN(SimilarityModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, n_rows: str = "10k"):
        # Load datasets
        base = pd.read_csv(f"{DATASET_ROOT}/base_ENG_{n_rows}.csv")
        emotions = pd.read_csv(f"{DATASET_ROOT}/emotions.csv", usecols=["book_id", "emotions", "top_emotion"])
        locations = pd.read_csv(f"{DATASET_ROOT}/locations.csv", usecols=["book_id", "country", "region"])

        # Merge on book_id with left joins
        merged = base.merge(emotions, on="book_id", how="left").merge(locations, on="book_id", how="left")
        # Fill NaNs:


class SimilaritySOTA(SimilarityModel):
    def __init__(self):
        super().__init__()
        self.model = None

    def train(self, n_rows: str = "10k"):
        # Load datasets
        base = pd.read_csv(f"{DATASET_ROOT}/base_ENG_{n_rows}.csv")
        emotions = pd.read_csv
        locations = pd.read_csv(
