import os
from abc import ABC, abstractmethod
from typing import Any
import pandas as pd
from my_shelves.ml.similarity.params import DATASET_ROOT, MODELS_ROOT

class SimilarityModel(ABC):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.data = None
        self.book_ids = None

    def load_dataset(self, n_rows: str) -> pd.DataFrame:
        """Standard method to load the merged training dataset."""
        data_path = f"{DATASET_ROOT}/similarity/extended_ENG_{n_rows}.csv"
        df = pd.read_csv(data_path)
        df = df.set_index("book_id")
        self.data = df
        self.book_ids = df.index.tolist()
        return self.data

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
            print(f"Saved {self.model_name} model for {n_rows} detected, loading...", flush=True)
            self.load(n_rows=n_rows)
        else:
            print(f"No saved {self.model_name} model for {n_rows} found, training...", flush=True)
            self.train(n_rows=n_rows)
            self.save(n_rows=n_rows)

    @abstractmethod
    def get_similar(self, book_id: Any, n_neighbors: int = 10) -> list:
        pass

    def _to_binary(self, series: pd.Series):
        """Utility to convert is_series values to binary."""
        values = series.astype(str).str.strip().str.lower()
        positive = values.isin({"y", "yes", "true", "1", "t"})
        return positive.astype(int).values
