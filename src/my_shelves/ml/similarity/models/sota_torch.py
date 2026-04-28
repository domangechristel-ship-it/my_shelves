"""PYTHONPATH=src python src/my_shelves/ml/similarity/from_llm/copilot/sota_torch.py"""

import os

import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler
import joblib
import torch
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


class SimilaritySotaTorch:
    def __init__(self):
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        # Use GPU if available
        if torch.cuda.is_available():
            self.embedder = self.embedder.to('cuda')
        self.scaler = StandardScaler()
        self.index = None
        self.book_ids = None
        self.embeddings = None

    def prepare_model(self, n_rows: str = "10k"):
        # Load datasets
        self.data = pd.read_csv(f"{DATASET_ROOT}/similarity/extended_ENG_{n_rows}.csv")
        self.book_ids = self.data.index.tolist()

    def encode(self):
        # text_cols = ['author_names', 'top_emotion', 'emotions', 'country', 'region']
        self.data['combined_text'] = self.data[CAT_COLS].astype(str).agg(' '.join, axis=1)

        # Generate text embeddings
        texts = self.data['combined_text'].tolist()
        text_embeddings = self.embedder.encode(texts, show_progress_bar=True, batch_size=32)

        # Scale numerical
        # num_cols = ['n_votes', 'read_duration', 'average_rating', 'num_pages', 'ratings_count', 'total_shelves_count']
        num_scaled = self.scaler.fit_transform(self.data[NUM_COLS])

        # Binary
        binary = self._to_binary(self.data['is_series']).reshape(-1, 1)

        # Concatenate embeddings
        self.embeddings = np.hstack([text_embeddings, num_scaled, binary]).astype(np.float32)
        return self.embeddings

    def train(self,n_rows:str = "10k"):
        # Load datasets
        self.prepare_model(n_rows=n_rows)

        # Prepare text for embedding
        self.embeddings = self.encode()

        # Normalize for cosine similarity
        faiss.normalize_L2(self.embeddings)

        # Build Faiss index with GPU if available
        dim = self.embeddings.shape[1]
        if faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.GpuIndexFlatIP(res, dim)  # Inner product on GPU
        else:
            self.index = faiss.IndexFlatIP(dim)  # CPU fallback
        self.index.add(self.embeddings)

    def save(self,n_rows:str = "10k"):
        faiss.write_index(self.index, f"{MODELS_ROOT}/sota_torch/sota_torch_faiss_{n_rows}.index")
        np.save(f"{MODELS_ROOT}/sota_torch/sota_torch_embeddings_{n_rows}.npy", self.embeddings)
        joblib.dump(self.scaler, f"{MODELS_ROOT}/sota_torch/sota_torch_scaler_{n_rows}.pkl")
        joblib.dump(self.book_ids, f"{MODELS_ROOT}/sota_torch/sota_torch_book_ids_{n_rows}.pkl")
        joblib.dump(self.data, f"{MODELS_ROOT}/sota_torch/sota_torch_data_{n_rows}.pkl")  # For query preprocessing

    def load(self,n_rows:str = "10k"):
        self.index = faiss.read_index(f"{MODELS_ROOT}/sota_torch/sota_torch_faiss_{n_rows}.index")
        self.embeddings = np.load(f"{MODELS_ROOT}/sota_torch/sota_torch_embeddings_{n_rows}.npy")
        self.scaler = joblib.load(f"{MODELS_ROOT}/sota_torch/sota_torch_scaler_{n_rows}.pkl")
        self.book_ids = joblib.load(f"{MODELS_ROOT}/sota_torch/sota_torch_book_ids_{n_rows}.pkl")
        self.data = joblib.load(f"{MODELS_ROOT}/sota_torch/sota_torch_data_{n_rows}.pkl")

    def is_saved(self,n_rows:str = "10k"):
        required_files = [
            f"{MODELS_ROOT}/sota_torch/sota_torch_faiss_{n_rows}.index",
            f"{MODELS_ROOT}/sota_torch/sota_torch_embeddings_{n_rows}.npy",
            f"{MODELS_ROOT}/sota_torch/sota_torch_scaler_{n_rows}.pkl",
            f"{MODELS_ROOT}/sota_torch/sota_torch_book_ids_{n_rows}.pkl",
            f"{MODELS_ROOT}/sota_torch/sota_torch_data_{n_rows}.pkl",
        ]
        return all(os.path.exists(path) for path in required_files)

    def train_or_load(self,n_rows:str = "10k"):
        if self.is_saved(n_rows=n_rows):
            print(f"Saved sota_torch model with {n_rows} dataset model detected, loading from disk.", flush=True)
            self.load(n_rows=n_rows)
        else:
            print(f"No saved sota_torch model with {n_rows} dataset model found, training now.", flush=True)
            self.train(n_rows=n_rows)
            self.save(n_rows=n_rows)

    def _to_binary(self, series):
        values = series.astype(str).str.strip().str.lower()
        positive = values.isin({"y", "yes", "true", "1", "t"})
        return positive.astype(int).values

    def get_similar(self, book_id):
        if book_id not in self.book_ids:
            return []
        # Get the row
        row = self.data.loc[[book_id]]
        # Prepare text
        # text_cols = ['author_names', 'top_emotion', 'emotions', 'country', 'region']
        combined_text = row[CAT_COLS].astype(str).agg(' '.join, axis=1).iloc[0]
        # Generate text embedding
        text_emb = self.embedder.encode([combined_text])[0]
        # Scale numerical
        # num_cols = ['n_votes', 'read_duration', 'average_rating', 'num_pages', 'ratings_count', 'total_shelves_count']
        num_scaled = self.scaler.transform(row[NUM_COLS])
        # Binary
        binary = self._to_binary(row['is_series']).reshape(1, -1)
        # Concatenate
        query_emb = np.hstack([text_emb, num_scaled.flatten(), binary.flatten()]).reshape(1, -1).astype(np.float32)
        # Normalize
        faiss.normalize_L2(query_emb)
        # Search
        distances, indices = self.index.search(query_emb, 10)
        similar_book_ids = [self.book_ids[i] for i in indices[0]]
        return similar_book_ids


if __name__ == "__main__":
    model = SimilaritySotaTorch()
    for n_rows in N_ROWS_NAMES:
        model.train_or_load(n_rows=n_rows)
    # model.train_or_load(n_rows="10k")
    # Test
    similar_books = model.get_similar(1)
    print("Similar books to book_id 1:", similar_books)
