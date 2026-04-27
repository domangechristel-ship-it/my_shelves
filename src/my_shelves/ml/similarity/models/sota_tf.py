"""PYTHONPATH=src python src/my_shelves/ml/similarity/from_llm/copilot/sota_tf.py"""


import os

import pandas as pd
import faiss
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
import joblib
from my_shelves.ml.similarity.params import DATASET_ROOT, MODELS_ROOT, N_ROWS_NAMES


class SimilaritySotaTF:
    def __init__(self):
        # Use TensorFlow Hub's Universal Sentence Encoder for text embeddings
        self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.scaler = StandardScaler()
        self.index = None
        self.book_ids = None
        self.embeddings = None
        self.les = {}

    def train(self, n_rows: str = "10k"):
        # Load datasets
        base = pd.read_csv(f"{DATASET_ROOT}/base_ENG_{n_rows}.csv")
        emotions = pd.read_csv(f"{DATASET_ROOT}/emotions.csv", usecols=["book_id", "emotions", "top_emotion"])
        locations = pd.read_csv(f"{DATASET_ROOT}/locations.csv", usecols=["book_id", "country", "region"])

        # Merge on book_id with left joins
        merged = base.merge(emotions, on="book_id", how="left").merge(locations, on="book_id", how="left")
        # Fill NaN: 0 for numerical, "unknown" for others
        num_cols = ["n_votes", "read_duration", "average_rating", "num_pages", "ratings_count", "total_shelves_count"]
        merged[num_cols] = merged[num_cols].fillna(0)
        merged = merged.fillna("unknown")
        merged = merged.set_index("book_id")

        self.data = merged
        self.book_ids = merged.index.tolist()

        # Prepare text for embedding
        text_cols = ['author_names', 'top_emotion', 'emotions', 'country', 'region']
        merged['combined_text'] = merged[text_cols].astype(str).agg(' '.join, axis=1)

        # Generate text embeddings with TF USE (GPU if available)
        texts = merged['combined_text'].tolist()
        text_embeddings = self.embedder(texts).numpy()  # Convert to numpy

        # Scale numerical
        num_cols = ['n_votes', 'read_duration', 'average_rating', 'num_pages', 'ratings_count', 'total_shelves_count']
        num_scaled = self.scaler.fit_transform(merged[num_cols])

        # Binary
        binary = self._to_binary(merged['is_series']).reshape(-1, 1)

        # Concatenate embeddings
        self.embeddings = np.hstack([text_embeddings, num_scaled, binary]).astype(np.float32)

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

    def save(self, n_rows: str = "10k"):
        faiss.write_index(self.index, f"{MODELS_ROOT}/sota_tf/sota_tf_faiss_tf_{n_rows}.index")
        np.save(f"{MODELS_ROOT}/sota_tf/sota_tf_embeddings_tf_{n_rows}.npy", self.embeddings)
        joblib.dump(self.scaler, f"{MODELS_ROOT}/sota_tf/sota_tf_scaler_tf_{n_rows}.pkl")
        joblib.dump(self.les, f"{MODELS_ROOT}/sota_tf/sota_tf_les_tf_{n_rows}.pkl")
        joblib.dump(self.book_ids, f"{MODELS_ROOT}/sota_tf/sota_tf_book_ids_tf_{n_rows}.pkl")
        joblib.dump(self.data, f"{MODELS_ROOT}/sota_tf/sota_tf_data_tf_{n_rows}.pkl")

    def load(self, n_rows: str = "10k"):
        self.index = faiss.read_index(f"{MODELS_ROOT}/sota_tf/sota_tf_faiss_tf_{n_rows}.index")
        self.embeddings = np.load(f"{MODELS_ROOT}/sota_tf/sota_tf_embeddings_tf_{n_rows}.npy")
        self.scaler = joblib.load(f"{MODELS_ROOT}/sota_tf/sota_tf_scaler_tf_{n_rows}.pkl")
        self.les = joblib.load(f"{MODELS_ROOT}/sota_tf/sota_tf_les_tf_{n_rows}.pkl")
        self.book_ids = joblib.load(f"{MODELS_ROOT}/sota_tf/sota_tf_book_ids_tf_{n_rows}.pkl")
        self.data = joblib.load(f"{MODELS_ROOT}/sota_tf/sota_tf_data_tf_{n_rows}.pkl")

    def is_saved(self, n_rows: str = "10k"):
        required_files = [
            f"{MODELS_ROOT}/sota_tf/sota_tf_faiss_tf_{n_rows}.index",
            f"{MODELS_ROOT}/sota_tf/sota_tf_embeddings_tf_{n_rows}.npy",
            f"{MODELS_ROOT}/sota_tf/sota_tf_scaler_tf_{n_rows}.pkl",
            f"{MODELS_ROOT}/sota_tf/sota_tf_les_tf_{n_rows}.pkl",
            f"{MODELS_ROOT}/sota_tf/sota_tf_book_ids_tf_{n_rows}.pkl",
            f"{MODELS_ROOT}/sota_tf/sota_tf_data_tf_{n_rows}.pkl",
        ]
        return all(os.path.exists(path) for path in required_files)

    def train_or_load(self, n_rows: str = "10k"):
        if self.is_saved(n_rows=n_rows):
            print(f"Saved sota_tf model with {n_rows} dataset detected, loading from disk.", flush=True)
            self.load(n_rows=n_rows)
        else:
            print(f"No saved sota_tf model with {n_rows} dataset found, training now.", flush=True)
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
        text_cols = ['author_names', 'top_emotion', 'emotions', 'country', 'region']
        combined_text = row[text_cols].astype(str).agg(' '.join, axis=1).iloc[0]
        # Generate text embedding with TF
        text_emb = self.embedder([combined_text]).numpy()[0]
        # Scale numerical
        num_cols = ['n_votes', 'read_duration', 'average_rating', 'num_pages', 'ratings_count', 'total_shelves_count']
        num_scaled = self.scaler.transform(row[num_cols])
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
    # Enable GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    model = SimilaritySotaTF()
    for n_rows in N_ROWS_NAMES:
        model.train_or_load(n_rows=n_rows)
    # model.train_or_load(n_rows="10k")
    # Test
    similar_books = model.get_similar(1)
    print("Similar books to book_id 1:", similar_books)
