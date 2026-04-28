"""PYTHONPATH=src python src/my_shelves/ml/similarity/from_llm/copilot/sota_tf.py"""


import os
import logging

import pandas as pd
import faiss
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.preprocessing import StandardScaler
import joblib

from my_shelves.ml.similarity.params import DATASET_ROOT, MODELS_ROOT, N_ROWS_NAMES, NUM_COLS, CAT_COLS
from my_shelves.ml.similarity.models.base_similarity import SimilarityModel


class BookSimilaritySOTATF(SimilarityModel):
    def __init__(self):
        super().__init__("sota_tf")

        # Step 15.1: Ensure TensorFlow doesn't hog all VRAM,
        # which allows FAISS to initialize its own GPU context.
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"TensorFlow configured with memory growth for {len(gpus)} GPUs.")
        except Exception as e:
            logging.warning(f"Could not configure TF memory growth: {e}")

        # Use TensorFlow Hub's Universal Sentence Encoder for text embeddings
        self.embedder = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        self.scaler = StandardScaler()
        self.index = None
        self.embeddings = None
        self.les = {}

    def train(self, n_rows: str = "10k"):
        data = self.get_data(n_rows)
        self.embeddings = self.preprocess(data)

        # Step 15.1: Enhanced GPU detection logic
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            logging.info(f"FAISS detected {faiss.get_num_gpus()} GPU(s). Attempting GPU index creation.")
            if hasattr(faiss, 'StandardGpuResources'):
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.GpuIndexFlatIP(res, dim)
                    logging.info("SOTA TF: FAISS index successfully built on GPU.")
                except Exception:
                    logging.warning("SOTA TF: FAISS GPU index creation failed. Falling back to CPU.")
        else:
            logging.info("SOTA TF: FAISS did not detect any GPUs. Using CPU index.")

        self.index.add(self.embeddings)

    def preprocess(self, data: pd.DataFrame):
        """Encode text using TF USE and normalize metadata."""
        text_cols = [c for c in CAT_COLS if c != "is_series"]
        data['combined_text'] = data[text_cols].astype(str).agg(' '.join, axis=1)

        texts = data['combined_text'].tolist()
        text_embeddings = self.embedder(texts).numpy()

        num_scaled = self.scaler.fit_transform(data[NUM_COLS])
        binary = self._to_binary(data['is_series']).reshape(-1, 1)
        embeddings = np.hstack([
            text_embeddings, num_scaled, binary
        ]).astype(np.float32)

        faiss.normalize_L2(embeddings)
        return embeddings

    def save(self, n_rows: str = "10k"):
        """Save the FAISS index and scaled embeddings."""
        self.save_common(n_rows)
        self.save_common_scaler(n_rows, self.scaler)
        os.makedirs(os.path.join(MODELS_ROOT, "sota_tf"), exist_ok=True)
        faiss.write_index(
            self.index,
            f"{MODELS_ROOT}/sota_tf/sota_tf_faiss_tf_{n_rows}.index"
        )
        np.save(
            f"{MODELS_ROOT}/sota_tf/sota_tf_embeddings_tf_{n_rows}.npy",
            self.embeddings
        )
        joblib.dump(
            self.les,
            f"{MODELS_ROOT}/sota_tf/sota_tf_les_tf_{n_rows}.pkl"
        )

    def load(self, n_rows: str = "10k"):
        """Load the FAISS index and scaled embeddings."""
        self.load_common(n_rows)
        self.scaler = self.load_common_scaler(n_rows)
        self.index = faiss.read_index(
            f"{MODELS_ROOT}/sota_tf/sota_tf_faiss_tf_{n_rows}.index"
        )
        self.embeddings = np.load(
            f"{MODELS_ROOT}/sota_tf/sota_tf_embeddings_tf_{n_rows}.npy"
        )
        self.les = joblib.load(
            f"{MODELS_ROOT}/sota_tf/sota_tf_les_tf_{n_rows}.pkl"
        )

    def is_saved(self, n_rows: str = "10k"):
        """Check for SOTA TF files."""
        required_files = [
            f"{MODELS_ROOT}/sota_tf/sota_tf_faiss_tf_{n_rows}.index",
            f"{MODELS_ROOT}/sota_tf/sota_tf_embeddings_tf_{n_rows}.npy",
            f"{MODELS_ROOT}/sota_tf/sota_tf_les_tf_{n_rows}.pkl",
        ]
        return (
            self.is_common_saved(n_rows, include_scaler=True) and
            all(os.path.exists(path) for path in required_files)
        )

    def get_similar(self, book_id, n_neighbors: int = 10):
        """Find similar books by performing a FAISS vector search."""
        cached = self._get_cached_similar(book_id, n_neighbors)
        if cached is not None:
            return cached

        if book_id not in self.book_ids:
            return []

        row = self.data.loc[[book_id]]
        text_cols = [c for c in CAT_COLS if c != "is_series"]
        combined_text = row[text_cols].astype(str).agg(' '.join, axis=1).iloc[0]

        text_emb = self.embedder([combined_text]).numpy()[0]
        num_scaled = self.scaler.transform(row[NUM_COLS])
        binary = self._to_binary(row['is_series']).reshape(1, -1)

        query_emb = np.hstack([
            text_emb, num_scaled.flatten(), binary.flatten()
        ]).reshape(1, -1).astype(np.float32)

        faiss.normalize_L2(query_emb)
        _, indices = self.index.search(query_emb, n_neighbors + 1)
        similar_book_ids = [
            self.book_ids[i] for i in indices[0]
            if self.book_ids[i] != book_id
        ]
        return similar_book_ids[:n_neighbors]


if __name__ == "__main__":
    model = BookSimilaritySOTATF()
    for n_rows in N_ROWS_NAMES:
        model.train_or_load(n_rows=n_rows)
    # model.train_or_load(n_rows="10k")
    # Test
    similar_books = model.get_similar(1)
    print("Similar books to book_id 1:", similar_books)
