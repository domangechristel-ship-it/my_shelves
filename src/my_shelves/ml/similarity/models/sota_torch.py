"""
State-of-the-Art Similarity implementation using SentenceTransformers and FAISS.
"""

import os
import logging
import joblib
import torch
import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import StandardScaler

from my_shelves.ml.similarity.params import MODELS_ROOT, N_ROWS_NAMES, NUM_COLS, CAT_COLS
from my_shelves.ml.similarity.models.base_similarity import SimilarityModel


class BookSimilaritySOTA(SimilarityModel):
    """
    Uses Sentence-BERT (MiniLM) and FAISS for semantic search.
    """
    def __init__(self):
        super().__init__("sota_torch")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')

        if torch.cuda.is_available():
            logging.info(f"SOTA Torch: PyTorch detected GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.is_available():
            self.embedder = self.embedder.to('cuda')
        self.scaler = StandardScaler()
        self.index = None
        self.embeddings = None

    def train(self,n_rows:str = "10k"):
        """Train the model and build a FAISS index."""
        data = self.get_data(n_rows)
        self.embeddings = self.preprocess(data)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)

        # Step 15.1: Log detection status
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            logging.info(f"SOTA Torch: FAISS detected {faiss.get_num_gpus()} GPU(s).")
            if hasattr(faiss, 'StandardGpuResources'):
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.GpuIndexFlatIP(res, dim)
                    logging.info("SOTA Torch: FAISS index successfully built on GPU.")
                except Exception:
                    logging.warning("SOTA Torch: FAISS GPU initialization failed. Using CPU.")
        else:
            logging.info("SOTA Torch: FAISS did not detect any GPUs.")

        self.index.add(self.embeddings)

    def preprocess(self, data: pd.DataFrame):
        """Encode text using SentenceTransformers and normalize metadata."""
        text_cols = [c for c in CAT_COLS if c != "is_series"]
        data['combined_text'] = data[text_cols].astype(str).agg(' '.join, axis=1)

        texts = data['combined_text'].tolist()
        text_embeddings = self.embedder.encode(
            texts, show_progress_bar=True, batch_size=32
        )

        num_scaled = self.scaler.fit_transform(data[NUM_COLS])
        binary = self._to_binary(data['is_series']).reshape(-1, 1)
        embeddings = np.hstack([
            text_embeddings, num_scaled, binary
        ]).astype(np.float32)

        faiss.normalize_L2(embeddings)
        return embeddings

    def save(self,n_rows:str = "10k"):
        """Save the FAISS index and scaled embeddings."""
        self.save_common(n_rows)
        self.save_common_scaler(n_rows, self.scaler)
        os.makedirs(os.path.join(MODELS_ROOT, "sota_torch"), exist_ok=True)
        faiss.write_index(
            self.index,
            f"{MODELS_ROOT}/sota_torch/sota_torch_faiss_{n_rows}.index"
        )
        np.save(
            f"{MODELS_ROOT}/sota_torch/sota_torch_embeddings_{n_rows}.npy",
            self.embeddings
        )

    def load(self,n_rows:str = "10k"):
        """Load the FAISS index and scaled embeddings."""
        self.load_common(n_rows)
        self.scaler = self.load_common_scaler(n_rows)
        self.index = faiss.read_index(
            f"{MODELS_ROOT}/sota_torch/sota_torch_faiss_{n_rows}.index"
        )
        self.embeddings = np.load(
            f"{MODELS_ROOT}/sota_torch/sota_torch_embeddings_{n_rows}.npy"
        )

    def is_saved(self,n_rows:str = "10k"):
        """Check for SOTA Torch model files."""
        required_files = [
            f"{MODELS_ROOT}/sota_torch/sota_torch_faiss_{n_rows}.index",
            f"{MODELS_ROOT}/sota_torch/sota_torch_embeddings_{n_rows}.npy",
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

        text_emb = self.embedder.encode([combined_text])[0]
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
    model = BookSimilaritySOTA()
    for n_rows in N_ROWS_NAMES:
        model.train_or_load(n_rows=n_rows)
    # model.train_or_load(n_rows="10k")
    # Test
    similar_books = model.get_similar(1)
    print("Similar books to book_id 1:", similar_books)
