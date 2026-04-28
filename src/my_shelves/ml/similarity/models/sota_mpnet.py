"""
Implementation of a Hybrid SOTA Similarity Model using MPNet and FAISS.
"""

import os
import logging
import numpy as np
import pandas as pd
import torch
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import RobustScaler

from my_shelves.ml.similarity.models.base_similarity import SimilarityModel
from my_shelves.ml.similarity.params import MODELS_ROOT, NUM_COLS, CAT_COLS


class HybridSOTAModel(SimilarityModel):
    """
    State-of-the-art Hybrid Similarity Model.
    Uses MPNet for semantic text embeddings and RobustScaler for numerical features.
    Optimized for GPU acceleration (RTX 3070).
    """
    def __init__(self):
        super().__init__("sota_mpnet")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if self.device == "cuda":
            logging.info(f"New Model: PyTorch detected GPU: {torch.cuda.get_device_name(0)}")
        self.embedder = SentenceTransformer(
            'all-mpnet-base-v2', device=self.device
        )

        self.scaler = RobustScaler()
        self.index = None
        self.embeddings = None

    def preprocess(self, data: pd.DataFrame) -> np.ndarray:
        """
        Fuse semantic text embeddings with normalized numerical metadata.
        """
        logging.info(
            f"Preprocessing {len(data)} books using Hybrid SOTA approach..."
        )

        text_cols = [c for c in CAT_COLS if c != "is_series"]
        combined_text = data[text_cols].astype(str).agg(
            ' | '.join, axis=1
        ).tolist()

        logging.info("Generating MPNet semantic embeddings...")
        text_embs = self.embedder.encode(
            combined_text, show_progress_bar=True, batch_size=64
        )

        num_scaled = self.scaler.fit_transform(data[NUM_COLS])
        binary = self._to_binary(data['is_series']).reshape(-1, 1)

        combined = np.hstack([text_embs, num_scaled, binary]).astype(np.float32)
        faiss.normalize_L2(combined)

        return combined

    def train(self, n_rows: str = "10k"):
        """
        Train the model and build a GPU-accelerated FAISS index.
        """
        data = self.get_data(n_rows)
        self.embeddings = self.preprocess(data)

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)

        # Step 15.1: Explicit verification for RTX 3070 detection
        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            if hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu'):
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logging.info(f"New Model: FAISS index built on GPU (RTX 3070 detected via get_num_gpus={faiss.get_num_gpus()}).")
                except Exception as e:
                    logging.warning(f"Failed to move FAISS index to GPU: {e}. Using CPU.")
        else:
            logging.warning("New Model: FAISS did not detect any GPUs. Check if faiss-gpu is installed instead of faiss-cpu.")

        self.index.add(self.embeddings)
        logging.info(f"Model trained. Index location: {'GPU' if 'Gpu' in str(type(self.index)) else 'CPU'}")

    def get_similar(self, book_id: int, n_neighbors: int = 10) -> list:
        """Wrapper for get_similar_batch for single queries."""
        cached = self._get_cached_similar(book_id, n_neighbors)
        if cached is not None:
            return cached

        return self.get_similar_batch([book_id], n_neighbors)[0]

    def get_similar_batch(self, book_ids: list, n_neighbors: int = 10) -> list:
        """Vectorized search using FAISS GPU index."""
        valid_ids = [bid for bid in book_ids if bid in self.book_ids]
        if not valid_ids:
            return [[] for _ in book_ids]

        # Extract indices for the batch
        indices = [self.book_ids.index(bid) for bid in valid_ids]
        query_vectors = self.embeddings[indices]

        _, top_indices = self.index.search(query_vectors, n_neighbors + 1)

        results = []
        for i, idx_list in enumerate(top_indices):
            sim_ids = [
                self.book_ids[j] for j in idx_list
                if self.book_ids[j] != valid_ids[i]
            ]
            results.append(sim_ids[:n_neighbors])

        return results

    def save(self, n_rows: str = "10k"):
        """Move index to CPU and save to disk."""
        self.save_common(n_rows)
        self.save_common_scaler(n_rows, self.scaler)
        model_dir = os.path.join(MODELS_ROOT, "sota_mpnet")
        os.makedirs(model_dir, exist_ok=True)

        save_index = self.index
        if hasattr(faiss, 'index_gpu_to_cpu') and 'Gpu' in str(type(self.index)):
            try:
                save_index = faiss.index_gpu_to_cpu(self.index)
            except Exception:
                pass

        faiss.write_index(
            save_index, os.path.join(model_dir, f"index_{n_rows}.faiss")
        )
        np.save(os.path.join(model_dir, f"embs_{n_rows}.npy"), self.embeddings)

    def load(self, n_rows: str = "10k"):
        """Load index from CPU and move back to GPU."""
        self.load_common(n_rows)
        self.scaler = self.load_common_scaler(n_rows)
        model_dir = os.path.join(MODELS_ROOT, "sota_mpnet")

        self.index = faiss.read_index(
            os.path.join(model_dir, f"index_{n_rows}.faiss")
        )

        if hasattr(faiss, 'get_num_gpus') and faiss.get_num_gpus() > 0:
            if hasattr(faiss, 'StandardGpuResources') and hasattr(faiss, 'index_cpu_to_gpu'):
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                except Exception:
                    pass
        self.embeddings = np.load(os.path.join(model_dir, f"embs_{n_rows}.npy"))

    def is_saved(self, n_rows: str = "10k") -> bool:
        """Check if FAISS index and embeddings exist."""
        model_dir = os.path.join(MODELS_ROOT, "sota_mpnet")
        files = [f"index_{n_rows}.faiss", f"embs_{n_rows}.npy"]
        return self.is_common_saved(n_rows, include_scaler=True) and \
               all(os.path.exists(os.path.join(model_dir, f)) for f in files)
