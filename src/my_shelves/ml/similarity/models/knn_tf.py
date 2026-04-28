"""PYTHONPATH=src python src/my_shelves/ml/similarity/from_llm/copilot/knn_tf.py"""
import os

import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow import keras

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


class SimilarityKNNTF:
    def __init__(self):
        self.scaler = None
        self.les = {}
        self.encoder_model = None
        self.model = None
        self.data = None
        self.book_ids = None
        self.embeddings = None

    def prepare_model(self, n_rows: str = "10k"):
        # Load datasets
        self.data = pd.read_csv(f"{DATASET_ROOT}/similarity/extended_ENG_{n_rows}.csv")
        self.book_ids = self.data.index.tolist()
        # self.book_ids = self.data["book_id"].to_list()

    def encode(self):
        self.scaler = StandardScaler()
        merged = self.data.copy()
        merged[NUM_COLS] = self.scaler.fit_transform(merged[NUM_COLS])

        for col in CAT_COLS:
            le = LabelEncoder()
            merged[col] = le.fit_transform(merged[col].astype(str)) # <-- astype(str) is key
            self.les[col] = le

        # Save the encoded version for inference
        self.encoded_data = merged.copy()
        return self.encoded_data

    def train(self, n_rows: str = "10k"):
        # Load datasets
        self.prepare_model(n_rows=n_rows)
        # Save the encoded version for inference
        self.encoded_data = self.encode()

        # Build model
        inputs = {}
        embeddings = []
        vocab_sizes = {}
        emb_dim = 10  # embedding dim for cat

        for col in NUM_COLS:
            inputs[col] = keras.Input(shape=(1,), name=col)
            embeddings.append(inputs[col])

        for col in CAT_COLS:
            vocab_sizes[col] = len(self.les[col].classes_)
            inputs[col] = keras.Input(shape=(1,), name=col)
            emb_layer = keras.layers.Embedding(vocab_sizes[col], emb_dim, name=f'emb_{col}')(inputs[col])
            emb_flat = keras.layers.Flatten()(emb_layer)
            embeddings.append(emb_flat)

        concat = keras.layers.Concatenate()(embeddings)
        encoder = keras.layers.Dense(128, activation='relu')(concat)
        encoder = keras.layers.Dense(64, activation='relu')(encoder)
        embedding = keras.layers.Dense(32, activation='relu', name='embedding')(encoder)

        # Decoder for reconstruction (only num for simplicity)
        decoder = keras.layers.Dense(64, activation='relu')(embedding)
        decoder = keras.layers.Dense(128, activation='relu')(decoder)
        num_outputs = [keras.layers.Dense(1, name=f"output_{col}")(decoder) for col in NUM_COLS]

        self.full_model = keras.Model(inputs=list(inputs.values()), outputs=num_outputs)
        self.full_model.compile(optimizer='adam', loss='mse')

        # Prepare data for training
        X_train = {col: self.encoded_data[col].values.reshape(-1, 1) for col in NUM_COLS + CAT_COLS}
        y_train = [self.encoded_data[col].values for col in NUM_COLS]

        self.full_model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

        # Create encoder model
        self.encoder_model = keras.Model(inputs=list(inputs.values()), outputs=embedding)

        # Get embeddings for all books
        self.embeddings = self.encoder_model.predict(X_train)

        # Fit NearestNeighbors on embeddings
        self.model = NearestNeighbors(n_neighbors=10)
        self.model.fit(self.embeddings)

    def save(self, n_rows: str = "10k"):
        self.encoder_model.save(f"{MODELS_ROOT}/knn_tf/knn_tf_encoder_tf_{n_rows}.h5")
        joblib.dump(self.scaler, f"{MODELS_ROOT}/knn_tf/knn_tf_scaler_{n_rows}.pkl")
        joblib.dump(self.les, f"{MODELS_ROOT}/knn_tf/knn_tf_les_{n_rows}.pkl")
        joblib.dump(self.book_ids, f"{MODELS_ROOT}/knn_tf/knn_tf_book_ids_{n_rows}.pkl")
        joblib.dump(self.model, f"{MODELS_ROOT}/knn_tf/knn_tf_nn_{n_rows}.pkl")
        joblib.dump(self.data, f"{MODELS_ROOT}/knn_tf/knn_tf_data_tf_{n_rows}.pkl")
        joblib.dump(self.encoded_data, f"{MODELS_ROOT}/knn_tf/knn_tf_encoded_data_tf_{n_rows}.pkl")  # ADD
        joblib.dump(self.embeddings, f"{MODELS_ROOT}/knn_tf/knn_tf_embeddings_{n_rows}.pkl")

    def load(self, n_rows: str = "10k"):
        self.encoder_model = keras.models.load_model(f"{MODELS_ROOT}/knn_tf/knn_tf_encoder_tf_{n_rows}.h5")
        self.scaler = joblib.load(f"{MODELS_ROOT}/knn_tf/knn_tf_scaler_{n_rows}.pkl")
        self.les = joblib.load(f"{MODELS_ROOT}/knn_tf/knn_tf_les_{n_rows}.pkl")
        self.book_ids = joblib.load(f"{MODELS_ROOT}/knn_tf/knn_tf_book_ids_{n_rows}.pkl")
        self.model = joblib.load(f"{MODELS_ROOT}/knn_tf/knn_tf_nn_{n_rows}.pkl")
        self.data = joblib.load(f"{MODELS_ROOT}/knn_tf/knn_tf_data_tf_{n_rows}.pkl")
        self.encoded_data = joblib.load(f"{MODELS_ROOT}/knn_tf/knn_tf_encoded_data_tf_{n_rows}.pkl")  # ADD
        self.embeddings = joblib.load(f"{MODELS_ROOT}/knn_tf/knn_tf_embeddings_{n_rows}.pkl")

    def is_saved(self, n_rows: str = "10k"):
        required_files = [
            f"{MODELS_ROOT}/knn_tf/knn_tf_encoder_tf_{n_rows}.h5",
            f"{MODELS_ROOT}/knn_tf/knn_tf_scaler_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_tf/knn_tf_les_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_tf/knn_tf_book_ids_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_tf/knn_tf_nn_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_tf/knn_tf_data_tf_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_tf/knn_tf_encoded_data_tf_{n_rows}.pkl",  # ADD
            f"{MODELS_ROOT}/knn_tf/knn_tf_embeddings_{n_rows}.pkl",
        ]
        return all(os.path.exists(path) for path in required_files)

    def train_or_load(self, n_rows: str = "10k"):
        if self.is_saved(n_rows=n_rows):
            print(f"Saved knn_tf model with {n_rows} dataset detected, loading from disk.", flush=True)
            self.load(n_rows=n_rows)
        else:
            print(f"No saved knn_tf model with {n_rows} dataset found, training now.", flush=True)
            self.train(n_rows=n_rows)
            self.save(n_rows=n_rows)

    def get_similar(self, book_id, n_rows: str = "10k"):
        if book_id not in self.book_ids:
            return []
        # if os.path.exists(f"{DATASET_ROOT}/similarity/knn_tf_cache_{n_rows}.csv"):
        #     print("_" * 80)
        #     print("Use cached similar books")
        #     print("_" * 80)
        #     cache_df = pd.read_csv(f"{DATASET_ROOT}/similarity/knn_tf_cache_{n_rows}.csv")
        #     similar_indices = cache_df[cache_df["book_id"] == book_id]["similar_books"].apply(eval).values[0]
        #     # similar_book_ids = [self.book_ids[i] for i in similar_indices]
        #     print(type(similar_indices), similar_indices)
        #     return similar_indices

        # Get and preprocess the row
        num_cols = ["n_votes", "read_duration", "average_rating", "num_pages", "ratings_count", "total_shelves_count"]
        cat_cols = list(self.les.keys())

        # Use already-encoded + scaled row directly
        row = self.encoded_data.loc[[book_id]]  # No re-encoding needed

        input_dict = {col: row[col].values.reshape(-1, 1) for col in num_cols + cat_cols}
        # Get embedding
        emb = self.encoder_model.predict(input_dict, verbose=0)
        # Find neighbors
        distances, indices = self.model.kneighbors(emb)
        similar_indices = indices[0]
        similar_book_ids = [self.book_ids[i] for i in similar_indices]
        return similar_book_ids

    def save_cache(self, n_rows: str = "10k"):
        """
        Save a minimal similarity cache:
        ["book_id", "similar_books", "similarity_distance"]
        """

        if self.embeddings is None:
            raise ValueError("Embeddings not found. Train or load model first.")

        # Compute neighbors for ALL rows at once (fast)
        distances, indices = self.model.kneighbors(self.embeddings, return_distance=True)

        # Map indices -> book_ids
        similar_books = [
            [self.book_ids[i] for i in row]
            for row in indices
        ]

        similarity_distances = distances.tolist()

        # Build minimal dataframe
        cache_df = pd.DataFrame({
            "book_id": self.book_ids,
            "similar_books": similar_books,
            "similarity_distance": similarity_distances
        })

        # Optional: serialize lists to make CSV cleaner
        import json
        cache_df["similar_books"] = cache_df["similar_books"].apply(json.dumps)
        cache_df["similarity_distance"] = cache_df["similarity_distance"].apply(json.dumps)

        # Save
        output_path = f"{DATASET_ROOT}/similarity/knn_tf_cache_{n_rows}.csv"
        cache_df.to_csv(output_path, index=False)

        return cache_df


if __name__ == "__main__":
    model = SimilarityKNNTF()
    for n_rows in N_ROWS_NAMES:
        model.train_or_load(n_rows=n_rows)
    # model.train_or_load(n_rows="10k")
    # Test
    similar_books = model.get_similar(1)
    print("Similar books to book_id 1:", similar_books)
