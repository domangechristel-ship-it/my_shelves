"""
Neural Autoencoder based Nearest Neighbors implementation using TensorFlow.
"""

import os
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import NearestNeighbors
import tensorflow as tf
from tensorflow import keras

from my_shelves.ml.similarity.params import MODELS_ROOT, N_ROWS_NAMES, NUM_COLS, CAT_COLS
from my_shelves.ml.similarity.models.base_similarity import SimilarityModel


class BookSimilarityModelTF(SimilarityModel):
    """
    Uses a Neural Autoencoder to compress features into embeddings.
    """
    def __init__(self):
        super().__init__("knn_tf")
        self.scaler = None
        self.les = {}
        self.encoder_model = None
        self.model = None
        self.data = None
        self.book_ids = None
        self.embeddings = None

    def train(self, n_rows: str = "10k"):
        """Train the neural autoencoder and fit neighbors."""
        data = self.get_data(n_rows)
        self.embeddings = self.preprocess(data)

        self.model = NearestNeighbors(n_neighbors=10)
        self.model.fit(self.embeddings)

    def preprocess(self, data: pd.DataFrame):
        """Build the TF model and extract embeddings."""
        merged = data.copy()
        self.scaler = StandardScaler()
        merged[NUM_COLS] = self.scaler.fit_transform(merged[NUM_COLS])

        for col in CAT_COLS:
            le = LabelEncoder()
            # astype(str) is key to handle mixed types
            merged[col] = le.fit_transform(merged[col].astype(str))
            self.les[col] = le

        self.encoded_data = merged.copy()

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
            emb_layer = keras.layers.Embedding(
                vocab_sizes[col], emb_dim, name=f'emb_{col}'
            )(inputs[col])
            emb_flat = keras.layers.Flatten()(emb_layer)
            embeddings.append(emb_flat)

        concat = keras.layers.Concatenate()(embeddings)
        encoder = keras.layers.Dense(128, activation='relu')(concat)
        encoder = keras.layers.Dense(64, activation='relu')(encoder)
        embedding = keras.layers.Dense(
            32, activation='relu', name='embedding'
        )(encoder)

        decoder = keras.layers.Dense(64, activation='relu')(embedding)
        decoder = keras.layers.Dense(128, activation='relu')(decoder)
        num_outputs = [
            keras.layers.Dense(1, name=f"output_{col}")(decoder)
            for col in NUM_COLS
        ]

        self.full_model = keras.Model(
            inputs=list(inputs.values()), outputs=num_outputs
        )
        self.full_model.compile(optimizer='adam', loss='mse')

        X_train = {
            col: merged[col].values.reshape(-1, 1)
            for col in NUM_COLS + CAT_COLS
        }
        y_train = [merged[col].values for col in NUM_COLS]

        self.full_model.fit(
            X_train, y_train, epochs=10, batch_size=32, verbose=1
        )

        self.encoder_model = keras.Model(
            inputs=list(inputs.values()), outputs=embedding
        )

        return self.encoder_model.predict(X_train)

    def save(self, n_rows: str = "10k"):
        """Save the TF encoder and neighborhood model."""
        self.save_common(n_rows)
        self.save_common_scaler(n_rows, self.scaler)
        os.makedirs(os.path.join(MODELS_ROOT, "knn_tf"), exist_ok=True)
        self.encoder_model.save(
            f"{MODELS_ROOT}/knn_tf/knn_tf_encoder_tf_{n_rows}.h5"
        )
        joblib.dump(self.les, f"{MODELS_ROOT}/knn_tf/knn_tf_les_{n_rows}.pkl")
        joblib.dump(self.model, f"{MODELS_ROOT}/knn_tf/knn_tf_nn_{n_rows}.pkl")
        joblib.dump(
            self.encoded_data,
            f"{MODELS_ROOT}/knn_tf/knn_tf_encoded_data_tf_{n_rows}.pkl"
        )

    def load(self, n_rows: str = "10k"):
        """Load the TF encoder and neighborhood model."""
        self.load_common(n_rows)
        self.scaler = self.load_common_scaler(n_rows)
        self.encoder_model = keras.models.load_model(
            f"{MODELS_ROOT}/knn_tf/knn_tf_encoder_tf_{n_rows}.h5"
        )
        self.les = joblib.load(f"{MODELS_ROOT}/knn_tf/knn_tf_les_{n_rows}.pkl")
        self.model = joblib.load(f"{MODELS_ROOT}/knn_tf/knn_tf_nn_{n_rows}.pkl")
        self.encoded_data = joblib.load(
            f"{MODELS_ROOT}/knn_tf/knn_tf_encoded_data_tf_{n_rows}.pkl"
        )

    def is_saved(self, n_rows: str = "10k"):
        """Check for TF model files."""
        required_files = [
            f"{MODELS_ROOT}/knn_tf/knn_tf_encoder_tf_{n_rows}.h5",
            f"{MODELS_ROOT}/knn_tf/knn_tf_les_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_tf/knn_tf_nn_{n_rows}.pkl",
            f"{MODELS_ROOT}/knn_tf/knn_tf_encoded_data_tf_{n_rows}.pkl",
        ]
        return (
            self.is_common_saved(n_rows, include_scaler=True) and
            all(os.path.exists(path) for path in required_files)
        )

    def get_similar(self, book_id, n_neighbors: int = 10):
        """Retrieve similar books by passing input through the encoder."""
        cached = self._get_cached_similar(book_id, n_neighbors)
        if cached is not None:
            return cached

        if book_id not in self.book_ids:
            return []
        row = self.encoded_data.loc[[book_id]]

        input_dict = {
            col: row[col].values.reshape(-1, 1)
            for col in NUM_COLS + CAT_COLS
        }
        emb = self.encoder_model.predict(input_dict, verbose=0)
        _, indices = self.model.kneighbors(emb, n_neighbors=n_neighbors + 1)
        similar_indices = indices[0]
        similar_book_ids = [
            self.book_ids[i] for i in similar_indices
            if self.book_ids[i] != book_id
        ]
        return similar_book_ids[:n_neighbors]


if __name__ == "__main__":
    model = BookSimilarityModelTF()
    for n_rows in N_ROWS_NAMES:
        model.train_or_load(n_rows=n_rows)
    # model.train_or_load(n_rows="10k")
    # Test
    similar_books = model.get_similar(1)
    print("Similar books to book_id 1:", similar_books)
