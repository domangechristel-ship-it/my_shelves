
# get dataset
# get features
# encode features on a subset of the dataset
# create model
# fit model

# get similarity:
# get X
# X_encoded = encode(X)
# model.get_similar(X_encoded)

# Add functions to API
# Add streamlit page
import os
import pickle
import pandas as pd
from google.cloud import bigquery

from my_shelves.ml.similarity.model import SimilarityKNN
from my_shelves.ml.similarity.encoder import get_transformer
from my_shelves.ml.similarity.params import DATASET_ROOT, MODELS_ROOT, N_ROWS_NAMES


def get_locations_df() -> pd.DataFrame:
    """
    Retrieve locations book from BigQuery.

    Returns
    -------
    pd.DataFrame
        DataFrame containing locations data.
    """
    if os.path.exists(f"{DATASET_ROOT}/locations.csv"):
        df = pd.read_csv(f"{DATASET_ROOT}/locations.csv", index_col="book_id")
        return df

    client = bigquery.Client()

    full_table_name = "books_dataset.book_locations"

    query = f"""
        SELECT book_id, country, region
        FROM `{full_table_name}`
    """

    df = client.query(query).to_dataframe()
    df = df.set_index("book_id")
    df.to_csv(f"{DATASET_ROOT}/locations.csv")
    return df


def get_emotions_df() -> pd.DataFrame:
    """
    Retrieve emotions book from BigQuery.

    Returns
    -------
    pd.DataFrame
        DataFrame containing emotions data.
    """
    if os.path.exists(f"{DATASET_ROOT}/emotions.csv"):
        df = pd.read_csv(f"{DATASET_ROOT}/emotions.csv", index_col="book_id")
        return df

    client = bigquery.Client()

    full_table_name = "books_dataset.emotion"

    query = f"""
        SELECT *
        FROM `{full_table_name}`
    """

    df = client.query(query).to_dataframe()
    df = df.set_index("book_id")
    df.to_csv(f"{DATASET_ROOT}/emotions.csv")
    return df

# base_features = ['book_id', 'n_votes', 'read_duration', 'series',
#        'average_rating', 'similar_books', 'publisher',
#        'num_pages', 'ratings_count', 'total_shelves_count',
#        'is_series', 'author_names']

base_features = ['book_id', 'n_votes', 'read_duration',
       'average_rating',
       'num_pages', 'ratings_count', 'total_shelves_count',
       'is_series', 'author_names']


def get_dataset(n_rows: str = "10k"):
    base_df = pd.read_csv(f"{DATASET_ROOT}/base_ENG_{n_rows}.csv",
                          usecols=base_features)
    locations_df = get_locations_df()
    df = base_df.merge(locations_df, on="book_id", how="left")
    df = df.set_index("book_id")

    emotions = get_emotions_df()
    df = df.merge(emotions, on="book_id", how="left")
    df = df.set_index("book_id")
    return df


def encode_features(X):
    encoder = get_transformer()
    X_encoded = encoder.fit_transform(X)
    return X_encoded


def create_model(X_encoded):
    model = SimilarityKNN(X_encoded, n_neighbors=10)
    model.fit_model()
    return model


def prepare_model(n_rows: str = "10k"):
    print(f"Preparing model with {n_rows} rows dataset...")
    print("Getting dataset...")
    df = get_dataset(n_rows=n_rows)
    # X = df[base_features]
    X = df
    print("Encoding features...")
    X_encoded = encode_features(X)
    X_encoded = X_encoded.dropna()
    print("Creating model...")
    model = create_model(X_encoded)
    print("Saving model...")
    with open(f"{MODELS_ROOT}/knn_model_{n_rows}.pkl", "wb") as f:
        pickle.dump(model, f)
    print("Model saved.")
    return model


def load_model(n_rows: str = "10k"):
    print("Loading model...")
    with open(f"{MODELS_ROOT}/knn_model_{n_rows}.pkl", "rb") as f:
        model = pickle.load(f)
    print("Model loaded.")
    return model


def get_similarity(book_id: int,
                   model_name: str = None,
                   n_rows: str = "20k") -> list[int]:
    """
    Retrieve a list of similar books based on a given book_id.

    Parameters
    ----------
    book_id : int
        The unique identifier of the book to find similarities for.

    Returns
    -------
    list[dict]
        A list of IDs of similar books.
    """
    print("Getting similarity...")
    if model_name == "knn_tf":
        # if os.path.exists(f"{DATASET_ROOT}/similarity/knn_tf_cache_{n_rows}.csv"):
        #     print("_" * 80)
        #     print("Use cached similar books")
        #     print("_" * 80)
        #     cache_df = pd.read_csv(f"{DATASET_ROOT}/similarity/knn_tf_cache_{n_rows}.csv")
        #     similar_books = cache_df[cache_df["book_id"] == book_id]["similar_books"].apply(eval).values[0]
        #     return similar_books
        from my_shelves.ml.similarity.models import knn_tf
        model = knn_tf.SimilarityKNNTF()
        model.train_or_load(n_rows=n_rows)
        # Test
        similar_books = model.get_similar(book_id, n_rows=n_rows)
        similar_books.insert(0, book_id)
        return similar_books
    elif model_name == "knn_sk":
        from my_shelves.ml.similarity.models import knn_sk
        model = knn_sk.SimilarityKNNSK()
        model.train_or_load(n_rows=n_rows)
        # Test
        print(f"Getting similar with knn_sk model with {n_rows} rows dataset...",
              flush=True)
        similar_books = model.get_similar(1)
        similar_books.insert(0, book_id)
        return similar_books
    elif model_name == "sota_torch":
        from my_shelves.ml.similarity.models import sota_torch
        model = sota_torch.SimilaritySotaTorch()
        model.train_or_load(n_rows=n_rows)
        # Test
        similar_books = model.get_similar(book_id)
        similar_books.insert(0, book_id)
        return similar_books
    elif model_name == "sota_tf":
        from my_shelves.ml.similarity.models import sota_tf
        model = sota_tf.SimilaritySotaTF()
        model.train_or_load(n_rows=n_rows)
        # Test
        similar_books = model.get_similar(book_id)
        similar_books.insert(0, book_id)
        return similar_books

    return []





if __name__ == "__main__":
    # for n_rows in N_ROWS_NAMES:
    #     prepare_model(n_rows=n_rows)
    # prepare_model()
    # print(get_similarity(1))
    compare_models = {}
    for ref_book_id in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
        for n_rows in N_ROWS_NAMES:
            if n_rows not in compare_models:
                compare_models[n_rows] = {}
            for model_name in ["knn_base", "knn_tf", "knn_sk", "sota_torch", "sota_tf"]:
                if model_name not in compare_models[n_rows]:
                    compare_models[n_rows][model_name] = {}
                print(f"Getting similarity with {model_name} model with {n_rows} rows dataset...", flush=True)
                similar_books = get_similarity(ref_book_id, model_name=model_name, n_rows=n_rows)
                print(f"Similar books to book_id {ref_book_id}: {similar_books}", flush=True)
                compare_models[n_rows][model_name][ref_book_id] = similar_books
    df = pd.DataFrame(compare_models)
    print(df)
