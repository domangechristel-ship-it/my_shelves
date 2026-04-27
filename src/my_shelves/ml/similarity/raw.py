import ast
import pandas as pd


from sklearn.neighbors import NearestNeighbors
from my_shelves.ml.similarity.encoder import get_transformer
from my_shelves.ml.similarity.params import DATASET_ROOT




def encode_features(X):
    encoder = get_transformer()
    X_encoded = encoder.fit_transform(X)
    return X_encoded


def main():
    base_features = ['book_id', 'n_votes', 'read_duration',
       'average_rating',
       'num_pages', 'ratings_count', 'total_shelves_count']

    index = 1

    base_dataset = f"{DATASET_ROOT}/base_ENG_all.csv"
    # The dataset to train the model
    base_df = pd.read_csv(base_dataset,
                          usecols=base_features)
    base_df = base_df.set_index("book_id")

    # The dataset to display the results
    base_display = pd.read_csv(base_dataset,
                          usecols=["book_id", "title", "author_names", "similar_books"])
    base_display = base_display.set_index("book_id")

    # The initial book we want to find similars
    main_book = base_display.iloc[index-1:index]

    # Similar books from the original dataset
    similar_books = list(int(i) for i in ast.literal_eval(main_book["similar_books"].values[0]))

    # Encode base_df before training
    X_encoded = encode_features(base_df)
    X_encoded = X_encoded.dropna()

    model = NearestNeighbors(n_neighbors=10)
    model.fit(X_encoded)

    # The initial book we want to find similars
    X_new = X_encoded.iloc[index-1:index]

    # Get Similar books
    distances, indices = model.kneighbors(X_new)
    neighbors_indices = indices[0]
    neighbors_book_ids = X_encoded.index[neighbors_indices]
    # print("Neighbor indices:", neighbors_indices)
    # print("Neighbor book_ids:", neighbors_book_ids)
    # print("All book_ids in original dataset:", all(bid in base_10k.index for bid in neighbors_book_ids))

    books_indices = indices[0, 1:]
    books_distances = distances[0, 1:]

    # Display
    print("_" * 120)
    print(f"Books similar to {main_book['title'].values[0]} by {main_book['author_names'].values[0]}")
    print("_" * 120)
    playlist = base_display.iloc[books_indices].copy()
    playlist['distance'] = books_distances
    print("_" * 120)
    print(playlist)
    print("_" * 120)
    # similar_books_ids = list(similar_books[:min(len(similar_books), 10)])
    # similar_playlist = base_display.iloc[similar_books_ids].copy()
    # print(similar_playlist)




if __name__ == "__main__":
    main()
