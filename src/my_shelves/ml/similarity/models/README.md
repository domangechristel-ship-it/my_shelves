# Similarity models

## Description

Find a collection of books with similarities to one book item based on a
selection of features.


## Features

Base Dataset:
- author_names
- n_votes
- read_duration
- average_rating
- num_pages
- ratings_count
- total_shelves_count

Emotions Dataset:
- top_emotion
- emotions

Location Dataset:
- country
- region


## Models

- knn_sk
- knn_tf
- sota_tf
- sota_torch

> [!TODO]
> - Make a `SimilarityModel` BaseClass
> - `data_[n_rows].plk` should be common for all models
> - `book_ids_[n_rows].plk` should be common for all models
> - `faiss_[n_rows].index` should be common for all SOTA models
> - Create a docker container that have all the code
> - Upload models data to gcp bucket
