"""
This module contains functions to process the raw datasets of books and reviews
by language and number of rows.
"""

import ast
import os
import csv
import pandas as pd

from prefect import task, flow
from prefect.artifacts import create_table_artifact

# Package imports
from my_shelves.prepare import clean
from my_shelves.prepare import utils
from my_shelves.utils import bigquery

from my_shelves.utils.params import GCP_PROJECT


PREFECT_FLOW_NAME="preprocess-data"

N_WORDS_THRESHOLD = 50
N_ROWS_MAP = {"10k": 10_000,
              "20k": 20_000,
              "50k": 50_000,
              "100k": 100_000,
              "150k": 150_000,
              "200k": 200_000,
              "all": None
}

N_ROWS_NAMES = N_ROWS_MAP.keys()

@task
def process_books_by_lang_all(chunksize: int = 100000, force: bool = False) -> str:
    """
    Process the raw books dataset by language and save the results in separate
    CSV files for each language (French, English, Dutch). The function reads the
    raw books dataset from a JSON file, filters the books by language, and saves
    the results in separate CSV files for each language. The function also counts
    the number of books for each language and prints the results.

    Returns
    -------
    str
        The paths to the processed books CSV files for each language.
    """
    json_file = "data/kaggle/goodreads_books.json"

    output_file = "data/raw/goodreads_books_ENG.csv"


    if os.path.exists(output_file) and not force:
        print("Books files already exist. Skipping processing.")
        return output_file

    headers_written = False
    print("Processing books by language...")
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = None
        for i, chunk in enumerate(pd.read_json(json_file, lines=True, chunksize=chunksize)):
            print(f"Processing chunk {i + 1}...")
            for _, row in chunk.iterrows():
                lang = row.get("language_code")
                if pd.isna(lang):
                    continue

                lang = str(lang)
                row_dict = row.to_dict()
                if not headers_written:
                    fieldnames = list(row_dict.keys())
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    headers_written = True
                if lang.startswith("en"):
                    writer.writerow(row_dict)

    # Very important save the output file sorted by "book_id" and reindexed.
    print("Sorting books by book_id...")
    df = pd.read_csv(output_file)
    df = df.sort_values(by="book_id").reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print(f"Saving books to {output_file}...")
    return output_file


# pylint: disable-msg=too-many-locals
@task
def process_reviews_by_lang_all(chunksize: int=100000, force: bool=False) -> str:
    """
    Process the raw reviews dataset by language and number of rows and save
    the results in separate CSV files for each language and number of rows.
    The function reads the raw reviews dataset from a JSON file, filters the
    reviews by language and number of rows, and saves the results in separate CSV
    files for each language and number of rows. The function also counts the
    number of reviews for each language and number of rows and prints the results.

    Parameters
    ----------
    lang : str
        The language code used to filter the reviews (e.g., 'FRA', 'ENG', 'DUT').
    nrows : str, default='all'
        The number of rows to process (e.g., '10k', '100k', '1M', 'all').
    chunksize : int, default=10000
    Returns
    -------
    str
        The path to the processed reviews CSV file for the specified language
        and number of rows.
    """
    n_rows = "all"
    lang = "ENG"

    print("_" * 80)
    print(f"Processing {lang} reviews with {n_rows} rows...")
    print("_" * 80)

    json_file = "data/kaggle/goodreads_reviews_dedup.json"
    books_file = f'data/raw/goodreads_books_{lang}.csv'
    output_file = f"data/raw/goodreads_reviews_{lang}_{n_rows}.csv"

    if os.path.exists(output_file) and not force:
        print(f"{output_file} already exists. Skipping processing.")
        return output_file

    def get_books_ids(books_file):
        books = pd.read_csv(books_file,
                            nrows=N_ROWS_MAP[n_rows],
                            usecols=['book_id'])
        book_ids = books["book_id"].values

        # to delete books and released memory
        books = pd.DataFrame()
        del books
        return book_ids

    book_ids = get_books_ids(books_file)
    header_written = False
    n_reviews = 0

    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = None

        for i, chunk in enumerate(pd.read_json(json_file,
                                               lines=True,
                                               chunksize=chunksize)):
            print(f"Processing chunk {i + 1}...")
            for _, row in chunk.iterrows():
                if row['book_id'] in book_ids:
                    n_reviews += 1
                    row_dict = row.to_dict()

                    # Initialize writer once with headers
                    if not header_written:
                        writer = csv.DictWriter(f, fieldnames=row_dict.keys())
                        writer.writeheader()
                        header_written = True

                    # Write ONE row immediately
                    writer.writerow(row_dict)
                    # if nrows != "all" and n_reviews >= N_ROWS_MAP[nrows]:
                    #     break
            # if nrows != "all" and n_reviews >= N_ROWS_MAP[nrows]:
            #     break

    print("Done.")
    print(f"{lang} reviews count:{n_reviews}")

    # Very important save the output file sorted by "book_id" and reindexed.
    df = pd.read_csv(output_file)
    df = df.sort_values(by="book_id").reset_index(drop=True)
    df.to_csv(output_file, index=False)

    return output_file


def process_reviews_chunk(n_rows: str="10k"):
    """Split the reviews dataset to smaller chunks."""
    print(f"Processing {n_rows} rows...")
    print("Reading raw reviews dataset...")
    reviews = pd.read_csv("data/goodreads_reviews_ENG_all.csv",
                            nrows=N_ROWS_MAP[n_rows])
    print("Sorting reviews by book_id...")
    reviews.sort_values(by="book_id", inplace=True)
    print("Saving reviews to CSV file...")
    reviews.to_csv(f"data/goodreads_reviews_ENG_{n_rows}.csv", index=False)


@task
def clean_reviews_all(force: bool=False) -> str:
    """
    Process the raw reviews dataset by language and number of rows and
    save the results in separate CSV files for each language and number of rows.
    The function reads the raw reviews dataset from a CSV file,
    cleans the reviews text, adds a new column that counts the number of words
    in the review text, removes reviews with less than a specified
    number of words, cleans the date columns, computes the read duration,
    drops useless columns, groups the reviews by book_id,
    and saves the results in separate CSV files for each language
    and number of rows. The function also counts the number of reviews
    for each language and number of rows and prints the results.

    Parameters
    ----------
    lang : str
        The language code used to filter the reviews (e.g., 'FRA', 'ENG', 'DUT').
    nrows : str, default='all'
        The number of rows to process (e.g., '10k', '100k', '1M', 'all').
    force : bool, default=False
        Whether to force the processing even if the output file already exists.

    Returns
    -------
    str
        The path to the processed reviews CSV file for the specified language
        and number of rows.
    """
    # Get reviews raw dataset for the specified language and number of rows
    lang = "ENG"
    n_rows = "all"

    print("_" * 80)
    print(f"Processing cleaned reviews for {lang} with {n_rows} rows...")
    print("_" * 80)

    output_file = f"data/reviews_cleaned_{lang}_{n_rows}.csv"

    if os.path.exists(output_file) and not force:
        print(f"{output_file} already exists. Skipping processing.")
        return output_file

    # Read the reviews dataset for the specified language and number of rows
    print(f"Reading reviews dataset for {lang} with {n_rows} rows...")
    reviews_df = pd.read_csv(f"data/raw/goodreads_reviews_{lang}_all.csv",
                             nrows=N_ROWS_MAP[n_rows])
    # Clean the reviews text by removing rows with missing or empty review text
    print("Cleaning reviews text...")
    reviews_df = clean.clean_text(reviews_df, column_name="review_text")

    # Add a new column that counts the number of words in the review text
    print("Adding reviews word count...")
    reviews_df = utils.add_reviews_count_words(reviews_df, column_name="review_text")

    # Clean the reviews by removing rows where the number of words in the review
    #  text is less than or equal to 5
    print(f"Removing reviews with less than {N_WORDS_THRESHOLD} words...")
    reviews_df = clean.drop_small_text(reviews_df,
                                       column_name="review_text",
                                       min_words=N_WORDS_THRESHOLD)

    # Clean the date columns in the reviews DataFrame by converting them to
    # datetime format and replacing invalid entries with appropriate default dates
    print("Cleaning date columns...")
    reviews_df = clean.convert_column_to_datetime(reviews_df,
                                                  columns=["started_at", "read_at"])

    # Compute the read duration in hours by calculating the difference between
    # the "read_at" and "started_at" datetime columns and adding a new column
    # to the DataFrame with the result
    print("Computing read duration...")
    reviews_df = utils.compute_read_duration(reviews_df,
                                             start_column="started_at",
                                             end_column="read_at")
    reviews_df.to_csv(f"data/reviews_cleaned_{lang}_{n_rows}_debug.csv", index=False)
    # Drop useless columns
    print("Dropping useless columns...")
    reviews_df = reviews_df.drop(columns=["started_at",
                                          "read_at",
                                          "user_id",
                                          "date_added",
                                          "date_updated",
                                          "n_comments"])

    print("Grouping reviews by book_id...")
    reviews_df = reviews_df.groupby("book_id").agg({
        "review_text": list,
        "rating": "mean",
        "n_votes": "sum",
        "read_duration": "median"
        }).reset_index()

    print(f"Saving cleaned reviews to {output_file}...")
    reviews_df.to_csv(output_file, index=False)

    return output_file


def process_cleaned_reviews_chunk(n_rows: str="10k", force: bool=False):
    """Split the cleaned_review dataset to smaller chunks."""

    output_file = f"data/reviews_cleaned_ENG_{n_rows}.csv"
    if os.path.exists(output_file) and not force:
        print(f"{output_file} already exists. Skipping processing.")
        return output_file

    print(f"Processing {n_rows} rows...")
    print("Reading raw reviews dataset...")
    reviews = pd.read_csv("data/reviews_cleaned_ENG_all.csv",
                            nrows=N_ROWS_MAP[n_rows])
    print("Sorting reviews by book_id...")
    reviews.sort_values(by="book_id", inplace=True)
    print("Saving reviews to CSV file...")
    reviews.to_csv(f"data/reviews_cleaned_ENG_{n_rows}.csv", index=False)
    return output_file


def extend_books_df(books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extend the books DataFrame with additional features.

    Parameters
    ----------
    books_df : pd.DataFrame
        The original books DataFrame to be extended.

    Returns
    -------
    pd.DataFrame
        The extended books DataFrame with additional features.
    """
    # Add total shelves count feature
    books_df['total_shelves_count'] = utils.parallelize_on_rows(books_df,
                                                                utils.total_shelves_count)

    # Add is_series feature
    books_df['is_series'] = utils.parallelize_on_rows(books_df,
                                                      utils.is_series)

    return books_df


def get_extended_books_df(lang: str) -> pd.DataFrame:
    """
    Get the extended books DataFrame for a specific language.

    Parameters
    ----------
    lang : str
        The language code used to filter the books (e.g., 'FRA', 'ENG', 'DUT').

    Returns
    -------
    pd.DataFrame
        The extended books DataFrame for the specified language.
    """
    books_features = ['book_id', 'text_reviews_count', 'series', 'popular_shelves',
                  'is_ebook', 'average_rating', 'similar_books', 'description',
                  'format', 'link', 'authors','publisher', 'num_pages',
                  'publication_year', 'url', 'image_url', 'book_id',
                  'ratings_count', 'title', 'title_without_series']

    books_file = f"data/raw/goodreads_books_{lang}.csv"

    books_df = pd.read_csv(books_file, usecols=books_features)
    extended_books_df = extend_books_df(books_df)
    return extended_books_df


def add_books_features(reviews_df: pd.DataFrame, books_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add features from the books DataFrame to the reviews DataFrame by merging
    them on the "book_id" column.

    Parameters
    ----------
    reviews_df : pd.DataFrame
        The reviews DataFrame containing the reviews data.
    books_df : pd.DataFrame
        The books DataFrame containing the books data.

    Returns
    -------
    pd.DataFrame
        A DataFrame resulting from merging the reviews and books DataFrames on
        the "book_id" column, with features from both DataFrames.
    """
    return reviews_df.merge(books_df, on="book_id", how="left")


def add_author_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add author names to the DataFrame by mapping author IDs to their
    corresponding names.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the reviews and books data with an 'authors' column.
    Returns
    -------
    pd.DataFrame
        A DataFrame with an additional 'author_names' column containing
        the names of the authors.
    """
    authors_df = pd.read_json("data/kaggle/goodreads_book_authors.json", lines=True)
    def get_name_by_author_id(author_id):
        return authors_df[authors_df["author_id"] == author_id]["name"].values[0]
    def get_author_names(row):
        authors = ast.literal_eval(row['authors'])
        author_names = []
        for author in authors:
            author_id = int(author['author_id'])
            author_name = get_name_by_author_id(author_id)
            author_names.append(author_name)
        return author_names
    df["author_names"] = df.apply(get_author_names, axis=1)
    # df["author_names"] = utils.parallelize_on_rows(df, get_author_names)
    df.drop(columns=["authors"], inplace=True)
    return df


@task
def process_base_df(lang: str, force: bool=False) -> str:
    """
    Process the base reviews DataFrame by language and number of rows.

    Parameters
    ----------
    lang : str
        The language code used to filter the reviews (e.g., 'FRA', 'ENG', 'DUT').

    Returns
    -------
    pd.DataFrame
        The processed base reviews DataFrame for the specified language and number of rows.
    """
    print("_" * 80)
    print(f"Processing base reviews for {lang}...")
    print("_" * 80)
    print("Reading extended books dataset...")
    books_df = None
    result = ""
    for nrows in N_ROWS_NAMES:
    # for nrows in ["10k", "100k", "all"]:
        print("_" * 80)
        print(f"Processing base reviews for {lang} with {nrows} rows...")
        print("_" * 80)
        output_file = f"data/base_{lang}_{nrows}.csv"
        result += f"{output_file}\n"
        if os.path.exists(output_file) and not force:
            print(f"{output_file} already exists. Skipping processing.")
            continue
        if books_df is None:
            print("Reading extended books dataset...")
            books_df = get_extended_books_df(lang)

        print("Reading cleaned reviews dataset...")
        reviews_file = f"data/reviews_cleaned_{lang}_all.csv"
        reviews_df = pd.read_csv(reviews_file, nrows=N_ROWS_MAP[nrows])

        print("Adding books features to reviews dataset...")
        base_df = add_books_features(reviews_df, books_df)
        print("Adding author names to base dataset...")
        base_df = add_author_names(base_df)

        cols_to_remove = ["rating",
                  "text_reviews_count",
                  "popular_shelves",
                  "is_ebook",
                  "format",
                  "link",
                  "title_without_series"]

        base_df = base_df.drop(columns=cols_to_remove)

        print(f"Saving base reviews to {output_file}...")
        base_df.to_csv(output_file, index=False)
    return result


@task
def upload_to_bigquery(lang: str):
    """
    Upload the processed base reviews DataFrame for a specific language
    and number of rows to BigQuery.
    Parameters    ----------
    lang : str
        The language code used to filter the reviews (e.g., 'FRA', 'ENG', 'DUT').
    Returns
    -------
    None
    """
    # for nrows in N_ROWS_NAMES:
    for nrows in ["all"]:
        input_file = f"data/base_{lang}_{nrows}.csv"
        base_df = pd.read_csv(input_file)

        table_name = f"base_reviews_{lang}_{nrows}"

        print(f"Uploading {input_file} to BigQuery table {table_name}...")

        bigquery.upload_dataframe_to_bigquery(df=base_df,
                                              project=GCP_PROJECT,
                                              dataset="books_dataset",
                                              table=table_name,
                                              write_mode="WRITE_TRUNCATE",
                                              chunk_size=50_000)


@flow(name=PREFECT_FLOW_NAME)
def train_flow():
    """
    Main function to process books and reviews datasets.
    """

    n_rows = "10k"
    lang = "ENG"

    print(f"Processing nrows: {n_rows}")
    print()

    # Prepare books datasets by language (ENG)
    print("Preparing books datasets by language...")
    process_all_books_task = process_books_by_lang_all.submit(force=False)
    print()

    # Prepare reviews datasets by language and number of rows
    print("Preparing reviews datasets by language and number of rows...")
    process_all_reviews_task = process_reviews_by_lang_all.submit(force=False,
                                                                  wait_for=[process_all_books_task])
    # process_reviews_chunk(n_rows=n_rows, force=False)
    print()

    # Process reviews dataset by language and number of rows
    print("Processing cleaned reviews datasets...")
    clean_reviews_all_task = clean_reviews_all.submit(force=False,
                                                      wait_for=[process_all_reviews_task])
    # process_cleaned_reviews_chunk(n_rows=n_rows, force=False)
    print()

    # Process base reviews dataset by language and number of rows
    print("Processing base reviews datasets...")
    process_base_df_task = process_base_df.submit(lang=lang,
                                                  force=False,
                                                  wait_for=[clean_reviews_all_task])
    print()

    print("All processing completed!")

    # Upload base reviews datasets to BigQuery
    # print("Uploading base reviews datasets to BigQuery...")
    # upload_to_bigquery(lang=lang)
    # print("All datasets uploaded to BigQuery!")


if __name__ == "__main__":
    train_flow()
