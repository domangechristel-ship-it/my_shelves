"""
This module contains functions to process the raw datasets of books and reviews
by language and number of rows.
"""

import os
import csv
import pandas as pd

# Package imports
import read
import my_shelves.utils.dataset.prepare.clean as clean
import my_shelves.utils.dataset.prepare.utils as utils


DATA_DIR = "data"
N_ROWS = None
N_WORDS_THRESHOLD = 50
N_ROWS_MAP = {"10k": 10_000,
              "100k": 100_000,
              "1M": 1_000_000,
              "all": None
}


def process_books_by_lang() -> str:
    """Process the raw books dataset by language and save the results in separate
    CSV files for each language (French, English, Dutch). The function reads the
    raw books dataset from a JSON file, filters the books by language, and saves
    the results in separate CSV files for each language. The function also counts
    the number of books for each language and prints the results.

    Returns
    -------
    str
        The paths to the processed books CSV files for each language.
    """
    json_file = "data/raw/goodreads_books.json"

    fra_file = "data/raw/goodreads_books_FRA.csv"
    eng_file = "data/raw/goodreads_books_ENG.csv"
    dut_file = "data/raw/goodreads_books_DUT.csv"

    if all([os.path.exists(f) for f in [fra_file, eng_file, dut_file]]):
        print("Books files already exist. Skipping processing.")
        return

    chunksize = 10000
    headers_written = False
    fra = 0
    eng = 0
    dut = 0

    with open(fra_file, "w", newline="", encoding="utf-8") as fra_f, \
         open(eng_file, "w", newline="", encoding="utf-8") as eng_f, \
         open(dut_file, "w", newline="", encoding="utf-8") as dut_f:

        fra_writer = None
        eng_writer = None
        dut_writer = None

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
                    fra_writer = csv.DictWriter(fra_f, fieldnames=fieldnames)
                    eng_writer = csv.DictWriter(eng_f, fieldnames=fieldnames)
                    dut_writer = csv.DictWriter(dut_f, fieldnames=fieldnames)

                    fra_writer.writeheader()
                    eng_writer.writeheader()
                    dut_writer.writeheader()

                    headers_written = True

                if lang == "fre":
                    fra_writer.writerow(row_dict)
                    fra = fra + 1
                elif lang.startswith("en"):
                    eng_writer.writerow(row_dict)
                    eng = eng + 1
                elif lang == "nl":
                    dut_writer.writerow(row_dict)
                    dut = dut + 1

    print("Done.")
    print(f"fra:{fra}")
    print(f"eng:{eng}")
    print(f"dut:{dut}")
    return fra_file, eng_file, dut_file


def process_reviews_by_lang(lang: str, nrows: str="all") -> str:
    """Process the raw reviews dataset by language and number of rows and save
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

    Returns
    -------
    str
        The path to the processed reviews CSV file for the specified language
        and number of rows.
    """
    print("_" * 80)
    print(f"Processing {lang} reviews with {nrows} rows...")
    print("_" * 80)

    json_file = "data/raw/goodreads_reviews_dedup.json"
    books_file = f'data/raw/goodreads_books_{lang}.csv'
    output_file = f"data/goodreads_reviews_{lang}_{nrows}.csv"

    if os.path.exists(output_file):
        print(f"{output_file} already exists. Skipping processing.")
        return output_file

    books = pd.read_csv(books_file)
    book_ids = set(books['book_id'])

    # to delete books and released memory
    books = pd.DataFrame()
    del books

    chunksize = 10000
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
                    if nrows != "all" and n_reviews >= N_ROWS_MAP[nrows]:
                        break
            if nrows != "all" and n_reviews >= N_ROWS_MAP[nrows]:
                break

    print("Done.")
    print(f"{lang} reviews count:{n_reviews}")
    return output_file


def process_reviews(lang: str, nrows: str="all", force: bool=False) -> str:
    """Process the raw reviews dataset by language and number of rows and
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
    # Get reviews raw dataset
    # reviews_df = read.get_reviews(DATA_DIR, lang='ENG', nrows=nrows)
    print("_" * 80)
    print(f"Processing cleaned reviews for {lang} with {nrows} rows...")
    print("_" * 80)

    output_file = f"{DATA_DIR}/reviews_cleaned_{lang}_{nrows}.csv"

    if os.path.exists(output_file) and not force:
        print(f"{output_file} already exists. Skipping processing.")
        return pd.read_csv(output_file)

    # Read the reviews dataset for the specified language and number of rows
    print(f"Reading reviews dataset for {lang} with {nrows} rows...")
    reviews_df = pd.read_csv(f"{DATA_DIR}/goodreads_reviews_{lang}_{nrows}.csv")
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
                                                  column_name="started_at",
                                                  replacement_value="futur")
    reviews_df = clean.convert_column_to_datetime(reviews_df,
                                                  column_name="read_at",
                                                  replacement_value="past")

    # Compute the read duration in hours by calculating the difference between
    # the "read_at" and "started_at" datetime columns and adding a new column
    # to the DataFrame with the result
    print("Computing read duration...")
    reviews_df = utils.compute_read_duration(reviews_df,
                                             start_column="started_at",
                                             end_column="read_at")

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
        "read_duration": "mean"
        }).reset_index()

    print(f"Saving cleaned reviews to {output_file}...")
    reviews_df.to_csv(output_file, index=False)

    return output_file


if __name__ == "__main__":
    # Prepare books datasets by language
    process_books_by_lang()

    # Prepare reviews datasets by language and number of rows
    # for lang in ["FRA", "ENG", "DUT"]:
    for lang in ["ENG"]:
        # for nrows in N_ROWS_MAP.keys():
        for nrows in ["all"]:
            process_reviews_by_lang(lang=lang, nrows=nrows)

    # Process reviews dataset by language and number of rows
    # for lang in ["FRA", "ENG", "DUT"]:
    for lang in ["ENG"]:
        for nrows in N_ROWS_MAP.keys():
            process_reviews(lang=lang, nrows=nrows)
