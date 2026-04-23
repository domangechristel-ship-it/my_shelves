"""
bigquery.py
-----------
This module provides functions to interact with the BigQuery database
for the my-shelves project.

It handles all data retrieval and querying operations against the
BigQuery tables, including fetching books and other related data.

Usage
-----
    from bigquery import get_book

    df = get_book(22077083)
"""
import pandas as pd
from google.cloud import bigquery


def get_book(book_id: int) -> pd.DataFrame:
    """
    Retrieve book information from BigQuery based on its book_id.

    Parameters
    ----------
    book_id : int
        The ID of the book to retrieve.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the matching book row(s).
        Returns an empty DataFrame if no book is found.
    """
    client = bigquery.Client()
    full_table_name = "books_dataset.books"

    query = f"""
        SELECT *
        FROM {full_table_name}
        WHERE book_id = @book_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("book_id", "INT64", book_id)
        ]
    )

    df = client.query(query, job_config=job_config).to_dataframe()
    return df


def upload_dataframe_to_bigquery(df: pd.DataFrame,
                                 project: str,
                                 dataset: str,
                                 table: str,
                                 write_mode: str = "WRITE_TRUNCATE") -> str:
    """
    Upload a DataFrame to a BigQuery table.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to upload.
    project : str
        The GCP project ID.
    dataset : str
        The BigQuery dataset name.
    table : str
        The BigQuery table name.
    write_mode : str, optional
        The write disposition mode (default is "WRITE_TRUNCATE").
        Options include "WRITE_TRUNCATE", "WRITE_APPEND", and "WRITE_EMPTY".

    Returns
    -------
    str
        A message indicating the success of the upload operation.
    """
    client = bigquery.Client(project=project)
    full_table_name = f"{project}.{dataset}.{table}"

    job_config = bigquery.LoadJobConfig(write_disposition=write_mode)
    job = client.load_table_from_dataframe(df, full_table_name, job_config=job_config)
    result = job.result()  # Wait for the job to complete
    print(f"Data uploaded to {full_table_name} with write mode {write_mode}.")
    return result
