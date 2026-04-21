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

    df = get_book(22077083,GCP_PROJECT,BQ_DATASET)
"""
import pandas as pd
from google.cloud import bigquery

def get_book(
        book_id: int,
        gcp_project:str,
        bq_dataset:str
    ) -> pd.DataFrame:
    """
    Retrieve a book information from BigQuery based on its book_id.

    Parameters
    ----------
    book_id : int
        The ID of the book to retrieve.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the matching book row(s).
    """

    client = bigquery.Client()
    table = "books"
    full_table_name = f"{gcp_project}.{bq_dataset}.{table}"
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
