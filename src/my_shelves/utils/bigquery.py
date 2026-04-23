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

    df = get_book(6668764)
"""
import streamlit as st
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
    full_table_name = "books_dataset.base_reviews_ENG_1M"

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

@st.cache_data(ttl=3600)  # cache for 1 hour
def get_country_counts() -> pd.DataFrame:
    """
    Retrieve country-level book counts from BigQuery.

    Returns
    -------
    pd.DataFrame
        DataFrame containing country counts data.
    """

    client = bigquery.Client()

    full_table_name = "books_dataset.country_counts"

    query = f"""
        SELECT *
        FROM `{full_table_name}`
    """

    df = client.query(query).to_dataframe()
    return df


def get_books(book_id_list: list[int], nbr_rows: int = 10) -> pd.DataFrame:
    """
    Retrieve books from BigQuery based on a list of book_ids.

    Parameters
    ----------
    book_id_list : list[int]
        List of book IDs to retrieve.
    nbr_rows : int, optional
        Maximum number of rows to return. If None, returns all matching rows.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the matching books.
    """

    client = bigquery.Client()
    full_table_name = "books_dataset.base_reviews_ENG_1M"

    query = f"""
        SELECT *
        FROM `{full_table_name}`
        WHERE book_id IN UNNEST(@book_id_list)
    """

    if nbr_rows is not None:
        query += f"\nLIMIT {nbr_rows}"

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ArrayQueryParameter("book_id_list", "INT64", book_id_list)
        ]
    )

    df = client.query(query, job_config=job_config).to_dataframe()

    return df

def get_id_by_country(country: str) -> list[int]:
    """
    Retrieve book IDs for a given country from BigQuery.

    Parameters
    ----------
    country : str
        Name of the country (case-insensitive).

    Returns
    -------
    list[int]
        List of book_id values corresponding to the given country.
    """

    client = bigquery.Client()

    query = """
        SELECT DISTINCT book_id
        FROM `books_dataset.book_locations`
        WHERE LOWER(country) = LOWER(@country)
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("country", "STRING", country)
        ]
    )

    df = client.query(query, job_config=job_config).to_dataframe()

    # Return as a clean Python list
    return df["book_id"].dropna().astype(int).tolist()
