"""
Module for interacting with BigQuery to retrieve book similarities.
"""

import ast
import logging
from google.cloud import bigquery


def get_book_similarity(book_id: int, model_name: str, n_rows: str) -> list[int]:
    """
    Retrieves the list of similar book IDs from a BigQuery table.

    Args:
        book_id: The ID of the book to find similarities for.
        model_name: The name of the similarity model used.
        n_rows: The dataset size (e.g., '10k', '20k').

    Returns:
        A list of similar book IDs.
    """
    client = bigquery.Client()
    table_id = f"books_dataset.similarities_{model_name}_{n_rows}"

    # Using a parameterized query to ensure security and efficiency
    query = f"SELECT similar_books FROM `{table_id}` WHERE book_id = @book_id"
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("book_id", "INT64", book_id),
        ]
    )

    try:
        query_job = client.query(query, job_config=job_config)
        results = query_job.result()

        for row in results:
            res = row.similar_books
            return ast.literal_eval(res) if isinstance(res, str) else list(res)
    except Exception as e:
        logging.error(f"BigQuery query failed for {table_id}: {e}")

    return []
