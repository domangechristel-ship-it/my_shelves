from google.cloud import bigquery
import pandas as pd

def get_book(book_id: int) -> pd.DataFrame:
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

    client = bigquery.Client(project="my-shelves-493916")

    query = """
        SELECT *
        FROM my-shelves-493916.books_dataset.books
        WHERE book_id = @book_id
    """

    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("book_id", "INT64", book_id)
        ]
    )

    df = client.query(query, job_config=job_config).to_dataframe()

    return df
