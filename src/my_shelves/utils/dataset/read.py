"""
Module to retrieve data from csv
"""

import pandas as pd


def get_extended_reviews_local(data_dir: str, lang: str = 'ENG', nrows: int = None) -> pd.DataFrame:
    """
    Load an extended reviews dataset from a CSV file based on the specified language.

    Parameters
    ----------
    data_dir : str
        The data path.
    lang : str, default='ENG'
        Language code used to select the dataset file
        (e.g., 'ENG', 'FRA', 'DUT').

    nrows : int, optional
        Number of rows to read from the CSV file. If None, the entire file is loaded.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the extended reviews data.

    Notes
    -----
    The function expects the file to be located at:
    '../../data/reviews_extended_<lang>_<nrows>.csv'
    """

    file_name = f'{data_dir}/reviews_extended_{lang}_{nrows}.csv'
    if nrows is None:
        return pd.read_csv(file_name)
    return pd.read_csv(file_name, nrows=nrows)


def get_extended_reviews_gcp(lang: str = 'ENG', nrows: int = None) -> pd.DataFrame:
    from google.cloud import bigquery
    client = bigquery.Client()
    table_id = f"reviews_extended_{lang}_{nrows}"
    query = f"SELECT * FROM `{table_id}`"
    return client.query(query).to_dataframe()


def save_reviews_to_gcp(df: pd.DataFrame, lang: str = 'ENG', nrows: int = None):
    from google.cloud import bigquery
    client = bigquery.Client()
    table_id = f"reviews_extended_{lang}_{nrows}"
    job_config = bigquery.LoadJobConfig(
        write_disposition=bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    job.result()  # Wait for the job to complete
    print(f"DataFrame saved to BigQuery table {table_id}.")


def get_books(data_dir: str, lang: str = 'ENG', nrows: int = None) -> pd.DataFrame:
    """
    Load a books dataset from a CSV file based on the specified language.

    Parameters
    ----------
    data_dir : str
        The data path.
    lang : str, default='ENG'
        Language code used to select the dataset file
        (e.g., 'ENG', 'FRA', 'DUT').

    nrows : int, optional
        Number of rows to read from the CSV file. If None, the entire file is loaded.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the books data.

    Notes
    -----
    The function expects the file to be located at:
    '../../data/books_<lang>_mini.csv'
    """

    file_name = f'{data_dir}/books_{lang}_mini.csv'
    if nrows is None:
        return pd.read_csv(file_name)
    return pd.read_csv(file_name, nrows=nrows)


def get_reviews(data_dir: str, lang: str = 'ENG', nrows: int = None) -> pd.DataFrame:
    """
    Load a reviews dataset from a CSV file based on the specified language.

    Parameters
    ----------
    data_dir : str
        The data path.
    lang : str, default='ENG'
        Language code used to select the dataset file
        (e.g., 'ENG', 'FRA', 'DUT').

    nrows : int, optional
        Number of rows to read from the CSV file. If None, the entire file is loaded.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame containing the reviews data.

    Notes
    -----
    The function expects the file to be located at:
    '../../data/reviews_<lang>_mini.csv'
    """

    file_name = f'{data_dir}/reviews_{lang}_mini.csv'
    if nrows is None:
        return pd.read_csv(file_name)
    return pd.read_csv(file_name, nrows=nrows)
