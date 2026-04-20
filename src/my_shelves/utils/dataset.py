"""
Module to retrieve data from csv
"""
import pandas as pd


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
