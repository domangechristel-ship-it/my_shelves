""" Utility functions for processing and cleaning the books and reviews datasets. """

import pandas as pd


def add_reviews_count_words(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Add a new column to the DataFrame that counts the number of words in the review text
    for each row.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the reviews data.
    column_name : str
        The name of the column containing the review text for which to count the words.
    Returns
    -------
    pd.DataFrame
        A DataFrame with an additional column named '{column_name}_n_words'
        that contains the word count for each review."""
    df[f"{column_name}_n_words"] = df[column_name].apply(lambda x: len(x.split()))
    return df


def compute_read_duration(df: pd.DataFrame,
                          start_column: str,
                          end_column: str) -> pd.DataFrame:
    """Compute the read duration in hours by calculating the difference between
    the end and start datetime columns and adding a new column to the DataFrame with the result.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the reviews data with start and end datetime columns.
    start_column : str
        The name of the column representing the start datetime (e.g., "started_at").
    end_column : str
        The name of the column representing the end datetime (e.g., "read_at").
    Returns
    -------
    pd.DataFrame
        A DataFrame with the read duration column added.
    """
    df["read_duration"] = (df[end_column] - df[start_column]).dt.total_seconds() / 3600
    df["read_duration"] = df["read_duration"].apply(lambda x: max(x, 0))
    return df
