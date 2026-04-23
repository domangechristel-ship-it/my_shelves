""" Utility functions for processing and cleaning the books and reviews datasets. """
from multiprocessing import  Pool
from functools import partial
import ast

import numpy as np
import pandas as pd


# Parallelization utilities inspired by
# https://towardsdatascience.com/parallelize-your-pandas-code-1ff9c8e1aee0
def parallelize(data, func, num_of_processes=8):
    """
    Parallelize the execution of a function on a DataFrame using multiprocessing.
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to process in parallel.
    func : function
        The function to apply to each subset of the DataFrame.
    num_of_processes : int, optional
        The number of processes to use for parallelization (default is 8).
    Returns
    -------
    pd.DataFrame
        The DataFrame resulting from applying the function in parallel.
    """
    data_split = np.array_split(data, num_of_processes)
    pool = Pool(num_of_processes) # pylint: disable=R1732
    data = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def run_on_subset(func, data_subset):
    """
    Helper function to apply a given function to a subset of the DataFrame.
    Parameters
    ----------
    func : function
        The function to apply to the subset of the DataFrame.
    data_subset : pd.DataFrame
        The subset of the DataFrame to process.
    Returns
    -------
    pd.DataFrame
        The DataFrame resulting from applying the function to the subset.
    """
    return data_subset.apply(func, axis=1)


def parallelize_on_rows(data, func, num_of_processes=8):
    """
    Parallelize the execution of a function on each row of a DataFrame
    using multiprocessing.
    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to process in parallel.
    func : function
        The function to apply to each row of the DataFrame.
    num_of_processes : int, optional
        The number of processes to use for parallelization (default is 8).
    Returns
    -------
    pd.DataFrame
        The DataFrame resulting from applying the function to each row in parallel.
    """
    return parallelize(data, partial(run_on_subset, func), num_of_processes)


def total_shelves_count(row):
    """
    Calculate the total count of shelves for a given row.
    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing the 'popular_shelves' column.
    Returns
    -------
    int
        The total count of shelves.
    """
    return sum(int(item['count']) for item in ast.literal_eval(row['popular_shelves']))


def is_series(row):
    """
    Determine if a given row represents a series.
    Parameters
    ----------
    row : pd.Series
        A row from the DataFrame containing the 'series' column.
    Returns
    -------
    str
        'Y' if the row represents a series, 'N' otherwise.
    """
    return 'Y' if len(ast.literal_eval(row['series'])) > 0 else 'N'


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
