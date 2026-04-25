""" Text cleaning utilities for processing review data. """
import pandas as pd


def clean_text(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Clean the text by removing rows with missing or empty values
    in the specified column.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the text data.
    column_name : str
        The name of the column containing the text to be cleaned.
    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with rows containing missing or empty text removed."""
    df = df.dropna(subset=[column_name])
    df = df[df[column_name].str.strip() != ""]
    return df


def drop_small_text(df: pd.DataFrame,
                     column_name: str,
                     min_words: int) -> pd.DataFrame:
    """
    Clean the text by removing rows where the number of words in the text
    is less than or equal to a specified minimum threshold.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the reviews data.
    column_name : str
        The name of the column containing the text for which to count the words.
    min_words : int
        The minimum number of words required for a text to be retained in the DataFrame.
    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with rows containing text that have a word count less than
        or equal to the specified minimum threshold removed."""
    df = df[df[f"{column_name}_n_words"] > min_words]
    return df


def convert_column_to_datetime(df: pd.DataFrame,
                               columns: list[str]) -> pd.DataFrame:
    """Convert a specified column in the DataFrame to datetime format, handling errors by replacing
    invalid entries with a default date based on the provided replacement value.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the data to be converted.
    column_name : str
        The name of the column to be converted to datetime format.
    replacement_value : str, default="futur"
        The value to use for replacing invalid datetime entries.
        If "futur", invalid entries will be replaced with 'Sun Jul 1 00:00:00 -0700 2050'.
        If "past", invalid entries will be replaced with 'Sun Jul 1 00:00:00 -0700 1970'.
    Returns
    -------
    pd.DataFrame
        A DataFrame with the specified column converted to datetime format and
        invalid entries replaced with the appropriate default date.
    """
    replacement_date = 'Sun Jul 1 00:00:00 -0700 1970'
    for column_name in columns:
        df[column_name] = pd.to_datetime(df[column_name],
                                         format='%a %b %d %H:%M:%S %z %Y',
                                         utc=True,
                                         errors='coerce')
        df[column_name] = df[column_name].fillna(value=replacement_date)
    return df
