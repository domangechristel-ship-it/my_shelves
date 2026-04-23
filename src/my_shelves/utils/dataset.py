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


def clean_reviews_text(df: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """
    Clean the reviews text by removing rows with missing or empty values
    in the specified column.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the reviews data.
    column_name : str
        The name of the column containing the review text to be cleaned.
    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with rows containing missing or empty review text removed."""
    df = df.dropna(subset=[column_name])
    df = df[df[column_name].str.strip() != ""]
    return df


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


def clean_small_reviews(df: pd.DataFrame,
                        column_name: str,
                        min_words: int) -> pd.DataFrame:
    """
    Clean the reviews by removing rows where the number of words in the review text
    is less than or equal to a specified minimum threshold.
    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing the reviews data.
    column_name : str
        The name of the column containing the review text for which to count the words.
    min_words : int
        The minimum number of words required for a review to be retained in the DataFrame.
    Returns
    -------
    pd.DataFrame
        A cleaned DataFrame with rows containing reviews that have a word count less than
        or equal to the specified minimum threshold removed."""
    df = df[df[f"{column_name}_n_words"] > min_words]
    return df


def convert_column_to_datetime(df: pd.DataFrame,
                               column_name: str,
                               replacement_value: str="futur") -> pd.DataFrame:
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
    if replacement_value == "futur":
        replacement_date = 'Sun Jul 1 00:00:00 -0700 2050'
    df[column_name] = pd.to_datetime(df[column_name],
                                     format='%a %b %d %H:%M:%S %z %Y',
                                     utc=True,
                                     errors='coerce')
    df[column_name] = df[column_name].fillna(value=replacement_date)
    return df


def clean_reviews_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Clean the date columns in the reviews DataFrame by converting them to
    datetime format and replacing invalid entries with appropriate default dates
    based on whether they represent past or future events.
    Parameters    ----------
    df : pd.DataFrame
        The DataFrame containing the reviews data with date columns to be cleaned.
    Returns
    -------
    pd.DataFrame
        A DataFrame with the date columns cleaned and converted to datetime
        format, with invalid entries replaced by default dates corresponding to
        past or future events as appropriate."""

    date_columns = {"started_at": "past",
                    "read_at": "futur",
                    "date_added": "futur",
                    "date_updated": "futur"}

    for column, replacement_value in date_columns.items():
        df = convert_column_to_datetime(df,
                                        column_name=column,
                                        replacement_value=replacement_value)
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
    return df


if __name__ == "__main__":
    DATA_DIR = "../../data"
    books_df = get_books(DATA_DIR, lang='ENG', nrows=1000)
    reviews_df = get_reviews(DATA_DIR, lang='ENG', nrows=1000)
    print(books_df.head())
    print(reviews_df.head())
    # Clean the reviews text by removing rows with missing or empty review text
    reviews_df = clean_reviews_text(reviews_df, column_name="review_text")
    # Add a new column that counts the number of words in the review text
    reviews_df = add_reviews_count_words(reviews_df, column_name="review_text")
    # Clean the reviews by removing rows where the number of words in the review
    #  text is less than or equal to 5
    reviews_df = clean_small_reviews(reviews_df,
                                     column_name="review_text",
                                     min_words=5)
    # Clean the date columns in the reviews DataFrame by converting them to
    # datetime format and replacing invalid entries with appropriate default dates
    reviews_df = clean_reviews_dates(reviews_df)
    # Compute the read duration in hours by calculating the difference between
    # the "read_at" and "started_at" datetime columns and adding a new column
    # to the DataFrame with the result
    reviews_df = compute_read_duration(reviews_df,
                                       start_column="started_at",
                                       end_column="read_at")
    print(reviews_df.head())
