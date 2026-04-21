"""
Utilities for safely appending data to CSV files.

This module provides helper functions to write pandas DataFrames to CSV files
while avoiding duplicate entries based on a specified key column (e.g. 'book_id').

Typical use case
----------------
This utility is useful in data pipelines where data is incrementally collected
(e.g. from APIs, JSON streams, or batch processing) and needs to be stored
in a CSV file without introducing duplicate records.

Main features
-------------
- Append new rows to an existing CSV file
- Prevent duplicates based on a unique identifier column
- Efficient handling of large CSV files by only reading necessary columns
- Automatic file creation if the target CSV does not exist

Example
-------
>>> import pandas as pd
>>> from utils.csv_writer import append_unique_rows
>>>
>>> df = pd.DataFrame({
...     "book_id": [1, 2, 3],
...     "title": ["Book A", "Book B", "Book C"]
... })
>>>
>>> append_unique_rows(df, "books.csv")


"""
import pandas as pd
import os

def append_unique_rows(df: pd.DataFrame, csv_file: str, key: str = "book_id"):
    """
    Append rows from a DataFrame to a CSV file, avoiding duplicates based on a key column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing new rows to append.

    csv_file : str
        Path to the CSV file.

    key : str, default='book_id'
        Column used to detect duplicates.

    Returns
    -------
    None
    """

    # Si le fichier n'existe pas → on écrit directement tout
    if not os.path.isfile(csv_file):
        df.to_csv(csv_file, index=False)
        print(f"fichier créer et {len(df)} lignes ajoutées.")
        return

    # Lire uniquement la colonne clé existante (plus rapide)
    existing = pd.read_csv(csv_file, usecols=[key])

    # Filtrer les nouvelles lignes
    df_new = df[~df[key].isin(existing[key])]

    # Si rien à ajouter → stop
    if df_new.empty:
        print("Aucune nouvelle ligne à ajouter.")
        return

    # Append sans header
    df_new.to_csv(csv_file, mode='a', header=False, index=False)

    print(f"{len(df_new)} lignes ajoutées.")
