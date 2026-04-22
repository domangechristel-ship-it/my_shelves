
from importlib import abc

import pandas as pd


class Reader(abc.ABC):
    """Abstract base class for dataset readers."""

    @abc.abstractmethod
    def get_books(self, data_dir: str, lang: str = 'ENG', nrows: int = None) -> pd.DataFrame:
        """Load a books dataset from a CSV file based on the specified language."""
        pass

    @abc.abstractmethod
    def get_reviews(self, data_dir: str, lang: str = 'ENG', nrows: int = None) -> pd.DataFrame:
        """Load a reviews dataset from a CSV file based on the specified language."""
        pass


class CSVReader(Reader):
    """Concrete implementation of the Reader class for reading datasets from CSV files."""

    def get_books(self, data_dir: str, lang: str = 'ENG', nrows: int = None) -> pd.DataFrame:
        file_name = f'{data_dir}/books_{lang}_mini.csv'
        if nrows is None:
            return pd.read_csv(file_name)
        return pd.read_csv(file_name, nrows=nrows)

    def get_reviews(self, data_dir: str, lang: str = 'ENG', nrows: int = None) -> pd.DataFrame:
        file_name = f'{data_dir}/reviews_{lang}_mini.csv'
        if nrows is None:
            return pd.read_csv(file_name)
        return pd.read_csv(file_name, nrows=nrows)


class BigQueryReader(Reader):
    """Concrete implementation of the Reader class for reading datasets from Google BigQuery. This class is a placeholder and should be implemented with actual logic to connect to BigQuery and retrieve the datasets."""

    def get_books(self, data_dir: str, lang: str = 'ENG', nrows: int = None) -> pd.DataFrame:
        pass
