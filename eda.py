import pandas as pd


class EDA:
    def __init__(self, dataset: pd.DataFrame):
        self.dataset = dataset

    def get_summary(self) -> pd.DataFrame:
        """
        Get a summary of the dataset including basic statistics and data types.

        Returns:
            pd.DataFrame: Summary of the dataset.
        """
        summary = self.dataset.describe(include="all").transpose()
        summary["data_type"] = self.dataset.dtypes
        return summary

    def get_missing_values(self) -> pd.Series:
        """
        Get the count of missing values in each column of the dataset.

        Returns:
            pd.Series: Count of missing values for each column.
        """
        return self.dataset.isnull().sum()

    def get_unique_values(self) -> pd.Series:
        """
        Get the count of unique values in each column of the dataset.

        Returns:
            pd.Series: Count of unique values for each column.
        """
        return self.dataset.nunique()

    def get_column_types(self) -> pd.Series:
        """
        Get the data types of each column in the dataset.

        Returns:
            pd.Series: Data types of each column.
        """
        return self.dataset.dtypes

    def get_head(self, n: int = 5) -> pd.DataFrame:
        """
        Get the first n rows of the dataset.

        Returns:
            pd.DataFrame: The first n rows of the dataset.
        """
        return self.dataset.head(n)
