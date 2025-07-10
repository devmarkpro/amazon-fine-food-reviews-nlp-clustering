from dataset_download import download_dataset
import os
import pandas as pd

# logger
import logging

logger = logging.getLogger(__name__)


class Dataset:
    def __init__(self, dataset_folder_path: str = "./data"):
        # there is also a sqlite database file, but we will use the csv file
        self._dataset_file_name = "Reviews.csv"

        self._dataset_file_path = os.path.join(
            dataset_folder_path, self._dataset_file_name
        )

        self._dataset_folder_path = dataset_folder_path
        self._ensure_dataset_downloaded()
        self.reviews_df = self._load_reviews()

    def _load_reviews(self):

        if not os.path.exists(self._dataset_file_path):
            raise FileNotFoundError(
                f"{self._dataset_file_name} not found in {self._dataset_folder_path}"
            )

        logger.info("Loading reviews dataset...")
        reviews_df = pd.read_csv(self._dataset_file_path, encoding="utf-8")
        logger.info(f"Loaded {len(reviews_df)} reviews.")
        return reviews_df

    def _ensure_dataset_downloaded(self):
        logger.info("Checking if dataset is downloaded...")
        if not os.path.exists(self._dataset_folder_path):
            os.makedirs(self._dataset_folder_path)
        if os.path.exists(self._dataset_file_path):
            logger.info("Dataset already downloaded.")
            return

        # If the dataset file does not exist, download it
        logger.info("Downloading dataset...")
        download_dataset(self._dataset_folder_path)

    def load(self) -> pd.DataFrame:
        """
        Load the reviews dataset.

        Returns:
            pd.DataFrame: DataFrame containing the reviews.
        """
        return self.reviews_df
