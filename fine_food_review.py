from dataset import Dataset
from eda import EDA
import pandas as pd
from typing import Dict, Literal
import matplotlib.pyplot as plt
import seaborn as sns


class FineFoodReview:
    def __init__(self, dataset_path: str = "./data"):
        self.dataset = Dataset(dataset_folder_path=dataset_path).load()
        self.eda = EDA(self.dataset.copy(deep=True))

        self.stats: Dict[str, dict] = {}
        self.grouped_by_user: pd.Series[int] = pd.Series(dtype="int64")
        self.grouped_by_product: pd.Series[int] = pd.Series(dtype="int64")
        self.num_users: int = 0
        self.num_products: int = 0
        self.num_reviews: int = 0

        self._set_values()

    def _set_values(self):
        self.num_users = self.dataset["UserId"].nunique()
        self.num_products = self.dataset["ProductId"].nunique()
        self.num_reviews = self.dataset.shape[0]

        self.grouped_by_product = (
            self.dataset.groupby("ProductId").size().sort_values(ascending=False)
        )

        self.grouped_by_user = (
            self.dataset.groupby("UserId").size().sort_values(ascending=False)
        )
        self.stats = {
            "user": self.grouped_by_user.describe().to_dict(),
            "product": self.grouped_by_product.describe().to_dict(),
        }
