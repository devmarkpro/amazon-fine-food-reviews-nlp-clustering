from dataset import Dataset
from eda import EDA
import pandas as pd
from typing import Dict, Literal
import matplotlib.pyplot as plt
import seaborn as sns
from word_tokenizer import WordTokenizer
import logging
from vectorizer import TextVectorizer
import numpy as np

logger = logging.getLogger(__name__)


class FineFoodReview:
    def __init__(
        self,
        dataset_path: str = "./data",
        dataset_file_name: str = "fine_food_reviews.csv",
    ):
        self.dataset = Dataset(
            dataset_folder_path=dataset_path, dataset_file_name=dataset_file_name
        ).load()
        self.eda = EDA(self.dataset.copy(deep=True))

        self.stats: Dict[str, dict] = {}
        self.grouped_by_user: pd.Series[int] = pd.Series(dtype="int64")
        self.grouped_by_product: pd.Series[int] = pd.Series(dtype="int64")
        self.num_users: int = 0
        self.num_products: int = 0
        self.num_reviews: int = 0

        self._set_values()

        self.tokenizer = WordTokenizer(
            remove_stopwords=True,
            lower_case=True,
            use_lemmatization=True,
        )

    def _set_values(self) -> None:
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

    def tokenize_reviews(self, force: bool = False) -> list[str]:
        """
        Tokenizes the reviews in the dataset using the WordTokenizer.
        Adds a new column 'TokenizedText' to the dataset containing the tokenized reviews.
        This method processes each review in the 'Text' column of the dataset and applies the tokenizer.
        Returns:
            A list of tokenized reviews.
        """

        logger.info(f"Tokenizing {len(self.dataset)} reviews...")

        if "TokenizedText" in self.dataset.columns and not force:
            logger.info("TokenizedText column already exists, skipping tokenization.")
            return self.dataset["TokenizedText"].to_list()

        self.dataset["TokenizedText"] = self.dataset["Text"].apply(
            lambda x: self.tokenizer(x, return_tokens=True)
        )
        return (
            self.dataset["TokenizedText"]
            .apply(lambda tokens: " ".join(tokens))
            .to_list()
        )

    def vectorize_reviews(
        self, method: Literal["tfidf", "gensim", "sentence_transformers"]
    ) -> None:
        """
        Vectorizes the tokenized reviews using the specified vectorization method.
        """
        self.vectorizer = TextVectorizer(method=method)

        # If no TokenizedText yet, tokenize
        if "TokenizedText" not in self.dataset.columns:
            logger.info("TokenizedText column not found, tokenizing reviews...")
            self.tokenize_reviews()

        logger.info(f"Vectorizing reviews using {self.vectorizer.method}...")

        if self.vectorizer.method == "tfidf":
            # Fit on entire corpus
            self.vectorizer.fit(self.dataset["TokenizedText"])
            # Transform each
            vectors = [
                self.vectorizer.transform(doc).toarray()[0]
                for doc in self.dataset["TokenizedText"]
            ]
        else:
            vectors = [
                self.vectorizer.transform(doc) for doc in self.dataset["TokenizedText"]
            ]

        self.X_vectorized = np.vstack(vectors)
        logger.info(f"Vectorization complete. Shape: {self.X_vectorized.shape}")
        self.dataset[f"ReviewVector_{method}"] = list(self.X_vectorized)
        return self.X_vectorized
