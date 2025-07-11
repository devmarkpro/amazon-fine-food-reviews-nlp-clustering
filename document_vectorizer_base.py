from abc import ABC, abstractmethod
from typing import List


class DocumentVectorizerBase(ABC):

    def __init__(self):
        """
        Base class for document vectorizers. This class should be inherited by specific vectorizer implementations.
        """
        self.name = "Base Document Vectorizer"

    @abstractmethod
    def fit(self, documents: List[str]) -> None:
        """
        Fit the vectorizer to the provided documents.

        Args:
            documents (List[str]): List of documents to fit the vectorizer.
        """
        pass

    @abstractmethod
    def transform(self, documents: List[str]) -> List[List[float]]:
        """
        Transform the provided documents into their vector representations.

        Args:
            documents (List[str]): List of documents to transform.

        Returns:
            List[List[float]]: List of document vectors.
        """
        pass
