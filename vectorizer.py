from sklearn.feature_extraction.text import TfidfVectorizer
import gensim.downloader as api
from sentence_transformers import SentenceTransformer
import numpy as np
from typing import List, Union
import gensim


class TextVectorizer:
    def __init__(self, method: str = "tfidf"):
        """
        Initialize the TextVectorizer.

        Args:
            method (str): One of 'tfidf', 'gensim', 'sentence_transformers'
        """
        self.method = method
        self.gensim_model = None
        self.sentence_transformers_model = None
        self.tfidf_vectorizer = None

        if self.method == "gensim":
            self.gensim_model = api.load("glove-wiki-gigaword-100")
        elif self.method == "sentence_transformers":
            self.sentence_transformers_model = SentenceTransformer("all-MiniLM-L6-v2")

        elif self.method == "tfidf":
            self.tfidf_vectorizer = TfidfVectorizer()
        else:
            raise ValueError("Choose from 'tfidf', 'gensim', 'sentence_transformers'")

    def transform(self, doc: List[str] | str):
        """
        Transform a single document into a vector.

        Args:
            doc (List[str]): Tokenized document.

        Returns:
            np.ndarray or sparse matrix: Vector representation of the document.
        """
        if isinstance(doc, str):
            doc = [doc]
        text_str = " ".join(doc)
        if self.method == "gensim":
            vectors = [
                self.gensim_model[word] for word in doc if word in self.gensim_model  # type: ignore
            ]
            if len(vectors) == 0:
                return np.zeros(self.gensim_model.vector_size)  # type: ignore
            return np.mean(np.array(vectors), axis=0)
        elif self.method == "sentence_transformers":
            return self.sentence_transformers_model.encode(  # type: ignore
                text_str, show_progress_bar=False
            )
        elif self.method == "tfidf":
            return self.tfidf_vectorizer.transform([text_str])  # type: ignore
