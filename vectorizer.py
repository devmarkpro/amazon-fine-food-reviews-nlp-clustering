import gensim.downloader as api
import numpy as np
from typing import List, Union
import gensim
import logging
from document_vectorizer_base import DocumentVectorizerBase
from typing import Literal

logger = logging.getLogger(__name__)
self.model = api.load(f"glove-wiki-gigaword-100")
class GloVeVectorizer():
    def __init__(self):
        """
        Initialize the GloVe vectorizer.

        Args:
            dim (int): Dimension of the GloVe vectors. Default is 100.
        """
        super().__init__()
        self.name = "GloVe Vectorizer"
        

    def fit(self, documents) -> None:
        # GloVe is pre-trained, so no fitting is needed
        pass

    def transform(self, documents: List[str]):
        
        
    
    def _get_document_vector(tokens, embedding_index, dim=100):
        vecs = []
        for token in tokens:
            if token in embedding_index:
                vecs.append(embedding_index[token])
        if len(vecs) > 0:
            return np.mean(vecs, axis=0)
        else:
            return np.zeros(dim)    


class TextVectorizer:
    def __init__(self, method: Literal["tfidf", "glove"]):
        """
        Initialize the TextVectorizer.

        Args:
            method (str): One of 'tfidf', 'glove', or 'word2vec'.
                - 'tfidf': Uses TF-IDF vectorization.
                - 'glove': Uses GloVe embeddings from Gensim.
        """

    def transform(self, docs):
        pass

    def get_vectorizer_name(self):
        pass
