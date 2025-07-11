import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import logging

logger = logging.getLogger(__name__)


class USLClustering:
    """
    Unsupervised Learning Clustering with hyperparameter tuning using sklearn.
    Supports KMeans, AgglomerativeClustering, and DBSCAN.
    """

    def __init__(self, method="kmeans", param_grid=None, random_state=42):
        self.method = method.lower()
        self.param_grid = param_grid or self._default_param_grid()
        self.random_state = random_state
        self.best_model = None
        self.best_params = None
        self.best_score = None

    def _default_param_grid(self):
        if self.method == "kmeans":
            return {"n_clusters": [5, 10, 15], "n_init": [10, 20]}
        elif self.method == "agglomerative":
            return {"n_clusters": [5, 10, 15], "linkage": ["ward", "average"]}
        elif self.method == "dbscan":
            return {"eps": [0.5, 1.0, 1.5], "min_samples": [5, 10]}
        else:
            raise ValueError(f"Unknown clustering method: {self.method}")

    def fit(self, X):
        """
        Fit clustering model with hyperparameter tuning using silhouette score.
        """
        best_score = -1
        best_model = None
        best_params = None
        grid = list(ParameterGrid(self.param_grid))
        logger.info(f"Tuning {self.method} over {len(grid)} parameter combinations...")
        for params in grid:
            try:
                if self.method == "kmeans":
                    model = KMeans(random_state=self.random_state, **params)
                elif self.method == "agglomerative":
                    model = AgglomerativeClustering(**params)
                elif self.method == "dbscan":
                    model = DBSCAN(**params)
                else:
                    continue
                labels = model.fit_predict(X)
                # DBSCAN may assign -1 for noise, skip if only one cluster
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                if n_clusters < 2:
                    continue
                score = silhouette_score(X, labels, random_state=self.random_state)
                logger.debug(f"Params: {params}, Silhouette Score: {score:.4f}")
                if score > best_score:
                    best_score = score
                    best_model = model
                    best_params = params
            except Exception as e:
                logger.warning(f"Failed for params {params}: {e}")
                continue
        self.best_model = best_model
        self.best_params = best_params
        self.best_score = best_score
        logger.info(
            f"Best params: {best_params}, Best Silhouette Score: {best_score:.4f}"
        )
        return self

    def predict(self, X):
        """
        Predict cluster labels using the best model.
        """
        if self.best_model is None:
            raise ValueError("Model not fitted. Call fit(X) first.")
        # Only KMeans supports .predict; others use .fit_predict
        if isinstance(self.best_model, KMeans):
            return self.best_model.predict(X)
        else:
            return self.best_model.fit_predict(X)

    def get_best_params(self):
        return self.best_params

    def get_best_score(self):
        return self.best_score
