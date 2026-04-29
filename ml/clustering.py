"""
ml/clustering.py

K-means clustering implemented from scratch using NumPy only.
No sklearn KMeans -- this satisfies the course algorithm implementation requirement.

Groups LinkedIn job postings by skill profile so students can explore
role families rather than individual listings.

"""

import numpy as np
import pickle
from pathlib import Path


class KMeans:
    """
    K-means clustering from scratch using NumPy.

    Attributes:
        k: Number of clusters
        max_iters: Maximum number of update iterations
        tol: Convergence tolerance (minimum centroid shift to continue)
        centroids: Cluster centroids after fitting, shape (k, n_features)
        labels: Cluster assignment per sample after fitting, shape (n_samples,)
    """

    def __init__(self, k: int = 8, max_iters: int = 300, tol: float = 1e-4):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.centroids = None
        self.labels = None

    def _init_centroids(self, X: np.ndarray) -> np.ndarray:
        """
        Randomly initialize k centroids by sampling rows from X without replacement.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Initial centroids of shape (k, n_features)
        """
        rng = np.random.default_rng(seed=42)
        indices = rng.choice(X.shape[0], size=self.k, replace=False)
        return X[indices].copy()

    def _compute_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance from each sample to each centroid.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Distance matrix of shape (n_samples, k)
        """
        # using broadcasting: (n_samples, 1, n_features) - (k, n_features)
        diff = X[:, np.newaxis, :] - self.centroids[np.newaxis, :, :]
        return np.sqrt((diff ** 2).sum(axis=2))

    def _assign_labels(self, X: np.ndarray) -> np.ndarray:
        """
        Assign each sample to its nearest centroid.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Cluster labels of shape (n_samples,)
        """
        distances = self._compute_distances(X)
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X: np.ndarray, labels: np.ndarray) -> np.ndarray:
        """
        Recompute centroids as the mean of all samples assigned to each cluster.
        If a cluster is empty, reinitialize its centroid to a random sample.

        Args:
            X: Data matrix of shape (n_samples, n_features)
            labels: Current cluster assignments of shape (n_samples,)

        Returns:
            Updated centroids of shape (k, n_features)
        """
        new_centroids = np.zeros((self.k, X.shape[1]))
        for cluster_idx in range(self.k):
            members = X[labels == cluster_idx]
            if len(members) == 0:
                # reinitialize empty cluster to a random sample
                new_centroids[cluster_idx] = X[np.random.randint(0, X.shape[0])]
            else:
                new_centroids[cluster_idx] = members.mean(axis=0)
        return new_centroids

    def fit(self, X: np.ndarray) -> "KMeans":
        """
        Run K-means until convergence or max_iters is reached.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            self, with centroids and labels populated
        """
        self.centroids = self._init_centroids(X)

        for iteration in range(self.max_iters):
            new_labels = self._assign_labels(X)
            new_centroids = self._update_centroids(X, new_labels)

            # check convergence: max shift across all centroids
            centroid_shift = np.linalg.norm(new_centroids - self.centroids, axis=1).max()

            self.centroids = new_centroids
            self.labels = new_labels

            if centroid_shift < self.tol:
                print(f"Converged at iteration {iteration + 1}")
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Assign new samples to the nearest fitted centroid.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Cluster labels of shape (n_samples,)
        """
        if self.centroids is None:
            raise ValueError("Model has not been fitted yet. Call fit() first.")
        return self._assign_labels(X)

    def inertia(self, X: np.ndarray) -> float:
        """
        Compute inertia (sum of squared distances to nearest centroid).
        Useful for the elbow method to choose optimal k.

        Args:
            X: Data matrix of shape (n_samples, n_features)

        Returns:
            Inertia score (lower is better)
        """
        distances = self._compute_distances(X)
        nearest = distances[np.arange(X.shape[0]), self._assign_labels(X)]
        return float((nearest ** 2).sum())

    def save(self, path: Path) -> None:
        """Save fitted model to disk as a pickle file."""
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: Path) -> "KMeans":
        """Load a fitted KMeans model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
