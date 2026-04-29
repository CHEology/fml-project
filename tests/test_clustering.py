"""
tests/test_clustering.py

Unit tests for ml/clustering.py.
Verifies K-means cluster assignment, centroid shape, and convergence.

Run with: pytest tests/test_clustering.py

Owner: @trp8625
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parent.parent))

from ml.clustering import KMeans


# shared fixture: simple dataset used across multiple tests
@pytest.fixture
def fitted_model():
    """Return a KMeans model fitted on reproducible random data."""
    np.random.seed(0)
    X = np.random.rand(200, 50)
    model = KMeans(k=8, max_iters=300, tol=1e-4)
    model.fit(X)
    return model, X


def test_fit_produces_correct_number_of_centroids(fitted_model):
    """After fitting, centroids should have shape (k, n_features)."""
    model, X = fitted_model
    assert model.centroids.shape == (8, 50)


def test_labels_shape_matches_input(fitted_model):
    """After fitting, labels should have shape (n_samples,)."""
    model, X = fitted_model
    assert model.labels.shape == (X.shape[0],)


def test_labels_are_valid_cluster_indices(fitted_model):
    """All labels should be integers in range [0, k)."""
    model, X = fitted_model
    assert model.labels.min() >= 0
    assert model.labels.max() < model.k


def test_all_clusters_represented(fitted_model):
    """All k clusters should have at least one sample assigned."""
    model, X = fitted_model
    assert len(set(model.labels.tolist())) == model.k


def test_predict_assigns_labels(fitted_model):
    """predict() should return labels of shape (n_samples,) for new data."""
    model, X = fitted_model
    new_X = np.random.rand(20, 50)
    preds = model.predict(new_X)
    assert preds.shape == (20,)
    assert preds.min() >= 0
    assert preds.max() < model.k


def test_predict_raises_before_fit():
    """predict() should raise ValueError if called before fit()."""
    model = KMeans(k=4)
    X = np.random.rand(10, 5)
    with pytest.raises(ValueError):
        model.predict(X)


def test_centroids_converge_on_well_separated_data():
    """Fitting on well-separated clusters should assign all points correctly."""
    # create 3 tight clusters far apart
    cluster_a = np.random.rand(50, 10) + np.array([0] * 10)
    cluster_b = np.random.rand(50, 10) + np.array([100] * 10)
    cluster_c = np.random.rand(50, 10) + np.array([200] * 10)
    X = np.vstack([cluster_a, cluster_b, cluster_c])

    model = KMeans(k=3, max_iters=300)
    model.fit(X)

    # each true cluster should map to one predicted cluster
    labels_a = set(model.labels[:50].tolist())
    labels_b = set(model.labels[50:100].tolist())
    labels_c = set(model.labels[100:].tolist())

    assert len(labels_a) == 1
    assert len(labels_b) == 1
    assert len(labels_c) == 1
    assert labels_a != labels_b
    assert labels_b != labels_c


def test_inertia_positive(fitted_model):
    """Inertia should always be a positive number."""
    model, X = fitted_model
    assert model.inertia(X) > 0


def test_inertia_decreases_with_more_clusters():
    """Inertia should decrease as k increases (elbow method assumption)."""
    np.random.seed(0)
    X = np.random.rand(200, 20)

    inertias = []
    for k in [2, 4, 8]:
        model = KMeans(k=k, max_iters=100)
        model.fit(X)
        inertias.append(model.inertia(X))

    assert inertias[0] > inertias[1] > inertias[2]


def test_save_and_load(fitted_model, tmp_path):
    """Saved and loaded model should produce identical predictions."""
    model, X = fitted_model
    save_path = tmp_path / "kmeans.pkl"
    model.save(save_path)

    loaded = KMeans.load(save_path)
    original_preds = model.predict(X)
    loaded_preds = loaded.predict(X)

    assert np.array_equal(original_preds, loaded_preds)
