"""
ml/feedback.py

Implements resume feedback and gap analysis logic.
Identifies the "direction" a user needs to move their resume to match a target job or cluster.
"""

from typing import Any

import numpy as np


def compute_gap_analysis(
    resume_embedding: np.ndarray, target_embedding: np.ndarray
) -> dict[str, Any]:
    """
    Identifies the direction vector between a resume and a target.

    Args:
        resume_embedding: Embedding of the user's resume (1, D)
        target_embedding: Embedding of the target job or cluster centroid (1, D)

    Returns:
        Dictionary containing:
            - direction_vector: The raw difference (target - resume)
            - gap_magnitude: The Euclidean distance of the gap
            - similarity_improvement: Potential cosine similarity gain if the gap were closed
    """
    # Ensure inputs are 1D for vector ops or properly squeezed
    r = np.asarray(resume_embedding).flatten()
    t = np.asarray(target_embedding).flatten()

    direction = t - r
    magnitude = np.linalg.norm(direction)

    # Cosine similarity improvement logic:
    # Current similarity: dot(r, t) / (norm(r)*norm(t))
    # If we add 'direction' to r, new_r = r + (t-r) = t.
    # New similarity would be 1.0 (if r and t are normalized or same)

    return {
        "direction_vector": direction,
        "gap_magnitude": float(magnitude),
    }


def project_vector_to_terms(
    direction_vector: np.ndarray,
    term_embeddings: np.ndarray,
    terms: list[str],
    top_n: int = 5,
) -> list[tuple[str, float]]:
    """
    Identifies which terms (skills/keywords) are most aligned with the direction vector.
    This helps translate the abstract embedding gap into human-readable suggestions.

    Args:
        direction_vector: The (target - resume) vector
        term_embeddings: Matrix of embeddings for a vocabulary of terms (N, D)
        terms: List of N term strings
        top_n: Number of suggestions to return

    Returns:
        List of (term, alignment_score) tuples
    """
    v = direction_vector.flatten()
    # Normalize direction vector for cosine similarity
    v_norm = np.linalg.norm(v)
    if v_norm < 1e-9:
        return []
    v_unit = v / v_norm

    # term_embeddings should ideally be (N, D)
    # Compute dot products (alignment)
    # If term_embeddings are already unit vectors, this is cosine similarity
    scores = np.dot(term_embeddings, v_unit)

    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(terms[i], float(scores[i])) for i in top_indices]
