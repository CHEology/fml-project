"""
Build KMeans clustering artifacts from saved job embeddings.

Usage:
    uv run python scripts/build_clusters.py \
        --embeddings models/job_embeddings.npy \
        --jobs       data/processed/jobs.parquet \
        --k          8

The KMeans implementation comes from ml/clustering.py and is intentionally
NumPy-based for the course requirement. TF-IDF is used only to label fitted
clusters for the Streamlit app.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.clustering import KMeans  # noqa: E402

DEFAULTS = {
    "embeddings": PROJECT_ROOT / "models" / "job_embeddings.npy",
    "jobs": PROJECT_ROOT / "data" / "processed" / "jobs.parquet",
    "k": 8,
    "max_iters": 300,
    "tol": 1e-4,
    "model_out": PROJECT_ROOT / "models" / "kmeans_k8.pkl",
    "assignments_out": PROJECT_ROOT / "models" / "cluster_assignments.npy",
    "centroids_out": PROJECT_ROOT / "models" / "cluster_centroids.npy",
    "labels_out": PROJECT_ROOT / "models" / "cluster_labels.json",
}


def label_clusters(jobs: pd.DataFrame, labels: np.ndarray, k: int) -> dict[str, dict]:
    from sklearn.feature_extraction.text import TfidfVectorizer

    texts = jobs.get("text", pd.Series("", index=jobs.index)).fillna("").astype(str)
    titles = jobs.get("title", pd.Series("", index=jobs.index)).fillna("").astype(str)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        min_df=5,
        max_features=6000,
        ngram_range=(1, 2),
    )
    tfidf = vectorizer.fit_transform(texts)
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    output: dict[str, dict] = {}
    for cluster_id in range(k):
        mask = labels == cluster_id
        size = int(mask.sum())
        if size == 0:
            output[str(cluster_id)] = {
                "label": f"Cluster {cluster_id}",
                "size": 0,
                "top_terms": [],
                "common_titles": [],
            }
            continue

        mean_scores = np.asarray(tfidf[mask].mean(axis=0)).ravel()
        top_idx = np.argsort(-mean_scores)[:10]
        top_terms = [
            str(term) for term in feature_names[top_idx] if mean_scores[top_idx[0]] > 0
        ]
        common_titles = titles[mask].value_counts().head(5).index.astype(str).tolist()
        output[str(cluster_id)] = {
            "label": _cluster_label(top_terms, common_titles, cluster_id),
            "size": size,
            "top_terms": top_terms[:10],
            "common_titles": common_titles,
        }
    return output


def _cluster_label(
    top_terms: list[str],
    common_titles: list[str],
    cluster_id: int,
) -> str:
    text = " ".join(top_terms + common_titles).lower()
    patterns = [
        (
            "Healthcare / Clinical",
            ("nurse", "medical", "clinical", "health", "patient"),
        ),
        (
            "Finance / Accounting",
            ("finance", "accounting", "tax", "financial", "audit"),
        ),
        ("Business / Data Analysis", ("business", "data", "analyst", "management")),
        (
            "Sales / Customer Growth",
            (
                "sales",
                "account",
                "customer",
                "marketing",
                "store",
                "business development",
            ),
        ),
        (
            "Administrative / HR",
            (
                "administrative",
                "assistant",
                "office",
                "human resources",
                "receptionist",
            ),
        ),
        (
            "Operations / Logistics",
            (
                "warehouse",
                "maintenance",
                "equipment",
                "safety",
                "material",
                "technician",
            ),
        ),
        ("Legal / Compliance", ("attorney", "legal", "compliance", "law", "counsel")),
        (
            "Software / Engineering",
            ("software", "engineer", "developer", "backend", "frontend"),
        ),
        (
            "Machine Learning / AI",
            ("machine learning", "artificial intelligence", "ml", "model", "nlp"),
        ),
        ("Data / Analytics", ("data", "analytics", "sql", "analyst", "scientist")),
        ("Marketing / Content", ("marketing", "content", "brand", "social media")),
        (
            "Product / Operations",
            ("product", "operations", "program", "project", "manager"),
        ),
    ]
    best_label = None
    best_score = 0
    for label, keywords in patterns:
        score = sum(_contains_keyword(text, keyword) for keyword in keywords)
        if score > best_score:
            best_label = label
            best_score = score
    if best_label is not None:
        return best_label
    if common_titles:
        words = common_titles[0].split()
        return " ".join(words[:4])
    return f"Cluster {cluster_id}"


def _contains_keyword(text: str, keyword: str) -> bool:
    if " " in keyword:
        return keyword in text
    return re.search(rf"\b{re.escape(keyword)}\b", text) is not None


def main() -> None:
    parser = argparse.ArgumentParser(description="Build KMeans cluster artifacts")
    parser.add_argument("--embeddings", type=Path, default=DEFAULTS["embeddings"])
    parser.add_argument("--jobs", type=Path, default=DEFAULTS["jobs"])
    parser.add_argument("--k", type=int, default=DEFAULTS["k"])
    parser.add_argument("--max-iters", type=int, default=DEFAULTS["max_iters"])
    parser.add_argument("--tol", type=float, default=DEFAULTS["tol"])
    parser.add_argument("--model-out", type=Path, default=DEFAULTS["model_out"])
    parser.add_argument(
        "--assignments-out", type=Path, default=DEFAULTS["assignments_out"]
    )
    parser.add_argument("--centroids-out", type=Path, default=DEFAULTS["centroids_out"])
    parser.add_argument("--labels-out", type=Path, default=DEFAULTS["labels_out"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    np.random.seed(args.seed)

    print(f"Loading embeddings from {args.embeddings}")
    embeddings = np.load(args.embeddings).astype(np.float32, copy=False)
    print(f"Embeddings shape: {embeddings.shape}")

    print(f"Fitting KMeans(k={args.k}, max_iters={args.max_iters})")
    model = KMeans(k=args.k, max_iters=args.max_iters, tol=args.tol).fit(embeddings)
    if model.labels is None or model.centroids is None:
        raise RuntimeError("KMeans fit did not produce labels and centroids")

    for path in (
        args.model_out,
        args.assignments_out,
        args.centroids_out,
        args.labels_out,
    ):
        path.parent.mkdir(parents=True, exist_ok=True)

    model.save(args.model_out)
    np.save(args.assignments_out, model.labels.astype(np.int64))
    np.save(args.centroids_out, model.centroids.astype(np.float32))
    print(f"Saved KMeans model: {args.model_out}")
    print(f"Saved assignments:  {args.assignments_out}")
    print(f"Saved centroids:    {args.centroids_out}")

    print(f"Loading jobs from {args.jobs}")
    jobs = pd.read_parquet(args.jobs, columns=["title", "text"])
    cluster_labels = label_clusters(jobs, model.labels, args.k)
    with args.labels_out.open("w") as f:
        json.dump(cluster_labels, f, indent=2)
    print(f"Saved labels:       {args.labels_out}")

    for cluster_id, info in cluster_labels.items():
        print(
            f"  {cluster_id}: {info['label']} "
            f"({info['size']} jobs) terms={', '.join(info['top_terms'][:5])}"
        )


if __name__ == "__main__":
    main()
