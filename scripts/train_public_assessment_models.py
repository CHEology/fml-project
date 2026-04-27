"""
Train public-data assessment models.

This script uses three public resume datasets for three separate supervised
tasks instead of pretending they all label the same concept:

1. Kaggle Resume corpus mirror (`Resume.csv`) -> resume domain classifier.
2. 0xnbk ATS score dataset (`train.csv`, `validation.csv`) -> resume/job fit scorer.
3. DataTurks resume NER (`traindata.json`, `testdata.json`) -> entity-type classifier.

It also trains an auxiliary resume line/section classifier from the seven-class
corpus (`resume.txt`) because section recognition is directly useful for the
app's assessment errors.

The models are intentionally small raw-PyTorch MLPs over deterministic hashed
text features. They are fast to train locally, do not require downloading a
large transformer, and produce real supervised baselines that can later be
replaced by embedding-based models.

Usage:
    uv run python scripts/train_public_assessment_models.py

Outputs:
    models/public_domain_model.pt
    models/public_ats_fit_model.pt
    models/public_section_model.pt
    models/public_assessment_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


RAW_PUBLIC = PROJECT_ROOT / "data" / "raw" / "public_hf"
DEFAULT_OUT = PROJECT_ROOT / "models"
TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.\-]{1,}")
SEED = 42


@dataclass(frozen=True)
class TrainConfig:
    hash_dim: int = 2048
    hidden_dim: int = 128
    batch_size: int = 128
    epochs: int = 6
    lr: float = 1e-3
    max_domain_rows: int = 6000
    max_section_rows: int = 50000


class MLPClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPRegressor(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _stable_hash(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % dim


def hashed_features(texts: list[str], dim: int) -> np.ndarray:
    X = np.zeros((len(texts), dim), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = TOKEN_RE.findall(str(text).lower())
        if not tokens:
            continue
        for token in tokens[:1600]:
            X[row, _stable_hash(token, dim)] += 1.0
        norm = np.linalg.norm(X[row])
        if norm > 1e-6:
            X[row] /= norm
    return X


def _ats_split(text: str) -> tuple[str, str]:
    parts = re.split(r"\s+SEP\s+", str(text), maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    parts = re.split(r"\s+\[SEP\]\s+", str(text), maxsplit=1)
    if len(parts) == 2:
        return parts[0], parts[1]
    midpoint = max(1, len(str(text)) // 2)
    return str(text)[:midpoint], str(text)[midpoint:]


def ats_pair_features(texts: list[str], dim: int) -> np.ndarray:
    X = np.zeros((len(texts), dim + 8), dtype=np.float32)
    for row, raw in enumerate(texts):
        resume, job = _ats_split(raw)
        resume_tokens = TOKEN_RE.findall(resume.lower())[:1400]
        job_tokens = TOKEN_RE.findall(job.lower())[:1400]
        resume_set = set(resume_tokens)
        job_set = set(job_tokens)
        overlap = resume_set & job_set

        for token in resume_tokens:
            X[row, _stable_hash("r:" + token, dim)] += 1.0
        for token in job_tokens:
            X[row, _stable_hash("j:" + token, dim)] += 1.0
        for token in overlap:
            X[row, _stable_hash("x:" + token, dim)] += 1.5

        base_norm = np.linalg.norm(X[row, :dim])
        if base_norm > 1e-6:
            X[row, :dim] /= base_norm

        resume_len = max(1, len(resume_tokens))
        job_len = max(1, len(job_tokens))
        union_len = max(1, len(resume_set | job_set))
        X[row, dim:] = np.array(
            [
                len(overlap) / max(1, len(job_set)),
                len(overlap) / max(1, len(resume_set)),
                len(overlap) / union_len,
                np.log1p(resume_len) / 10.0,
                np.log1p(job_len) / 10.0,
                min(resume_len / job_len, 5.0) / 5.0,
                _count_year_tokens(resume_tokens) / 10.0,
                _count_year_tokens(job_tokens) / 10.0,
            ],
            dtype=np.float32,
        )
    return X


def _count_year_tokens(tokens: list[str]) -> float:
    count = 0
    for idx, token in enumerate(tokens):
        if (
            token.isdigit()
            and 0 <= idx + 1 < len(tokens)
            and tokens[idx + 1] in {"year", "years", "yr", "yrs"}
        ):
            count += float(token)
    return min(count, 30.0)


def train_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    labels: list[str],
    cfg: TrainConfig,
    out_path: Path,
) -> dict[str, float | int]:
    model = MLPClassifier(X_train.shape[1], cfg.hidden_dim, len(labels))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.long)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    best_acc = -1.0
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        acc = classifier_accuracy(model, X_val, y_val)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), out_path)

    return {"val_accuracy": round(best_acc, 4), "n_labels": len(labels)}


def classifier_accuracy(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(X))
        pred = torch.argmax(logits, dim=1).numpy()
    return float(np.mean(pred == y)) if len(y) else 0.0


def train_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: TrainConfig,
    out_path: Path,
) -> dict[str, float]:
    model = MLPRegressor(X_train.shape[1], cfg.hidden_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    criterion = nn.SmoothL1Loss()
    loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train, dtype=torch.float32)),
        batch_size=cfg.batch_size,
        shuffle=True,
    )

    best_mae = float("inf")
    for _ in range(cfg.epochs):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        mae = regressor_mae(model, X_val, y_val)
        if mae < best_mae:
            best_mae = mae
            torch.save(model.state_dict(), out_path)
    return {"val_mae": round(best_mae * 100.0, 2)}


def regressor_mae(model: nn.Module, X: np.ndarray, y: np.ndarray) -> float:
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(X)).numpy()
    return float(np.mean(np.abs(pred - y))) if len(y) else 0.0


def load_domain_data(path: Path, max_rows: int, seed: int) -> tuple[list[str], list[str]]:
    df = pd.read_csv(path, usecols=["Resume_str", "Category"])
    df = df.dropna(subset=["Resume_str", "Category"])
    df = df[df["Resume_str"].astype(str).str.len() > 200]
    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed)
    return df["Resume_str"].astype(str).tolist(), df["Category"].astype(str).tolist()


def load_ats_data(train_path: Path, val_path: Path) -> tuple[list[str], np.ndarray, list[str], np.ndarray]:
    train = pd.read_csv(train_path, usecols=["text", "ats_score"])
    val = pd.read_csv(val_path, usecols=["text", "ats_score"])
    train = train.dropna(subset=["text", "ats_score"])
    val = val.dropna(subset=["text", "ats_score"])
    return (
        train["text"].astype(str).tolist(),
        (train["ats_score"].astype(float).to_numpy(dtype=np.float32) / 100.0),
        val["text"].astype(str).tolist(),
        (val["ats_score"].astype(float).to_numpy(dtype=np.float32) / 100.0),
    )


def load_section_data(path: Path, max_rows: int, seed: int) -> tuple[list[str], list[str]]:
    rows: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f, delimiter="\t")
        for row in reader:
            if len(row) < 2:
                continue
            label, text = row[0].strip(), row[1].strip()
            if label and text and len(text) >= 3:
                rows.append((text, label))
    rng = np.random.default_rng(seed)
    if len(rows) > max_rows:
        idx = rng.choice(len(rows), size=max_rows, replace=False)
        rows = [rows[int(i)] for i in idx]
    texts, labels = zip(*rows, strict=True)
    return list(texts), list(labels)


def load_dataturks_entities(path: Path) -> tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open(encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            content = str(row.get("content", ""))
            for ann in row.get("annotation", []) or []:
                raw_labels = ann.get("label") or []
                if not raw_labels:
                    continue
                label = str(raw_labels[0])
                for point in ann.get("points", []) or []:
                    start = int(point.get("start", 0))
                    end = int(point.get("end", start))
                    span = str(point.get("text") or content[start : end + 1])
                    left = content[max(0, start - 80) : start]
                    right = content[end + 1 : min(len(content), end + 81)]
                    sample = f"{span} context {left} {right}"
                    if span.strip():
                        texts.append(sample)
                        labels.append(label)
    return texts, labels


def split_labels(
    texts: list[str],
    labels: list[str],
    seed: int,
    val_frac: float = 0.2,
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, list[str]]:
    label_names = sorted(set(labels))
    lookup = {label: i for i, label in enumerate(label_names)}
    y = np.array([lookup[label] for label in labels], dtype=np.int64)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(texts))
    n_val = max(1, int(len(texts) * val_frac))
    val_idx = order[:n_val]
    train_idx = order[n_val:]
    return (
        [texts[i] for i in train_idx],
        y[train_idx],
        [texts[i] for i in val_idx],
        y[val_idx],
        label_names,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=RAW_PUBLIC)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    parser.add_argument("--hash-dim", type=int, default=TrainConfig.hash_dim)
    parser.add_argument("--max-domain-rows", type=int, default=TrainConfig.max_domain_rows)
    parser.add_argument("--max-section-rows", type=int, default=TrainConfig.max_section_rows)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg = TrainConfig(
        epochs=args.epochs,
        hash_dim=args.hash_dim,
        max_domain_rows=args.max_domain_rows,
        max_section_rows=args.max_section_rows,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, object] = {
        "seed": args.seed,
        "hash_dim": cfg.hash_dim,
        "datasets": {},
    }

    domain_texts, domain_labels = load_domain_data(
        args.raw_dir / "Resume.csv", cfg.max_domain_rows, args.seed
    )
    d_train_texts, d_y_train, d_val_texts, d_y_val, d_labels = split_labels(
        domain_texts, domain_labels, args.seed
    )
    d_X_train = hashed_features(d_train_texts, cfg.hash_dim)
    d_X_val = hashed_features(d_val_texts, cfg.hash_dim)
    domain_metrics = train_classifier(
        d_X_train,
        d_y_train,
        d_X_val,
        d_y_val,
        labels=d_labels,
        cfg=cfg,
        out_path=args.out_dir / "public_domain_model.pt",
    )
    metrics["datasets"]["domain"] = {
        "source": "Divyaamith/Kaggle-Resume Resume.csv",
        "train_rows": len(d_train_texts),
        "val_rows": len(d_val_texts),
        "labels": d_labels,
        **domain_metrics,
    }

    ats_train_texts, ats_y_train, ats_val_texts, ats_y_val = load_ats_data(
        args.raw_dir / "train.csv", args.raw_dir / "validation.csv"
    )
    a_X_train = ats_pair_features(ats_train_texts, cfg.hash_dim)
    a_X_val = ats_pair_features(ats_val_texts, cfg.hash_dim)
    ats_metrics = train_regressor(
        a_X_train,
        ats_y_train,
        a_X_val,
        ats_y_val,
        cfg=cfg,
        out_path=args.out_dir / "public_ats_fit_model.pt",
    )
    metrics["datasets"]["ats_fit"] = {
        "source": "0xnbk/resume-ats-score-v1-en",
        "train_rows": len(ats_train_texts),
        "val_rows": len(ats_val_texts),
        **ats_metrics,
    }

    entity_train_texts, entity_train_labels = load_dataturks_entities(
        args.raw_dir.parent / "public_dataturks" / "traindata.json"
    )
    entity_val_texts, entity_val_labels = load_dataturks_entities(
        args.raw_dir.parent / "public_dataturks" / "testdata.json"
    )
    entity_labels = sorted(set(entity_train_labels) | set(entity_val_labels))
    entity_lookup = {label: i for i, label in enumerate(entity_labels)}
    e_y_train = np.array([entity_lookup[label] for label in entity_train_labels])
    e_y_val = np.array([entity_lookup[label] for label in entity_val_labels])
    e_X_train = hashed_features(entity_train_texts, cfg.hash_dim)
    e_X_val = hashed_features(entity_val_texts, cfg.hash_dim)
    entity_metrics = train_classifier(
        e_X_train,
        e_y_train,
        e_X_val,
        e_y_val,
        labels=entity_labels,
        cfg=cfg,
        out_path=args.out_dir / "public_entity_model.pt",
    )
    metrics["datasets"]["entity"] = {
        "source": "DataTurks-Engg/Entity-Recognition-In-Resumes-SpaCy",
        "train_rows": len(entity_train_texts),
        "val_rows": len(entity_val_texts),
        "labels": entity_labels,
        **entity_metrics,
    }

    section_texts, section_labels = load_section_data(
        args.raw_dir / "resume.txt", cfg.max_section_rows, args.seed
    )
    s_train_texts, s_y_train, s_val_texts, s_y_val, s_labels = split_labels(
        section_texts, section_labels, args.seed
    )
    s_X_train = hashed_features(s_train_texts, cfg.hash_dim)
    s_X_val = hashed_features(s_val_texts, cfg.hash_dim)
    section_metrics = train_classifier(
        s_X_train,
        s_y_train,
        s_X_val,
        s_y_val,
        labels=s_labels,
        cfg=cfg,
        out_path=args.out_dir / "public_section_model.pt",
    )
    metrics["datasets"]["section"] = {
        "source": "ganchengguang/resume_seven_class resume.txt",
        "train_rows": len(s_train_texts),
        "val_rows": len(s_val_texts),
        "labels": s_labels,
        **section_metrics,
    }

    with (args.out_dir / "public_assessment_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
