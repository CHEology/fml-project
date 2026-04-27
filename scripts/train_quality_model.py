"""
Train the resume-quality regressor.

Reads `data/eval/synthetic_resumes.parquet`, embeds each `resume_text`
with the project encoder (lazy-imported from `ml.embeddings` — same
deferral pattern as `ml/retrieval.py`), and trains a single-head
regression MLP on the synthetic `quality_score` (0–100).

Honest-eval caveat: targets come from the generator's own scoring
rule. The model learns to mimic that rule on text it has not seen,
which is the right calibration object for the demo but a *proxy*
until real labels exist. The script is documented so this caveat is
hard to miss.

Usage:
    python scripts/train_quality_model.py \
        --resumes data/eval/synthetic_resumes.parquet \
        --epochs 30 \
        --out models/quality_model.pt

    # CI / no-encoder smoke run (uses random unit vectors as embeddings):
    python scripts/train_quality_model.py --smoke --epochs 2 \
        --resumes data/eval/synthetic_resumes.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.quality import (  # noqa: E402
    DEFAULT_EMBEDDING_DIM,
    SEED,
    QualityScaler,
    ResumeQualityModel,
    split_data,
)

DEFAULTS = {
    "embedding_dim": DEFAULT_EMBEDDING_DIM,
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 30,
    "patience": 5,
    "dropout": 0.2,
    "weight_decay": 1e-5,
}


def _smoke_embeddings(n: int, dim: int, seed: int) -> np.ndarray:
    """Deterministic L2-normalized random vectors for smoke runs."""
    rng = np.random.default_rng(seed)
    vecs = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return vecs / norms


def _real_embeddings(texts: list[str], model_name: str, batch_size: int) -> np.ndarray:
    try:
        from ml.embeddings import Encoder
    except ImportError as exc:
        raise RuntimeError(
            "ml.embeddings.Encoder is required for non-smoke training. "
            "Build the embedding module first or pass --smoke."
        ) from exc
    encoder = Encoder(model_name=model_name)
    return np.asarray(encoder.encode(texts, batch_size=batch_size), dtype=np.float32)


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: ResumeQualityModel,
    device: torch.device,
    *,
    lr: float,
    epochs: int,
    patience: int,
    weight_decay: float,
    output_path: Path,
) -> dict[str, list[float]]:
    """Standard MSE training loop with early stopping."""
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2
    )

    best_val = float("inf")
    bad_epochs = 0
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            preds = model(X)
            loss = criterion(preds, y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                val_losses.append(criterion(model(X), y).item())

        avg_train = float(np.mean(train_losses)) if train_losses else 0.0
        avg_val = float(np.mean(val_losses)) if val_losses else 0.0
        scheduler.step(avg_val)
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        print(
            f"Epoch {epoch:>3d}/{epochs}  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
            f"lr={optimizer.param_groups[0]['lr']:.2e}"
        )

        if avg_val < best_val:
            best_val = avg_val
            bad_epochs = 0
            torch.save(model.state_dict(), output_path)
            print(f"  ↳ saved best model (val_loss={best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResumeQualityModel")
    parser.add_argument(
        "--resumes",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "synthetic_resumes.parquet",
        help="Parquet with `resume_text`, `quality_score`, optional `quality_label`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "models" / "quality_model.pt",
        help="Where to save the best checkpoint.",
    )
    parser.add_argument("--model-name", type=str, default="all-MiniLM-L6-v2")
    parser.add_argument("--embedding-dim", type=int, default=DEFAULTS["embedding_dim"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--weight-decay", type=float, default=DEFAULTS["weight_decay"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use deterministic random embeddings instead of ml.embeddings.Encoder.",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print(f"Loading resumes from {args.resumes}")
    df = pd.read_parquet(args.resumes)
    if "resume_text" not in df.columns or "quality_score" not in df.columns:
        raise ValueError(
            "resume parquet must contain 'resume_text' and 'quality_score' columns"
        )

    texts = df["resume_text"].astype(str).tolist()
    scores = df["quality_score"].to_numpy(dtype=np.float32)
    labels = df["quality_label"].to_numpy() if "quality_label" in df.columns else None

    if args.smoke:
        print(f"Smoke mode: generating {len(texts)} random embeddings.")
        embeddings = _smoke_embeddings(len(texts), args.embedding_dim, args.seed)
    else:
        print(f"Encoding {len(texts)} resumes with {args.model_name}.")
        embeddings = _real_embeddings(texts, args.model_name, args.batch_size)
        if embeddings.shape[1] != args.embedding_dim:
            print(
                f"  ! encoder produced dim={embeddings.shape[1]}, "
                f"overriding --embedding-dim ({args.embedding_dim})."
            )
            args.embedding_dim = int(embeddings.shape[1])

    train_ds, val_ds, test_ds, scaler = split_data(
        embeddings, scores, labels=labels, seed=args.seed
    )
    print(
        f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"scaler(mean={scaler.mean:.2f}, std={scaler.std:.2f})"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResumeQualityModel(
        embedding_dim=args.embedding_dim, dropout=args.dropout
    ).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    train(
        train_loader,
        val_loader,
        model,
        device,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        output_path=args.out,
    )

    # ---- evaluate on test split in original units ----
    if len(test_ds) > 0:
        model.load_state_dict(
            torch.load(args.out, map_location=device, weights_only=True)
        )
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        preds_all = []
        targets_all = []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                preds_all.append(model(X).cpu().numpy())
                targets_all.append(y.numpy())
        preds = scaler.inverse_transform(np.concatenate(preds_all))
        targets = scaler.inverse_transform(np.concatenate(targets_all))
        mae = float(np.mean(np.abs(preds - targets)))
        print(f"Test MAE (0–100 scale): {mae:.2f}")

    scaler_path = args.out.with_suffix(".scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(_scaler_meta(scaler, args.embedding_dim), f)
    print(f"Scaler saved to {scaler_path}")


def _scaler_meta(scaler: QualityScaler, embedding_dim: int) -> dict:
    payload = scaler.state_dict()
    payload["embedding_dim"] = int(embedding_dim)
    return payload


if __name__ == "__main__":
    main()
