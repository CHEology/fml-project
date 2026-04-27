"""
Train a resume-domain salary quantile model.

Background: Alan's `SalaryQuantileNet` was trained on
(JD embedding -> JD salary) pairs. Inferring on resume embeddings
relies on the assumption that resume vectors land in the same region
of MiniLM space as the JDs that hired them — which is *not* validated.
Resume language ("Built X for Y") differs systematically from JD
language ("Looking for someone to do X").

This script removes the domain shift by retraining the *same*
architecture (no model code changes — same `SalaryQuantileNet`,
same `PinballLoss`, same `split_data`) on
(resume embedding -> source job salary) pairs that
`scripts/generate_synthetic_resumes.py` now produces (`source_salary_annual`).

The output checkpoint goes to `models/resume_salary_model.pt` so it
sits alongside the JD-side model and the resume-inference pipeline
loads it explicitly.

Usage:
    python scripts/train_resume_salary_model.py \\
        --resumes data/eval/synthetic_resumes.parquet \\
        --epochs 30 \\
        --out models/resume_salary_model.pt

    # CI / no-encoder smoke run:
    python scripts/train_resume_salary_model.py --smoke --epochs 3 \\
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

from ml.salary_model import (  # noqa: E402
    QUANTILES,
    SEED,
    PinballLoss,
    SalaryQuantileNet,
    split_data,
)

DEFAULTS = {
    "embedding_dim": 384,
    "batch_size": 64,
    "lr": 1e-3,
    "epochs": 30,
    "patience": 5,
    "dropout": 0.2,
    "weight_decay": 1e-5,
}


def _smoke_embeddings(n: int, dim: int, seed: int) -> np.ndarray:
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
            "Land Task 2.1 first or pass --smoke."
        ) from exc
    encoder = Encoder(model_name=model_name)
    return np.asarray(encoder.encode(texts, batch_size=batch_size), dtype=np.float32)


def train(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model: SalaryQuantileNet,
    device: torch.device,
    *,
    lr: float,
    epochs: int,
    patience: int,
    weight_decay: float,
    output_path: Path,
) -> dict[str, list[float]]:
    """Standard pinball-loss training loop with early stopping."""
    criterion = PinballLoss().to(device)
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
            loss = criterion(model(X), y)
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
            print(f"  > saved best model (val_loss={best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"\nEarly stopping at epoch {epoch}.")
                break

    return history


def main() -> None:
    parser = argparse.ArgumentParser(description="Train resume-side SalaryQuantileNet")
    parser.add_argument(
        "--resumes",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "synthetic_resumes.parquet",
        help="Parquet with `resume_text` and `source_salary_annual`.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=PROJECT_ROOT / "models" / "resume_salary_model.pt",
        help="Checkpoint path.",
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
    if "resume_text" not in df.columns or "source_salary_annual" not in df.columns:
        raise ValueError(
            "resume parquet must contain 'resume_text' and 'source_salary_annual'. "
            "Regenerate with the latest scripts/generate_synthetic_resumes.py."
        )

    df = df.dropna(subset=["source_salary_annual"]).reset_index(drop=True)
    df = df[df["source_salary_annual"] > 0].reset_index(drop=True)
    if len(df) < 20:
        raise ValueError(
            f"only {len(df)} resumes have a positive source_salary_annual; "
            "regenerate with --jobs pointing at a salary-bearing dataset."
        )

    texts = df["resume_text"].astype(str).tolist()
    salaries = df["source_salary_annual"].to_numpy(dtype=np.float32)

    if args.smoke:
        print(f"Smoke mode: generating {len(texts)} random embeddings.")
        embeddings = _smoke_embeddings(len(texts), args.embedding_dim, args.seed)
    else:
        print(f"Encoding {len(texts)} resumes with {args.model_name}.")
        embeddings = _real_embeddings(texts, args.model_name, args.batch_size)
        if embeddings.shape[1] != args.embedding_dim:
            print(
                f"  ! encoder produced dim={embeddings.shape[1]}, overriding "
                f"--embedding-dim ({args.embedding_dim})."
            )
            args.embedding_dim = int(embeddings.shape[1])

    train_ds, val_ds, test_ds, scaler = split_data(embeddings, salaries, seed=args.seed)
    print(
        f"Split: train={len(train_ds)} val={len(val_ds)} test={len(test_ds)} "
        f"scaler(mean=${scaler.mean:,.0f}, std=${scaler.std:,.0f})"
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SalaryQuantileNet(
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

    # ---- evaluate on test split in USD ----
    if len(test_ds) > 0:
        model.load_state_dict(
            torch.load(args.out, map_location=device, weights_only=True)
        )
        model.eval()
        test_loader = DataLoader(test_ds, batch_size=args.batch_size)
        preds_all, targets_all = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(device)
                preds_all.append(model(X).cpu().numpy())
                targets_all.append(y.numpy())
        preds = scaler.inverse_transform(np.concatenate(preds_all))
        targets = scaler.inverse_transform(np.concatenate(targets_all))
        preds = np.sort(preds, axis=1)
        median_idx = list(QUANTILES).index(0.50)
        mae = float(np.abs(targets - preds[:, median_idx]).mean())
        print(f"Test median MAE (USD): ${mae:,.0f}")
        for i, q in enumerate(QUANTILES):
            frac = float((targets <= preds[:, i]).mean())
            print(f"  q{int(q * 100):>2d}: nominal={q:.2f} actual={frac:.3f}")

    scaler_path = args.out.with_suffix(".scaler.json")
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump({**scaler.state_dict(), "embedding_dim": int(args.embedding_dim)}, f)
    print(f"Scaler saved to {scaler_path}")


if __name__ == "__main__":
    main()
