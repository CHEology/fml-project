"""
Training script for the SalaryQuantileNet.

Usage:
    python scripts/train_salary_model.py \
        --embeddings models/job_embeddings.npy \
        --salaries   data/processed/salaries.npy \
        --output     models/salary_model.pt

Reads pre-computed embeddings + salary targets, trains with pinball loss,
applies early stopping on validation loss, and saves the best checkpoint.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure project root is on sys.path so `ml.*` is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ml.salary_model import (
    SalaryQuantileNet,
    PinballLoss,
    split_data,
    SEED,
)


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

DEFAULTS = {
    "embedding_dim": 384,
    "n_extra_features": 0,
    "batch_size": 256,
    "lr": 1e-3,
    "epochs": 200,
    "patience": 15,          # early-stopping patience
    "dropout": 0.2,
    "weight_decay": 1e-5,
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

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
    output_path: str,
) -> dict:
    """Full training loop with early stopping + ReduceLROnPlateau."""

    criterion = PinballLoss().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, verbose=True
    )

    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        train_losses = []
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train = np.mean(train_losses)

        # ---- validate ----
        model.eval()
        val_losses = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                val_losses.append(loss.item())

        avg_val = np.mean(val_losses)
        scheduler.step(avg_val)

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        # ---- logging ----
        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:>3d}/{epochs}  "
            f"train_loss={avg_train:.4f}  val_loss={avg_val:.4f}  "
            f"lr={current_lr:.2e}"
        )

        # ---- early stopping ----
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            epochs_no_improve = 0
            torch.save(model.state_dict(), output_path)
            print(f"  ↳ saved best model (val_loss={best_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(no improvement for {patience} epochs)")
                break

    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")
    print(f"Checkpoint saved to: {output_path}")
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train SalaryQuantileNet")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path to job embeddings .npy file")
    parser.add_argument("--salaries", type=str, required=True,
                        help="Path to salary targets .npy file")
    parser.add_argument("--extra-features", type=str, default=None,
                        help="Optional path to extra features .npy file")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "models" / "salary_model.pt"),
                        help="Where to save the best checkpoint")
    parser.add_argument("--embedding-dim", type=int,
                        default=DEFAULTS["embedding_dim"])
    parser.add_argument("--batch-size", type=int,
                        default=DEFAULTS["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULTS["lr"])
    parser.add_argument("--epochs", type=int, default=DEFAULTS["epochs"])
    parser.add_argument("--patience", type=int, default=DEFAULTS["patience"])
    parser.add_argument("--dropout", type=float, default=DEFAULTS["dropout"])
    parser.add_argument("--weight-decay", type=float,
                        default=DEFAULTS["weight_decay"])
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    # ---- reproducibility ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ---- load data ----
    print(f"Loading embeddings from {args.embeddings}")
    embeddings = np.load(args.embeddings)
    print(f"Loading salaries from {args.salaries}")
    salaries = np.load(args.salaries)

    extra = None
    n_extra = 0
    if args.extra_features:
        print(f"Loading extra features from {args.extra_features}")
        extra = np.load(args.extra_features)
        n_extra = extra.shape[1]

    print(f"Dataset: {len(salaries)} samples, "
          f"embedding_dim={embeddings.shape[1]}, n_extra={n_extra}")

    # ---- split ----
    train_ds, val_ds, test_ds = split_data(
        embeddings, salaries, extra, seed=args.seed
    )
    print(f"Split: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # ---- model ----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = SalaryQuantileNet(
        embedding_dim=args.embedding_dim,
        n_extra_features=n_extra,
        dropout=args.dropout,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # ---- ensure output dir exists ----
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    # ---- train ----
    history = train(
        train_loader,
        val_loader,
        model,
        device,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        weight_decay=args.weight_decay,
        output_path=args.output,
    )

    # ---- quick test-set evaluation ----
    print("\n--- Test set evaluation ---")
    model.load_state_dict(
        torch.load(args.output, map_location=device, weights_only=True)
    )
    model.eval()
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)
    criterion = PinballLoss().to(device)

    test_losses = []
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            test_losses.append(criterion(preds, y_batch).item())
            all_preds.append(preds.cpu().numpy())
            all_targets.append(y_batch.cpu().numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print(f"Test pinball loss: {np.mean(test_losses):.4f}")

    # Calibration: fraction of actuals below each predicted quantile
    from ml.salary_model import QUANTILES
    for i, q in enumerate(QUANTILES):
        frac_below = (all_targets < all_preds[:, i]).mean()
        print(f"  q{int(q*100):>2d}: nominal={q:.2f}, actual={frac_below:.3f}")

    # Median absolute error for q50
    median_idx = list(QUANTILES).index(0.50)
    mae = np.abs(all_targets - all_preds[:, median_idx]).mean()
    print(f"  Median prediction MAE: ${mae:,.0f}")


if __name__ == "__main__":
    main()
