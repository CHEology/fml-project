"""
Quantile Regression model for salary prediction.

Predicts salary distributions (10th, 25th, 50th, 75th, 90th percentiles)
using a multi-head neural network with pinball loss.

Grading constraint: raw PyTorch only — no sklearn regressors.
"""

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
DEFAULT_EMBEDDING_DIM = 384          # all-MiniLM-L6-v2
NUM_EXTRA_FEATURES = 0               # placeholder; updated after preprocessing
SEED = 42


# ---------------------------------------------------------------------------
# Salary Scaler (z-score normalisation)
# ---------------------------------------------------------------------------

@dataclass
class SalaryScaler:
    """Simple z-score scaler fit on training salaries.

    Keeps salary targets near zero during training so the network
    can learn effectively with standard weight initialisation.
    """
    mean: float = 0.0
    std: float = 1.0

    def fit(self, salaries: np.ndarray) -> "SalaryScaler":
        self.mean = float(np.mean(salaries))
        self.std = float(np.std(salaries))
        if self.std < 1e-8:
            self.std = 1.0
        return self

    def transform(self, salaries: np.ndarray) -> np.ndarray:
        return (salaries - self.mean) / self.std

    def inverse_transform(self, scaled: np.ndarray) -> np.ndarray:
        return scaled * self.std + self.mean

    def state_dict(self) -> dict:
        return {"mean": self.mean, "std": self.std}

    @classmethod
    def from_state_dict(cls, d: dict) -> "SalaryScaler":
        return cls(mean=d["mean"], std=d["std"])


# ---------------------------------------------------------------------------
# Pinball (Quantile) Loss
# ---------------------------------------------------------------------------

class PinballLoss(nn.Module):
    """Pinball loss for quantile regression.

    For quantile τ:
        L(y, ŷ) = τ * max(y - ŷ, 0) + (1 - τ) * max(ŷ - y, 0)
    """

    def __init__(self, quantiles: tuple[float, ...] = QUANTILES):
        super().__init__()
        self.register_buffer(
            "quantiles", torch.tensor(quantiles, dtype=torch.float32)
        )

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            y_pred: (batch, num_quantiles)
            y_true: (batch,) or (batch, 1)
        Returns:
            Scalar loss averaged over batch and quantiles.
        """
        if y_true.dim() == 1:
            y_true = y_true.unsqueeze(1)                   # (B, 1)

        errors = y_true - y_pred                            # (B, Q)
        loss = torch.max(self.quantiles * errors,
                         (self.quantiles - 1) * errors)     # (B, Q)
        return loss.mean()


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class SalaryQuantileNet(nn.Module):
    """Multi-head quantile regression network.

    Architecture:
        input (embedding_dim + n_extra) → 256 → 128 → 64 → 5 quantile heads
    Uses BatchNorm + Dropout for regularisation.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        n_extra_features: int = NUM_EXTRA_FEATURES,
        quantiles: tuple[float, ...] = QUANTILES,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.quantiles = quantiles
        in_dim = embedding_dim + n_extra_features

        self.backbone = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # One output per quantile
        self.head = nn.Linear(64, len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, embedding_dim + n_extra_features)
        Returns:
            (batch, num_quantiles) predicted salary at each quantile
        """
        h = self.backbone(x)
        return self.head(h)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SalaryDataset(Dataset):
    """PyTorch dataset that pairs job embeddings (+ optional extra features)
    with annualised salary targets.

    Args:
        embeddings:  (N, embedding_dim) numpy array of dense vectors.
        salaries:    (N,) numpy array of annualised salary values.
        extra_features: Optional (N, n_extra) numpy array (experience level, etc.).
    """

    def __init__(
        self,
        embeddings: np.ndarray,
        salaries: np.ndarray,
        extra_features: np.ndarray | None = None,
    ):
        assert len(embeddings) == len(salaries), "Length mismatch"
        self.X = torch.tensor(embeddings, dtype=torch.float32)
        self.y = torch.tensor(salaries, dtype=torch.float32)

        if extra_features is not None:
            extra = torch.tensor(extra_features, dtype=torch.float32)
            self.X = torch.cat([self.X, extra], dim=1)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


# ---------------------------------------------------------------------------
# Train / Val / Test Split Utility
# ---------------------------------------------------------------------------

def split_data(
    embeddings: np.ndarray,
    salaries: np.ndarray,
    extra_features: np.ndarray | None = None,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = SEED,
    scale: bool = True,
) -> tuple[SalaryDataset, SalaryDataset, SalaryDataset, SalaryScaler]:
    """Randomly split data into train / val / test datasets.

    When *scale* is True (default), a SalaryScaler is fit on the
    training split and applied to all splits so that the network
    trains on z-scored salary targets.

    Returns:
        (train_ds, val_ds, test_ds, scaler)
    """
    rng = np.random.default_rng(seed)
    n = len(salaries)
    indices = rng.permutation(n)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    scaler = SalaryScaler()
    if scale:
        scaler.fit(salaries[train_idx])
        sal = scaler.transform(salaries)
    else:
        sal = salaries

    def _make(idx):
        ef = extra_features[idx] if extra_features is not None else None
        return SalaryDataset(embeddings[idx], sal[idx], ef)

    return _make(train_idx), _make(val_idx), _make(test_idx), scaler


# ---------------------------------------------------------------------------
# Inference API
# ---------------------------------------------------------------------------

def load_model(
    checkpoint_path: str,
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    n_extra_features: int = NUM_EXTRA_FEATURES,
    device: str = "cpu",
) -> SalaryQuantileNet:
    """Load a trained SalaryQuantileNet from a .pt checkpoint."""
    model = SalaryQuantileNet(
        embedding_dim=embedding_dim,
        n_extra_features=n_extra_features,
    )
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model.to(device)


def predict_salary(
    model: SalaryQuantileNet,
    resume_embedding: np.ndarray,
    extra_features: np.ndarray | None = None,
    scaler: SalaryScaler | None = None,
) -> dict[str, float]:
    """Run inference and return predicted salary quantiles in USD.

    Args:
        model: A trained SalaryQuantileNet (already on the correct device).
        resume_embedding: (embedding_dim,) numpy vector.
        extra_features: Optional (n_extra,) numpy vector.
        scaler: Optional SalaryScaler used during training. When provided,
                predictions are inverse-transformed back to USD.

    Returns:
        Dict like {"q10": 60000.0, "q25": 75000.0, "q50": 95000.0,
                   "q75": 120000.0, "q90": 150000.0}
    """
    x = torch.tensor(resume_embedding, dtype=torch.float32).unsqueeze(0)
    if extra_features is not None:
        ef = torch.tensor(extra_features, dtype=torch.float32).unsqueeze(0)
        x = torch.cat([x, ef], dim=1)

    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        preds = model(x).squeeze(0).cpu().numpy()

    # Inverse-transform back to USD if a scaler was used
    if scaler is not None:
        preds = scaler.inverse_transform(preds)

    # Enforce monotonicity: sort so q10 <= q25 <= ... <= q90
    preds = np.sort(preds)

    return {
        f"q{int(q * 100)}": round(float(p), 2)
        for q, p in zip(QUANTILES, preds)
    }
