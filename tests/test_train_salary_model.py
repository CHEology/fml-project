from __future__ import annotations

import numpy as np
import torch
from ml.salary_model import SalaryDataset, SalaryQuantileNet
from scripts.train_salary_model import train
from torch.utils.data import DataLoader


def test_train_writes_checkpoint_with_current_torch_scheduler(tmp_path) -> None:
    rng = np.random.default_rng(42)
    embeddings = rng.standard_normal((12, 4)).astype(np.float32)
    salaries = rng.standard_normal(12).astype(np.float32)

    train_loader = DataLoader(SalaryDataset(embeddings[:8], salaries[:8]), batch_size=4)
    val_loader = DataLoader(SalaryDataset(embeddings[8:], salaries[8:]), batch_size=4)
    model = SalaryQuantileNet(embedding_dim=4, dropout=0.0)
    output_path = tmp_path / "salary_model.pt"

    history = train(
        train_loader,
        val_loader,
        model,
        torch.device("cpu"),
        lr=1e-3,
        epochs=1,
        patience=1,
        weight_decay=0.0,
        quantile_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        output_path=str(output_path),
    )

    assert output_path.exists()
    assert len(history["train_loss"]) == 1
    assert len(history["val_loss"]) == 1
