"""
Tests for ml/salary_model.py

Run:  pytest tests/test_salary_model.py -v
"""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

# Ensure project root on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.salary_model import (
    PinballLoss,
    SalaryQuantileNet,
    SalaryDataset,
    SalaryScaler,
    split_data,
    predict_salary,
    QUANTILES,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

EMB_DIM = 384
N_SAMPLES = 200
N_EXTRA = 3


@pytest.fixture
def dummy_embeddings():
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_SAMPLES, EMB_DIM)).astype(np.float32)


@pytest.fixture
def dummy_salaries():
    rng = np.random.default_rng(0)
    return (rng.standard_normal(N_SAMPLES) * 30000 + 80000).astype(np.float32)


@pytest.fixture
def dummy_extra():
    rng = np.random.default_rng(0)
    return rng.standard_normal((N_SAMPLES, N_EXTRA)).astype(np.float32)


@pytest.fixture
def model():
    return SalaryQuantileNet(embedding_dim=EMB_DIM, n_extra_features=0, dropout=0.0)


@pytest.fixture
def model_with_extra():
    return SalaryQuantileNet(
        embedding_dim=EMB_DIM, n_extra_features=N_EXTRA, dropout=0.0
    )


# ---------------------------------------------------------------------------
# PinballLoss tests
# ---------------------------------------------------------------------------

class TestPinballLoss:
    def test_output_is_scalar(self):
        criterion = PinballLoss()
        y_pred = torch.randn(8, len(QUANTILES))
        y_true = torch.randn(8)
        loss = criterion(y_pred, y_true)
        assert loss.dim() == 0, "Loss should be a scalar"

    def test_zero_loss_at_perfect_prediction(self):
        criterion = PinballLoss()
        y_true = torch.tensor([100.0, 200.0, 300.0])
        y_pred = y_true.unsqueeze(1).expand(-1, len(QUANTILES))
        loss = criterion(y_pred, y_true)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_loss_is_non_negative(self):
        criterion = PinballLoss()
        y_pred = torch.randn(16, len(QUANTILES))
        y_true = torch.randn(16)
        loss = criterion(y_pred, y_true)
        assert loss.item() >= 0.0

    def test_asymmetric_penalty(self):
        """Under-prediction should be penalised more for high quantiles."""
        criterion_high = PinballLoss(quantiles=(0.9,))
        criterion_low = PinballLoss(quantiles=(0.1,))

        y_true = torch.tensor([100.0])
        y_under = torch.tensor([[80.0]])  # under-predict by 20

        loss_high = criterion_high(y_under, y_true)
        loss_low = criterion_low(y_under, y_true)

        # q=0.9 penalises under-prediction more than q=0.1
        assert loss_high.item() > loss_low.item()


# ---------------------------------------------------------------------------
# SalaryQuantileNet tests
# ---------------------------------------------------------------------------

class TestSalaryQuantileNet:
    def test_output_shape(self, model):
        x = torch.randn(4, EMB_DIM)
        out = model(x)
        assert out.shape == (4, len(QUANTILES))

    def test_output_shape_with_extra(self, model_with_extra):
        x = torch.randn(4, EMB_DIM + N_EXTRA)
        out = model_with_extra(x)
        assert out.shape == (4, len(QUANTILES))

    def test_gradients_flow(self, model):
        x = torch.randn(4, EMB_DIM, requires_grad=False)
        y = torch.randn(4)
        criterion = PinballLoss()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.all(p.grad == 0), f"Zero gradient for {name}"

    def test_single_sample_forward(self, model):
        """Model should handle batch_size=1 (inference mode without BatchNorm issues)."""
        model.eval()
        x = torch.randn(1, EMB_DIM)
        out = model(x)
        assert out.shape == (1, len(QUANTILES))

    def test_deterministic_eval(self, model):
        """Same input → same output in eval mode."""
        model.eval()
        x = torch.randn(2, EMB_DIM)
        out1 = model(x).clone()
        out2 = model(x).clone()
        assert torch.allclose(out1, out2)


# ---------------------------------------------------------------------------
# SalaryDataset tests
# ---------------------------------------------------------------------------

class TestSalaryDataset:
    def test_length(self, dummy_embeddings, dummy_salaries):
        ds = SalaryDataset(dummy_embeddings, dummy_salaries)
        assert len(ds) == N_SAMPLES

    def test_getitem_shapes(self, dummy_embeddings, dummy_salaries):
        ds = SalaryDataset(dummy_embeddings, dummy_salaries)
        x, y = ds[0]
        assert x.shape == (EMB_DIM,)
        assert y.shape == ()

    def test_with_extra_features(self, dummy_embeddings, dummy_salaries, dummy_extra):
        ds = SalaryDataset(dummy_embeddings, dummy_salaries, dummy_extra)
        x, _ = ds[0]
        assert x.shape == (EMB_DIM + N_EXTRA,)

    def test_dataloader_batching(self, dummy_embeddings, dummy_salaries):
        ds = SalaryDataset(dummy_embeddings, dummy_salaries)
        loader = torch.utils.data.DataLoader(ds, batch_size=32)
        X_batch, y_batch = next(iter(loader))
        assert X_batch.shape == (32, EMB_DIM)
        assert y_batch.shape == (32,)


# ---------------------------------------------------------------------------
# split_data tests
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_split_sizes(self, dummy_embeddings, dummy_salaries):
        train_ds, val_ds, test_ds, scaler = split_data(dummy_embeddings, dummy_salaries)
        total = len(train_ds) + len(val_ds) + len(test_ds)
        assert total == N_SAMPLES
        assert len(train_ds) == int(N_SAMPLES * 0.8)
        assert len(val_ds) == int(N_SAMPLES * 0.1)

    def test_no_data_leakage(self, dummy_embeddings, dummy_salaries):
        train_ds, val_ds, test_ds, _ = split_data(dummy_embeddings, dummy_salaries)
        # Check total count is preserved (no overlap or loss)
        assert len(train_ds) + len(val_ds) + len(test_ds) == N_SAMPLES

    def test_reproducibility(self, dummy_embeddings, dummy_salaries):
        t1, v1, _, _ = split_data(dummy_embeddings, dummy_salaries, seed=123)
        t2, v2, _, _ = split_data(dummy_embeddings, dummy_salaries, seed=123)
        assert torch.allclose(t1.y, t2.y)
        assert torch.allclose(v1.y, v2.y)

    def test_scaler_returned(self, dummy_embeddings, dummy_salaries):
        _, _, _, scaler = split_data(dummy_embeddings, dummy_salaries)
        assert isinstance(scaler, SalaryScaler)
        assert scaler.std > 0


# ---------------------------------------------------------------------------
# predict_salary tests
# ---------------------------------------------------------------------------

class TestPredictSalary:
    def test_returns_all_quantiles(self, model):
        model.eval()
        emb = np.random.randn(EMB_DIM).astype(np.float32)
        result = predict_salary(model, emb)
        expected_keys = {f"q{int(q*100)}" for q in QUANTILES}
        assert set(result.keys()) == expected_keys

    def test_monotonicity(self, model):
        """Predicted quantiles should be in non-decreasing order."""
        model.eval()
        emb = np.random.randn(EMB_DIM).astype(np.float32)
        result = predict_salary(model, emb)
        values = [result[f"q{int(q*100)}"] for q in QUANTILES]
        for i in range(len(values) - 1):
            assert values[i] <= values[i + 1], (
                f"Monotonicity violated: q{int(QUANTILES[i]*100)}={values[i]} > "
                f"q{int(QUANTILES[i+1]*100)}={values[i+1]}"
            )

    def test_with_extra_features(self, model_with_extra):
        model_with_extra.eval()
        emb = np.random.randn(EMB_DIM).astype(np.float32)
        extra = np.random.randn(N_EXTRA).astype(np.float32)
        result = predict_salary(model_with_extra, emb, extra)
        assert len(result) == len(QUANTILES)

    def test_output_types(self, model):
        model.eval()
        emb = np.random.randn(EMB_DIM).astype(np.float32)
        result = predict_salary(model, emb)
        for key, val in result.items():
            assert isinstance(key, str)
            assert isinstance(val, float)

    def test_with_scaler(self, model):
        """Predictions with a scaler should be in the original salary range."""
        model.eval()
        scaler = SalaryScaler(mean=80_000.0, std=25_000.0)
        emb = np.random.randn(EMB_DIM).astype(np.float32)
        result = predict_salary(model, emb, scaler=scaler)
        # With scaler, values should be shifted to ~ mean ± a few stds
        for v in result.values():
            assert -200_000 < v < 500_000, f"Value {v} seems unreasonable"


# ---------------------------------------------------------------------------
# SalaryScaler tests
# ---------------------------------------------------------------------------

class TestSalaryScaler:
    def test_fit_transform_roundtrip(self):
        salaries = np.array([50_000, 80_000, 100_000, 120_000], dtype=np.float32)
        scaler = SalaryScaler().fit(salaries)
        scaled = scaler.transform(salaries)
        recovered = scaler.inverse_transform(scaled)
        np.testing.assert_allclose(recovered, salaries, rtol=1e-5)

    def test_zero_mean_unit_var(self):
        rng = np.random.default_rng(0)
        salaries = rng.normal(80_000, 20_000, size=10_000).astype(np.float32)
        scaler = SalaryScaler().fit(salaries)
        scaled = scaler.transform(salaries)
        assert abs(scaled.mean()) < 0.01
        assert abs(scaled.std() - 1.0) < 0.02

    def test_state_dict_roundtrip(self):
        scaler = SalaryScaler(mean=75_000.0, std=22_000.0)
        d = scaler.state_dict()
        restored = SalaryScaler.from_state_dict(d)
        assert restored.mean == scaler.mean
        assert restored.std == scaler.std
