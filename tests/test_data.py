import numpy as np
import pytest

from ml_model_dojo.data.datasets import (CLASSIFICATION, REGRESSION, SIZE_PRESETS, DataConfig,
                                                       generate)
from ml_model_dojo.seeding import make_rng


def test_rng_is_reproducible():
    assert np.allclose(make_rng(3).normal(size=5), make_rng(3).normal(size=5))
    assert not np.allclose(make_rng(3).normal(size=5), make_rng(4).normal(size=5))


@pytest.mark.parametrize("ds", list(CLASSIFICATION))
@pytest.mark.parametrize("n", [40, 150, 600])
def test_classification_generators(ds, n):
    d = generate(DataConfig("classification", ds, n_samples=n), seed=1)
    X, y = d.X, d.y
    assert X.shape == (n, 2) and len(y) == n
    assert X.min() >= -1 and X.max() <= 1
    assert set(np.unique(y)) == {0, 1}
    assert abs(y.mean() - 0.5) < 0.05
    assert len(d.y_test) == round(n * 0.3) or abs(len(d.y_test) - n * 0.3) <= 1
    assert set(np.unique(d.y_train)) == {0, 1} and set(np.unique(d.y_test)) == {0, 1}


@pytest.mark.parametrize("ds", list(CLASSIFICATION))
def test_large_sizes(ds):
    for n in (3000, 15000):
        d = generate(DataConfig("classification", ds, n_samples=n), seed=2)
        assert len(d.y) == n


@pytest.mark.parametrize("ds", list(REGRESSION))
def test_regression_generators(ds):
    info = REGRESSION[ds]
    for n in SIZE_PRESETS.values():
        d = generate(DataConfig("regression", ds, n_samples=n), seed=1)
        assert d.X.shape == (n, info.n_inputs)
        assert d.X.min() >= -1 and d.X.max() <= 1
        assert np.isfinite(d.y).all()


def test_determinism_all_datasets():
    for problem, reg in (("classification", CLASSIFICATION), ("regression", REGRESSION)):
        for ds in reg:
            cfg = DataConfig(problem, ds, n_samples=150, noise=0.4)
            a, b = generate(cfg, 3), generate(cfg, 3)
            assert np.array_equal(a.X_train, b.X_train) and np.array_equal(a.y_test, b.y_test)
            c = generate(cfg, 4)
            assert not np.array_equal(a.X_train, c.X_train)


def test_class_balance_and_label_noise():
    d = generate(DataConfig("classification", "blobs_sep", 1000, minority_share=0.05), 1)
    assert abs(d.y.mean() - 0.05) < 0.01
    # stratification keeps the share in both splits
    assert abs(d.y_train.mean() - d.y_test.mean()) < 0.02
    clean = generate(DataConfig("classification", "blobs_sep", 1000), 1)
    noisy = generate(DataConfig("classification", "blobs_sep", 1000, label_noise=0.2), 1)
    assert np.array_equal(np.sort(clean.X, axis=0), np.sort(noisy.X, axis=0))


def test_split_has_no_leakage():
    d = generate(DataConfig("classification", "moons", 300), 1)
    tr = {tuple(r) for r in d.X_train.round(12)}
    te = {tuple(r) for r in d.X_test.round(12)}
    assert not tr & te


def test_extrapolation_mode():
    d = generate(DataConfig("regression", "lin", 200, test_mode="extrapolation"), 1)
    assert d.extrapolation
    assert d.X_train.max() <= 0.4 and d.X_test.min() >= 0.6


def test_minority_floor_on_tiny_data():
    d = generate(DataConfig("classification", "xor", 40, minority_share=0.02), 1)
    assert d.y.sum() >= 4
