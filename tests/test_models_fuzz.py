"""Sweeps over data settings and hyperparameters. No GUI.

Exhaustive combinations are impossible, so there are three layers:
1. every data generator at extreme settings (shape, range, finite values, both classes),
2. one-factor sweeps: every hyperparameter at its minimum, maximum, middle and every choice,
3. seeded random combinations of data settings and hyperparameters.
"""
import itertools

import numpy as np
import pytest

from ml_model_dojo.data.datasets import (CLASSIFICATION, REGRESSION, SIZE_PRESETS, DataConfig, generate)
from ml_model_dojo.models import MODEL_ORDER, get_spec
from ml_model_dojo.models.base import make_grid
from ml_model_dojo.models.evaluate import evaluate, summarize
from ml_model_dojo.models.mlp import FEATURES_2D

pytestmark = pytest.mark.slow

# Keep iterative models quick without changing what is being tested.
SPEED = {"forest": {"n_estimators": 10}, "boosting": {"n_estimators": 10},
         "mlp": {"max_epochs": 6, "epochs_per_step": 3}, "linear": {"max_epochs": 8}}
CAPS = {("mlp", "max_epochs"): 40, ("forest", "n_estimators"): 40, ("boosting", "n_estimators"): 40}
PROBLEM_DATA = {
    "classification": [("moons", 100, {}), ("xor", 40, {"minority_share": 0.05}), ("blobs_sep", 20, {})],
    "regression": [("sine", 100, {}), ("inter", 100, {}), ("lin", 40, {"test_mode": "extrapolation"})],
}


def probe_values(model_id, p):
    if p.kind == "int":
        vals = [int(p.lo), int(p.hi), int((p.lo + p.hi) // 2)]
    elif p.kind == "float":
        vals = [float(p.lo), float(p.hi), float((p.lo + p.hi) / 2)]
    elif p.kind == "log":
        vals = [float(p.lo), float(p.hi), float(np.sqrt(p.lo * p.hi))]
    elif p.kind == "choice":
        vals = [v for v, _ in p.choices]
    elif p.kind == "bool":
        vals = [True, False]
    elif p.kind == "layers":
        vals = [[1], [16], [16] * 5, [1] * 5, [4, 8]]
    elif p.kind == "flags":
        keys = [k for k, _ in FEATURES_2D]
        vals = [[k] for k in keys[:2]] + [keys, [], ["x1x2"], ["sinx1", "sinx2"]]
    else:
        raise AssertionError(p.kind)
    cap = CAPS.get((model_id, p.key))
    if cap and p.kind == "int":
        vals = [min(v, cap) for v in vals]
    return vals


def run_one(model_id, problem, params, data, seed=1):
    spec = get_spec(model_id)
    run = spec.start(params, problem, seed, data)
    run.fit_initial()
    run.go_to_end()
    assert run.finished
    ev = evaluate(run)
    for split in ("train", "test"):
        pred = ev[split]["pred"]
        assert np.isfinite(pred).all(), (model_id, split)
        for k, v in ev[split].items():
            if isinstance(v, float) and k != "roc_auc":
                assert np.isfinite(v), (model_id, k, v)
    for n_in in {data.n_inputs}:
        g, G = make_grid(n_in, 30 if n_in == 2 else 50)
        p = run.predict(G)
        assert p.shape == (len(G),) and np.isfinite(p).all()
        if problem == "classification":
            assert p.min() >= -1e-9 and p.max() <= 1 + 1e-9
    assert isinstance(run.facts(), dict)
    run.extra()
    assert spec.summary(params, problem)
    assert run.predict_time(data.X_test, 1) >= 0
    return run


# --------------------------------------------------------------------------- data
def test_every_dataset_at_every_size_and_extreme_settings():
    problems = (("classification", CLASSIFICATION), ("regression", REGRESSION))
    sizes = list(SIZE_PRESETS.values()) + [20, 20000]
    count = 0
    for problem, reg in problems:
        for ds, n, noise, seed in itertools.product(reg, sizes, (0.0, 1.0), (1, 5)):
            if n >= 15000 and seed == 5:
                continue
            cfg = DataConfig(problem, ds, n, noise)
            d = generate(cfg, seed)
            assert len(d.y) == n and np.isfinite(d.X).all() and np.isfinite(d.y).all()
            assert d.X.min() >= -1 - 1e-9 and d.X.max() <= 1 + 1e-9
            assert len(d.y_train) > 0 and len(d.y_test) > 0
            count += 1
    assert count > 500


@pytest.mark.parametrize("ds", list(CLASSIFICATION))
def test_classification_extreme_balance_label_noise_and_split(ds):
    for n, share, ln, tf in itertools.product((20, 40, 150), (0.02, 0.5), (0.0, 0.3), (0.1, 0.5)):
        d = generate(DataConfig("classification", ds, n, 0.5, ln, share, tf), 3)
        assert len(d.y) == n and set(np.unique(d.y)) <= {0, 1}
        assert d.y.sum() >= 1 or ln > 0


@pytest.mark.parametrize("ds", list(REGRESSION))
@pytest.mark.parametrize("mode", ["random", "extrapolation"])
def test_regression_modes_and_splits(ds, mode):
    for n, tf in itertools.product((20, 150), (0.1, 0.5)):
        d = generate(DataConfig("regression", ds, n, 0.5, test_frac=tf, test_mode=mode), 2)
        assert len(d.y) == n and len(d.y_test) >= 2


# --------------------------------------------------------------------------- one-factor sweeps
@pytest.mark.parametrize("model_id", MODEL_ORDER)
@pytest.mark.parametrize("problem", ["classification", "regression"])
def test_one_factor_sweep(model_id, problem):
    spec = get_spec(model_id)
    base = {**spec.default_params(problem), **SPEED.get(model_id, {})}
    datasets = [generate(DataConfig(problem, ds, n, 0.3, **kw), 1) for ds, n, kw in PROBLEM_DATA[problem]]
    n_runs = 0
    for p in spec.param_schema(problem):
        for v in probe_values(model_id, p):
            params = {**base, p.key: v}
            for data in datasets:
                run_one(model_id, problem, params, data)
                n_runs += 1
    assert n_runs > 20


# --------------------------------------------------------------------------- random combinations
@pytest.mark.parametrize("model_id", MODEL_ORDER)
@pytest.mark.parametrize("problem", ["classification", "regression"])
def test_random_combinations(model_id, problem):
    spec = get_spec(model_id)
    rng = np.random.default_rng(MODEL_ORDER.index(model_id) * 10 + (problem == "regression"))
    reg = CLASSIFICATION if problem == "classification" else REGRESSION
    ds_ids = list(reg)
    schema = spec.param_schema(problem)
    for i in range(25):
        n = int(rng.choice([20, 40, 150, 600]))
        kw = {}
        if problem == "classification":
            kw = dict(label_noise=float(rng.choice([0, 0.1, 0.3])), minority_share=float(rng.choice([0.02, 0.2, 0.5])))
        cfg = DataConfig(problem, ds_ids[int(rng.integers(len(ds_ids)))], n, float(rng.uniform(0, 1)),
                         test_frac=float(rng.choice([0.1, 0.3, 0.5])),
                         test_mode=str(rng.choice(["random", "extrapolation"])) if problem == "regression" else "random", **kw)
        data = generate(cfg, int(rng.integers(1, 6)))
        params = {**spec.default_params(problem), **SPEED.get(model_id, {})}
        for p in schema:
            if p.key in SPEED.get(model_id, {}) and p.key != "max_epochs":
                continue
            vals = probe_values(model_id, p)
            params[p.key] = vals[int(rng.integers(len(vals)))]
        run_one(model_id, problem, params, data, seed=int(rng.integers(1, 6)))


# --------------------------------------------------------------------------- misc invariants
@pytest.mark.parametrize("model_id", MODEL_ORDER)
def test_summarize_is_picklable_and_stage_zero_is_safe(model_id):
    import pickle
    spec = get_spec(model_id)
    data = generate(DataConfig("classification", "moons", 100), 1)
    params = {**spec.default_params("classification"), **SPEED.get(model_id, {})}
    run = spec.start(params, "classification", 1, data)
    run.fit_initial()
    assert run.stage == 0 and not run.finished
    run.facts()
    run.extra()
    run.step()
    run.go_to_end()
    pickle.dumps(summarize(run))
    # stepping past the end is harmless
    assert run.step() is False


@pytest.mark.parametrize("model_id", ["tree", "forest", "boosting"])
def test_replay_after_reset(model_id):
    spec = get_spec(model_id)
    data = generate(DataConfig("classification", "moons", 100), 1)
    params = {**spec.default_params("classification"), **SPEED.get(model_id, {})}
    run = spec.start(params, "classification", 1, data)
    run.fit_initial()
    run.go_to_end()
    final = run.predict(data.X_test).copy()
    run.reset()
    assert run.stage == 0
    run.go_to_end()
    assert np.allclose(final, run.predict(data.X_test))
