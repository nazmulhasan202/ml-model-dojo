import numpy as np
import pytest

from ml_model_dojo.data.datasets import CLASSIFICATION, REGRESSION, DataConfig, Dataset, generate
from ml_model_dojo.models import MODEL_ORDER, get_spec, specs_for
from ml_model_dojo.models.evaluate import evaluate


def fit(model_id, data, problem=None, **params):
    spec = get_spec(model_id)
    problem = problem or data.problem
    run = spec.start({**spec.default_params(problem), **params}, problem, 1, data)
    run.fit_initial()
    run.go_to_end()
    return run


def make(problem="classification", ds="xor", n=300, **kw):
    return generate(DataConfig(problem, ds, n, **kw), 1)


def test_ols_recovers_coefficients():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (200, 2))
    y = 1.5 * X[:, 0] - 2.0 * X[:, 1] + 0.7
    d = Dataset(DataConfig("regression", "add2"), 1, REGRESSION["add2"], X[:150], X[150:], y[:150], y[150:])
    run = fit("linear", d, penalty="none", standardize=True)
    rows = dict(run.extra()["coef_rows"])
    assert rows["x1"] == pytest.approx(1.5, abs=1e-6)
    assert rows["x2"] == pytest.approx(-2.0, abs=1e-6)
    assert rows["intercept"] == pytest.approx(0.7, abs=1e-6)


def test_logistic_separates_blobs():
    d = make(ds="blobs_sep", noise=0.1)
    run = fit("linear", d)
    assert evaluate(run)["test"]["accuracy"] > 0.95


def test_logistic_fails_xor_and_poly2_fixes_it():
    d = make(ds="xor", n=600, noise=0.2)
    assert evaluate(fit("linear", d))["test"]["accuracy"] < 0.65
    assert evaluate(fit("linear", d, degree=2))["test"]["accuracy"] > 0.9


def test_depth1_tree_finds_threshold():
    rng = np.random.default_rng(1)
    X = rng.uniform(-1, 1, (200, 2))
    y = (X[:, 0] > 0.3).astype(int)
    d = Dataset(DataConfig(), 1, CLASSIFICATION["xor"], X[:140], X[140:], y[:140], y[140:])
    run = fit("tree", d, max_depth=1)
    t = run.extra()["tree"]
    assert t["feature"][0] == 0 and t["threshold"][0] == pytest.approx(0.3, abs=0.05)
    assert evaluate(run)["test"]["accuracy"] > 0.95


def test_tree_steps_grow_leaves():
    d = make(ds="moons")
    run = get_spec("tree").start(get_spec("tree").default_params("classification") | {"max_depth": 8}, "classification", 1, d)
    run.fit_initial()
    leaves = []
    while run.step():
        leaves.append(run.facts()["Leaves"])
    leaves.append(run.facts()["Leaves"])
    assert leaves == sorted(leaves) and leaves[-1] == run.full.get_n_leaves()


def test_knn1_memorizes_training_set():
    d = make(ds="moons", n=200, noise=0.5)
    run = fit("knn", d, k=1)
    assert evaluate(run)["train"]["accuracy"] == 1.0


def test_forest_beats_single_tree_on_average():
    wins = []
    for seed in range(1, 6):
        d = generate(DataConfig("classification", "moons", 300, noise=0.7), seed)
        spec_t, spec_f = get_spec("tree"), get_spec("forest")
        t = spec_t.start(spec_t.default_params("classification") | {"max_depth": 12}, "classification", seed, d)
        f = spec_f.start(spec_f.default_params("classification"), "classification", seed, d)
        for r in (t, f):
            r.fit_initial(); r.go_to_end()
        wins.append(evaluate(f)["test"]["accuracy"] - evaluate(t)["test"]["accuracy"])
    assert np.mean(wins) > 0


def test_xgboost_training_loss_decreases():
    d = make(ds="moons", n=300)
    run = fit("boosting", d, n_estimators=50)
    tr = run.curves["train_loss"]
    assert tr[-1] < tr[0] and all(b <= a + 1e-9 for a, b in zip(tr, tr[1:]))
    assert run.predict(d.X_test[:3]).shape == (3,)


def test_forest_stage_prediction_changes():
    d = make(ds="moons")
    run = get_spec("forest").start(get_spec("forest").default_params("classification") | {"n_estimators": 20}, "classification", 1, d)
    run.fit_initial()
    run.step()
    p1 = run.predict(d.X_test)
    run.go_to_end()
    assert not np.allclose(p1, run.predict(d.X_test))


def test_svm_rbf_and_approximation_notice():
    d = make(ds="rings", n=300)
    run = fit("svm", d)
    assert evaluate(run)["test"]["accuracy"] > 0.9 and run.extra()["support"] is not None
    big = make(ds="rings", n=4500)
    run = fit("svm", big)
    assert run.notice and run.extra()["support"] is None


def test_mlp_learns_xor():
    d = make(ds="xor", n=400, noise=0.2)
    run = fit("mlp", d, max_epochs=150, epochs_per_step=10)
    assert evaluate(run)["test"]["accuracy"] > 0.9


@pytest.mark.parametrize("model_id", MODEL_ORDER)
@pytest.mark.parametrize("problem,ds", [("classification", "moons"), ("regression", "lin"), ("regression", "inter")])
def test_every_model_fits_and_is_deterministic(model_id, problem, ds):
    spec = get_spec(model_id)
    d = generate(DataConfig(problem, ds, 150), 2)
    params = spec.default_params(problem)
    if model_id in ("forest", "boosting"):
        params["n_estimators"] = 15
    if model_id == "mlp":
        params["max_epochs"] = 20
    out = []
    for _ in range(2):
        run = spec.start(params, problem, 2, d)
        run.fit_initial(); run.go_to_end()
        out.append(run.predict(d.X_test))
    assert np.allclose(out[0], out[1])
    assert spec.summary(params, problem)
    assert run.facts() is not None


def test_regression_extrapolation_behavior():
    d = generate(DataConfig("regression", "lin", 300, noise=0.1, test_mode="extrapolation"), 1)
    lin = evaluate(fit("linear", d))["test"]["r2"]
    forest = evaluate(fit("forest", d))["test"]["r2"]
    assert lin > 0.8 and forest < lin


def test_specs_for_problem():
    assert len(specs_for("classification")) == 7 and len(specs_for("regression")) == 7


@pytest.mark.parametrize("model_id", MODEL_ORDER)
@pytest.mark.parametrize("problem", ["classification", "regression"])
def test_clean_params_repairs_garbage(model_id, problem):
    spec = get_spec(model_id)
    defaults = spec.default_params(problem)
    garbage = {}
    for p in spec.param_schema(problem):
        garbage[p.key] = {"int": "abc", "float": float("nan"), "log": -5, "choice": "weird", "bool": "yes",
                          "layers": "oops", "flags": 7}[p.kind]
    garbage["unknown_key"] = 1
    cleaned = spec.clean_params(garbage, problem)
    assert set(cleaned) == set(defaults)
    for p in spec.param_schema(problem):
        if p.kind in ("int", "float", "log"):
            assert p.lo <= cleaned[p.key] <= p.hi
        if p.kind == "choice":
            assert cleaned[p.key] in [v for v, _ in p.choices]
    # out-of-range numbers are clipped, not rejected
    extreme = {p.key: 10**9 for p in spec.param_schema(problem) if p.kind in ("int", "float", "log")}
    for k, v in spec.clean_params(extreme, problem).items():
        if k in extreme:
            assert v == next(p.hi for p in spec.param_schema(problem) if p.key == k)
    # and the repaired settings still train
    data = generate(DataConfig(problem, "moons" if problem == "classification" else "sine", 60), 1)
    run = spec.start({**cleaned, **({"n_estimators": 5} if "n_estimators" in cleaned else {}),
                      **({"max_epochs": 10} if "max_epochs" in cleaned else {})}, problem, 1, data)
    run.fit_initial()
    run.go_to_end()
