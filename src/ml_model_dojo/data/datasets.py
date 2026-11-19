"""Synthetic dataset generators with one common interface.

Use ``generate(config, seed)`` to get a ``Dataset`` that already contains the
train/test split. Everything is derived from seeded generators, so the same
config and seed always give the same result.

To add a dataset, write a sampler (classification) or a truth function
(regression), then register it in ``CLASSIFICATION`` or ``REGRESSION``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Callable

import numpy as np
from sklearn.model_selection import train_test_split

from ..seeding import SALT_DATA, make_rng

SIZE_PRESETS = {"Tiny (40)": 40, "Small (150)": 150, "Medium (600)": 600,
                "Large (3,000)": 3000, "Very large (15,000)": 15000}
MIN_SAMPLES, MAX_SAMPLES = 20, 20000
MIN_MINORITY = 4          # keeps stratified splitting possible
EXTRAP_TRAIN = (-1.0, 0.4)
EXTRAP_TEST = (0.6, 1.0)


@dataclass(frozen=True)
class DataConfig:
    problem: str = "classification"      # "classification" or "regression"
    dataset: str = "xor"
    n_samples: int = 150
    noise: float = 0.3                   # 0 to 1
    label_noise: float = 0.0             # 0 to 0.3
    minority_share: float = 0.5          # 0.02 to 0.5
    test_frac: float = 0.3               # 0.1 to 0.5
    test_mode: str = "random"            # "random" or "extrapolation"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "DataConfig":
        valid = {k: v for k, v in d.items() if k in DataConfig.__dataclass_fields__}
        return DataConfig(**valid)

    def with_(self, **kw) -> "DataConfig":
        return replace(self, **kw)


@dataclass
class DatasetInfo:
    id: str
    name: str
    problem: str
    description: str
    teaches: str
    tag: str                      # Easy, Moderate, Hard (classification) or "Level n"
    level: int                    # sorting key
    n_inputs: int = 2
    sampler: Callable | None = None      # classification: (rng, label, n, noise) -> (n, 2)
    truth: Callable | None = None        # regression: noise-free function of X
    noise_mult: float = 1.0
    post: Callable | None = None         # regression: custom noise (rng, X, y_clean, sigma) -> y


@dataclass
class Dataset:
    config: DataConfig
    seed: int
    info: DatasetInfo
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    extrapolation: bool = False

    @property
    def X(self) -> np.ndarray:
        return np.vstack([self.X_train, self.X_test])

    @property
    def y(self) -> np.ndarray:
        return np.concatenate([self.y_train, self.y_test])

    @property
    def is_test(self) -> np.ndarray:
        return np.r_[np.zeros(len(self.y_train), bool), np.ones(len(self.y_test), bool)]

    @property
    def problem(self) -> str:
        return self.config.problem

    @property
    def n_inputs(self) -> int:
        return self.X_train.shape[1]

    @property
    def meta(self) -> dict:
        return {"name": self.info.name, "description": self.info.description,
                "teaches": self.info.teaches, "tag": self.info.tag}


# --------------------------------------------------------------------------
# Classification samplers: each returns n points of one class.
# noise in [0, 1] is mapped to a dataset-specific scale.
# --------------------------------------------------------------------------
def _clip(X: np.ndarray) -> np.ndarray:
    return np.clip(X, -1.0, 1.0)


def _gauss(rng, center, std, n, cov=None):
    if cov is None:
        return rng.normal(center, std, size=(n, 2))
    return rng.multivariate_normal(center, cov, size=n)


def _blobs_sep(rng, k, n, noise):
    c = (-0.5, -0.5) if k == 0 else (0.5, 0.5)
    return _clip(_gauss(rng, c, 0.10 + 0.15 * noise, n))


def _blobs_overlap(rng, k, n, noise):
    c = (-0.3, -0.3) if k == 0 else (0.3, 0.3)
    return _clip(_gauss(rng, c, 0.25 + 0.20 * noise, n))


def _elliptical(rng, k, n, noise):
    s = 1.0 + 0.5 * noise
    if k == 0:
        cov = np.eye(2) * (0.22 * s) ** 2
    else:
        a = np.deg2rad(45)
        R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        cov = R @ np.diag([(0.65 * s) ** 2, (0.12 * s) ** 2]) @ R.T
    return _clip(_gauss(rng, (0, 0), None, n, cov))


def _xor(rng, k, n, noise):
    centers = [(0.5, 0.5), (-0.5, -0.5)] if k == 0 else [(0.5, -0.5), (-0.5, 0.5)]
    which = rng.integers(0, 2, n)
    base = np.array(centers)[which]
    return _clip(base + rng.normal(0, 0.15 + 0.20 * noise, size=(n, 2)))


def _sample_region(rng, n, label_fn, k, jitter):
    """Rejection sampling: uniform points whose (jittered) rule value equals k."""
    out: list[np.ndarray] = []
    got = 0
    while got < n:
        Z = rng.uniform(-1, 1, size=(max(4 * n, 64), 2))
        lab = label_fn(Z + rng.normal(0, jitter, size=Z.shape)) if jitter > 0 else label_fn(Z)
        sel = Z[lab == k]
        out.append(sel)
        got += len(sel)
    return np.vstack(out)[:n]


def _checker(rng, k, n, noise):
    def lab(Z):
        return (np.floor((Z[:, 0] + 1) * 2) + np.floor((Z[:, 1] + 1) * 2)).astype(int) % 2
    return _clip(_sample_region(rng, n, lab, k, 0.0) + rng.normal(0, 0.10 * noise, size=(n, 2)))


def _rings(rng, k, n, noise):
    r0, r1 = (0.35, 0.8)
    r = (r0 if k == 0 else r1) + rng.normal(0, 0.04 + 0.12 * noise, n)
    t = rng.uniform(0, 2 * np.pi, n)
    return _clip(np.c_[r * np.cos(t), r * np.sin(t)])


def _moons(rng, k, n, noise):
    t = rng.uniform(0, np.pi, n)
    if k == 0:
        pts = np.c_[np.cos(t), np.sin(t)]
    else:
        pts = np.c_[1 - np.cos(t), 0.5 - np.sin(t)]
    pts = (pts - np.array([0.5, 0.25])) / 1.6
    return _clip(pts + rng.normal(0, (0.05 + 0.25 * noise) / 1.6, size=(n, 2)))


def _spirals(rng, k, n, noise):
    r = np.sqrt(rng.uniform(0.01, 1.0, n)) * 0.92
    ang = 3.2 * np.pi * r / 0.92 + k * np.pi
    pts = np.c_[r * np.cos(ang), r * np.sin(ang)]
    return _clip(pts + rng.normal(0, 0.015 + 0.07 * noise, size=(n, 2)))


_MULTI = {0: [(-0.6, 0.6), (0.0, -0.6), (0.6, 0.5)],
          1: [(-0.6, -0.5), (0.0, 0.5), (0.6, -0.6)]}


def _multimodal(rng, k, n, noise):
    centers = np.array(_MULTI[k])
    base = centers[rng.integers(0, 3, n)]
    return _clip(base + rng.normal(0, 0.12 + 0.12 * noise, size=(n, 2)))


def _axis_rule(rng, k, n, noise):
    return _sample_region(rng, n, lambda Z: ((Z[:, 0] > 0.2) & (Z[:, 1] < 0.3)).astype(int), k, 0.15 * noise)


def _product(rng, k, n, noise):
    return _sample_region(rng, n, lambda Z: (Z[:, 0] * Z[:, 1] > 0.25).astype(int), k, 0.15 * noise)


def _band(rng, k, n, noise):
    return _sample_region(rng, n, lambda Z: (np.abs(Z[:, 0] - Z[:, 1]) < 0.35).astype(int), k, 0.15 * noise)


def _c(id_, name, desc, teaches, tag, sampler):
    lvl = {"Easy": 1, "Moderate": 2, "Hard": 3}[tag]
    return DatasetInfo(id_, name, "classification", desc, teaches, tag, lvl, 2, sampler=sampler)


CLASSIFICATION: dict[str, DatasetInfo] = {d.id: d for d in [
    _c("blobs_sep", "Linearly separable, well separated", "Two Gaussian blobs far apart.",
       "When a straight line is enough, every model works and the simplest one wins.", "Easy", _blobs_sep),
    _c("blobs_overlap", "Linearly separable, overlapping", "The same layout with a larger spread.",
       "Overlap creates irreducible error. No model can reach 100 percent on new data.", "Easy", _blobs_overlap),
    _c("elliptical", "Elliptical classes", "Two Gaussians with different covariance, one rotated.",
       "The best boundary is curved (quadratic). A straight line loses accuracy.", "Moderate", _elliptical),
    _c("xor", "XOR (non-linear)", "Four Gaussian quadrants with alternating classes.",
       "No single straight line separates XOR. Trees, kernels, and networks can.", "Moderate", _xor),
    _c("checker", "Checkerboard (4 by 4)", "Alternating axis-aligned squares.",
       "Complex axis-aligned structure rewards trees and flexible models, and needs more data.", "Hard", _checker),
    _c("rings", "Concentric rings", "A small ring inside a large ring.",
       "The classes are not linearly separable, but radius is an easy feature to engineer.", "Moderate", _rings),
    _c("moons", "Two moons", "Two interleaved half circles.",
       "A classic test of flexible boundaries, and of overfitting when noise is high.", "Moderate", _moons),
    _c("spirals", "Two spirals", "Two interleaved spiral arms.",
       "The hardest smooth boundary. It needs high capacity and enough data.", "Hard", _spirals),
    _c("multimodal", "Multimodal classes", "Each class is made of three separate clusters.",
       "A class does not have to be one blob. Local models handle this naturally.", "Moderate", _multimodal),
    _c("axis_rule", "Axis-aligned rule (tree-friendly)", "Class 1 when x1 is above 0.2 and x2 is below 0.3.",
       "Threshold rules are exactly what decision trees learn, with a small and readable model.", "Easy", _axis_rule),
    _c("product", "Product threshold (interaction)", "Class 1 when x1 times x2 is above 0.25.",
       "A hyperbolic boundary from an interaction, like torque times speed. Linear models fail unless the product is engineered.", "Hard", _product),
    _c("band", "Diagonal band", "Class 1 when the absolute difference of x1 and x2 is small.",
       "The class is not monotonic along any single direction, so a plain linear model fails.", "Moderate", _band),
]}


# --------------------------------------------------------------------------
# Regression truth functions.
# --------------------------------------------------------------------------
def _x(X):
    return X[:, 0]


def _outlier_post(rng, X, y, sigma):
    y = y + rng.normal(0, sigma, len(y))
    k = max(1, int(round(0.07 * len(y))))
    idx = rng.choice(len(y), k, replace=False)
    y[idx] += rng.choice([-1, 1], k) * rng.uniform(1.5, 3.0, k)
    return y


def _hetero_post(rng, X, y, sigma):
    scale = 0.15 + 1.85 * (_x(X) + 1) / 2
    return y + rng.normal(0, 1, len(y)) * sigma * scale


def _r(id_, name, desc, teaches, level, n_inputs, truth, mult=1.0, post=None):
    return DatasetInfo(id_, name, "regression", desc, teaches, f"Level {level}", level,
                       n_inputs, truth=truth, noise_mult=mult, post=post)


_REG_LIST = [
    _r("lin", "Linear (1 input)", "y = 2x + 0.5 plus noise.",
       "A linear model is the right tool and needs very little data.", 1, 1, lambda X: 2 * _x(X) + 0.5),
    _r("lin_noisy", "Linear, high noise (1 input)", "The same line with three times the noise.",
       "Noise limits every model. Flexible models start to chase it.", 2, 1, lambda X: 2 * _x(X) + 0.5, 3.0),
    _r("lin_outliers", "Linear trend with outliers (1 input)", "A line where about 7 percent of points are shifted far away.",
       "Squared error is sensitive to outliers. Averaging many trees is more stable than boosting them.", 2, 1,
       lambda X: 2 * _x(X) + 0.5, 1.0, _outlier_post),
    _r("hinge", "Piecewise linear with a kink (1 input)", "Flat slope that turns steeper at x = 0.1.",
       "A single line misses the kink. Trees and polynomials approximate it.", 3, 1,
       lambda X: np.where(_x(X) < 0.1, 0.2 * _x(X), 0.2 * _x(X) + 2.2 * (_x(X) - 0.1))),
    _r("quad", "Quadratic (1 input)", "A parabola.",
       "Polynomial features turn a linear model into a curve fitter.", 3, 1, lambda X: 2.0 * _x(X) ** 2 - 0.5),
    _r("hetero", "Heteroscedastic noise (1 input)", "A line whose noise grows with x.",
       "Constant-variance assumptions break, and fits are less certain on the right side.", 3, 1,
       lambda X: 1.5 * _x(X), 1.0, _hetero_post),
    _r("cubic", "Cubic (1 input)", "A cubic with two turning points.",
       "Degree 3 is enough. Degree 1 underfits and a high degree may wiggle.", 4, 1,
       lambda X: 3.0 * _x(X) ** 3 - 1.6 * _x(X)),
    _r("sine", "Sinusoidal (1 input)", "One full sine period over the range.",
       "A smooth non-linear curve. Trees approximate it with steps.", 5, 1, lambda X: np.sin(2 * np.pi * _x(X))),
    _r("step", "Step function (1 input)", "Piecewise constant with three levels.",
       "A tree-friendly shape. Linear models smear the jumps.", 5, 1,
       lambda X: np.where(_x(X) < -0.3, -0.5, np.where(_x(X) < 0.4, 1.0, 0.0))),
    _r("add2", "Additive linear (2 inputs)", "y = 1.5 x1 minus 2 x2 plus noise.",
       "With two inputs and a linear truth, linear regression is hard to beat.", 2, 2,
       lambda X: 1.5 * X[:, 0] - 2.0 * X[:, 1]),
    _r("power", "Power-plant style (2 inputs)", "Mostly linear with mild curvature near the extremes.",
       "Real process data are often nearly linear, so a simple model is usually enough.", 3, 2,
       lambda X: 1.0 * X[:, 0] - 0.8 * X[:, 1] + 0.25 * X[:, 0] ** 3 - 0.15 * X[:, 1] ** 3, 1.5),
    _r("inter", "Interaction (2 inputs)", "y = 2 x1 x2 plus small linear terms.",
       "A linear model fails unless the product feature is engineered. Trees approximate it.", 6, 2,
       lambda X: 2.0 * X[:, 0] * X[:, 1] + 0.3 * X[:, 0] - 0.2 * X[:, 1]),
    _r("smooth2", "Smooth non-linear (2 inputs)", "y = sin(3 x1) plus x2 squared.",
       "A smooth surface that needs curvature in both inputs.", 7, 2,
       lambda X: np.sin(3 * X[:, 0]) + X[:, 1] ** 2),
    _r("thresh2", "Threshold interaction (2 inputs)", "A jump that appears only when x1 and x2 are both positive.",
       "A jump in one corner is easy for trees and hard for smooth models.", 9, 2,
       lambda X: 2.0 * ((X[:, 0] > 0) & (X[:, 1] > 0)) + 0.3 * X[:, 0]),
]
REGRESSION: dict[str, DatasetInfo] = {d.id: d for d in sorted(_REG_LIST, key=lambda d: d.level)}


def dataset_infos(problem: str) -> list[DatasetInfo]:
    src = CLASSIFICATION if problem == "classification" else REGRESSION
    return list(src.values())


def get_info(problem: str, dataset_id: str) -> DatasetInfo:
    src = CLASSIFICATION if problem == "classification" else REGRESSION
    return src[dataset_id]


# --------------------------------------------------------------------------
# Generation
# --------------------------------------------------------------------------
def _gen_classification(cfg: DataConfig, seed: int, info: DatasetInfo) -> Dataset:
    rng = make_rng(seed, SALT_DATA)
    n = int(np.clip(cfg.n_samples, MIN_SAMPLES, MAX_SAMPLES))
    n1 = int(round(n * cfg.minority_share))
    n1 = min(max(n1, MIN_MINORITY), n // 2)
    n0 = n - n1
    X = np.vstack([info.sampler(rng, 0, n0, cfg.noise), info.sampler(rng, 1, n1, cfg.noise)])
    y = np.r_[np.zeros(n0, int), np.ones(n1, int)]
    n_flip = int(round(n * cfg.label_noise))
    if n_flip:
        idx = rng.choice(n, n_flip, replace=False)
        y[idx] = 1 - y[idx]
    perm = rng.permutation(n)
    X, y = X[perm], y[perm]
    try:
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_frac, stratify=y, random_state=seed)
    except ValueError:  # a class became too small after label flips
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_frac, random_state=seed)
    return Dataset(cfg, seed, info, Xtr, Xte, ytr, yte)


def _noisy_target(rng, info: DatasetInfo, X: np.ndarray, noise: float) -> np.ndarray:
    sigma = (0.03 + 0.45 * noise) * info.noise_mult
    clean = np.asarray(info.truth(X), float)
    if info.post is not None:
        return info.post(rng, X, clean.copy(), sigma)
    return clean + rng.normal(0, sigma, len(clean))


def _gen_regression(cfg: DataConfig, seed: int, info: DatasetInfo) -> Dataset:
    rng = make_rng(seed, SALT_DATA)
    n = int(np.clip(cfg.n_samples, MIN_SAMPLES, MAX_SAMPLES))
    d = info.n_inputs
    if cfg.test_mode == "extrapolation" and d == 1:
        n_te = max(int(round(n * cfg.test_frac)), 2)
        n_tr = max(n - n_te, 2)
        Xtr = rng.uniform(*EXTRAP_TRAIN, size=(n_tr, 1))
        Xte = rng.uniform(*EXTRAP_TEST, size=(n_te, 1))
        ytr = _noisy_target(rng, info, Xtr, cfg.noise)
        yte = _noisy_target(rng, info, Xte, cfg.noise)
        return Dataset(cfg, seed, info, Xtr, Xte, ytr, yte, extrapolation=True)
    X = rng.uniform(-1, 1, size=(n, d))
    y = _noisy_target(rng, info, X, cfg.noise)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=cfg.test_frac, random_state=seed)
    return Dataset(cfg, seed, info, Xtr, Xte, ytr, yte)


def generate(config: DataConfig, seed: int) -> Dataset:
    """Generate a dataset and its train/test split. Deterministic for (config, seed)."""
    info = get_info(config.problem, config.dataset)
    if config.problem == "classification":
        return _gen_classification(config, seed, info)
    return _gen_regression(config, seed, info)


def default_dataset(problem: str) -> str:
    return "xor" if problem == "classification" else "lin"
