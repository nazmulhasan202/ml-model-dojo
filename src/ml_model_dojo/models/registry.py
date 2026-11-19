"""Model registry. To add a model, create a ModelSpec and add it to ``_SPECS``."""
from __future__ import annotations

from .base import ModelSpec
from .boosting import BoostSpec
from .forest import ForestSpec
from .knn import KnnSpec
from .linear import LinearSpec
from .mlp import MlpSpec
from .svm import SvmSpec
from .tree import TreeSpec

_SPECS: dict[str, ModelSpec] = {s.id: s for s in
                                [LinearSpec(), KnnSpec(), TreeSpec(), ForestSpec(), BoostSpec(), SvmSpec(), MlpSpec()]}
MODEL_ORDER = list(_SPECS)


def get_spec(model_id: str) -> ModelSpec:
    return _SPECS[model_id]


def specs_for(problem: str) -> list[ModelSpec]:
    return [s for s in _SPECS.values() if problem in s.supports]
