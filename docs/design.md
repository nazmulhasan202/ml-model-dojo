# Design notes

## Layers

```
data/       datasets.py       generators, registry, train/test split
models/     base.py           ParamSpec, ModelSpec, Run (staged training), helpers
            linear.py ... mlp.py, registry.py, evaluate.py
workers/    qt_workers.py     QThreadPool task for the first fit of a model tab
            compare_runner.py picklable jobs for the process pool, cross-validation
ui/         main_window.py, data_panel.py, model_tab.py, param_panel.py,
            plots.py, side_plots.py, metrics_panel.py, details.py,
            compare_tab.py, dialogs.py, theme.py
scenarios/  definitions.py    the eleven guided lessons
            verify.py         runs them and writes docs/scenarios.md
```

The `models` and `data` packages do not import Qt, so they are tested without a GUI and the Compare jobs can run in worker processes.

## The Run abstraction

`ModelSpec` describes a family (parameters, concept card, summary text). `spec.start(params, problem, seed, data)` returns a `Run`:

* `fit_initial()` does the heavy work once, in a background thread.
* `step()` moves one stage forward (one tree, one epoch, one split). `n_stages` tells how many there are.
* `predict(X)` returns the class 1 probability, or the regression value, at the current stage. Models without probabilities (SVM) return a sigmoid of the decision function.
* `facts()` and `extra()` feed the model facts and the details tab.

Stage 0 always means "not trained yet", so the plot shows only data. Models that replay (tree, forest, boosting) can reset to stage 0 without refitting.

## Reproducibility

`seeding.make_rng(seed, salt)` gives independent streams for data, split and model initialization. Estimators receive `random_state` (or the XGBoost `random_state`) derived from the app seed. Thread counts are fixed to one. The tests check that data, splits and model predictions are identical on repeated runs.

## Leakage rules

* The split is made first. Models, scalers and polynomial features only see the training part.
* Cross-validation folds are cut from the training part only.
* The boosting loss curve shows test loss for teaching purposes, and the tooltip warns that choosing the number of trees from it leaks information.

## Compare tab

* Settings are read live from the model tabs through `ModelTab.current_params()`. Nothing is duplicated.
* A job key is the hash of (data config, seed, model, parameters, options). Unchanged models are served from the cache.
* A persistent `ProcessPoolExecutor` (spawn start method) is warmed up when the tab is first shown. Cancel terminates the worker processes and the pool is recreated on the next run.
* The stale badge compares the parameters used in the last run with the live ones.
