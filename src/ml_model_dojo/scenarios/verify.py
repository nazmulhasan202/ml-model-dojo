"""Run every scenario for real and write the numbers to docs/scenarios.md.

Usage: ``python -m ml_model_dojo.scenarios.verify [--only id1,id2] [--no-write]``
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from ..data.datasets import DataConfig
from ..models import get_spec
from ..seeding import SEEDS
from ..workers.compare_runner import make_job, run_job
from .definitions import SCENARIOS


def scenario_jobs(sc: dict, data_over: dict | None = None, param_over: dict | None = None):
    d = {**sc["data"], **(data_over or {})}
    cfg = DataConfig.from_dict(d)
    jobs = []
    for mid in sc["models"]:
        params = {**get_spec(mid).default_params(cfg.problem), **sc.get("params", {}).get(mid, {}),
                  **(param_over or {}).get(mid, {})}
        jobs.append(make_job(cfg, sc["seed"], mid, params, True, False))
        if sc.get("compare", {}).get("multiseed"):
            jobs += [make_job(cfg, s, mid, params, False, False) for s in SEEDS if s != sc["seed"]]
    return cfg, jobs


def run_scenario(ex, sc, data_over=None, param_over=None) -> dict:
    cfg, jobs = scenario_jobs(sc, data_over, param_over)
    results = list(ex.map(run_job, jobs))
    out: dict = {}
    for r in results:
        e = out.setdefault(r["model_id"], {"main": None, "seeds": []})
        if r["main"]:
            e["main"] = r
        e["seeds"].append(r)
    return {"cfg": cfg, "models": out}


def fmt(sc, res) -> list[str]:
    key = sc["key"]
    prob = res["cfg"].problem
    main = "accuracy" if prob == "classification" else "r2"
    lines = [f"| Model | Train {main} | Test {main} | Test {key} | Train time (ms) |", "|---|---|---|---|---|"]
    for mid, e in res["models"].items():
        m = e["main"]
        extra = ""
        if len(e["seeds"]) > 1:
            vals = [s["test"][key] for s in e["seeds"]]
            extra = f" (5 seeds: {np.mean(vals):.3f} +/- {np.std(vals):.3f})"
        lines.append(f"| {get_spec(mid).display_name} | {m['train'][main]:.3f} | {m['test'][main]:.3f} | "
                     f"{m['test'][key]:.3f}{extra} | {m['train_time'] * 1000:.0f} |")
    if prob == "classification":
        base = next(iter(res["models"].values()))["main"]["test"]["baseline"]
        lines.append(f"\nMajority class baseline on the test set: {base:.3f}.")
    return lines


def curves_note(res) -> list[str]:
    out = []
    for mid, e in res["models"].items():
        c = e["main"].get("curves")
        if c and mid == "boosting":
            tl = np.array(c["test_loss"])
            b = int(tl.argmin())
            out.append(f"\nBoosting test loss: minimum {tl[b]:.3f} at tree {b + 1}, final value {tl[-1]:.3f} at tree {len(tl)} "
                       f"({(tl[-1] / tl[b] - 1) * 100:+.0f} percent).")
        if c and mid == "forest":
            tl = np.array(c["test_loss"])
            out.append(f"\nForest test loss: {tl.min():.3f} at best, final {tl[-1]:.3f} at tree {len(tl)}.")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--no-write", action="store_true")
    args = ap.parse_args(argv)
    only = {s for s in args.only.split(",") if s}
    doc = ["# Verified scenario results", "",
           "These numbers come from `python -m ml_model_dojo.scenarios.verify`, which runs every scenario with the "
           "settings stored in `scenarios/definitions.py`. Timings depend on the machine. Each follow-up row applies the extra "
           "setting that the scenario text asks the student to change.", ""]
    with ProcessPoolExecutor(4, mp_context=mp.get_context("spawn")) as ex:
        for i, sc in enumerate(SCENARIOS, 1):
            if only and sc["id"] not in only:
                continue
            res = run_scenario(ex, sc)
            cfg = res["cfg"]
            doc += [f"## {i}. {sc['title']}", "",
                    f"Data: `{cfg.dataset}`, n = {cfg.n_samples}, noise = {cfg.noise}, minority share = {cfg.minority_share}, "
                    f"test mode = {cfg.test_mode}, seed = {sc['seed']}.", "", f"What to look for: {sc['look_for']}", ""]
            doc += fmt(sc, res) + curves_note(res)
            print(f"\n=== {sc['title']}")
            print("\n".join(fmt(sc, res)))
            print("\n".join(curves_note(res)))
            for fu in sc.get("followups", []):
                r2 = run_scenario(ex, sc, fu.get("data"), fu.get("params"))
                doc += ["", f"Follow-up: {fu['title']}", ""] + fmt(sc, r2)
                print(f"--- follow-up {fu['title']}")
                print("\n".join(fmt(sc, r2)))
            doc.append("")
    if not args.no_write and not only:
        out = Path(__file__).resolve().parents[3] / "docs" / "scenarios.md"
        out.parent.mkdir(exist_ok=True)
        out.write_text("\n".join(doc), encoding="utf-8")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
