#!/usr/bin/env python3
"""Build seed ensembles and their headline metrics from saved predictions."""
import json
import os

import numpy as np
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, average_precision_score,
                             confusion_matrix, f1_score, roc_auc_score)

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")
PRED = os.path.join(RES, "sota_preds")
SEEDS = (0, 1, 2, 42)


def _load(prefix, seeds):
    zs = [
        np.load(os.path.join(PRED, f"{prefix}_s{s}_b128.npz"))
        if "strict" in prefix or "direction" in prefix
        else np.load(os.path.join(PRED, f"{prefix}_s{s}_b512.npz"))
        for s in seeds
    ]
    y = zs[0]["y"]
    rows = zs[0]["rows"]
    if any(not np.array_equal(y, z["y"]) for z in zs):
        raise RuntimeError("Ensemble members do not share labels")
    if any(not np.array_equal(rows, z["rows"]) for z in zs):
        raise RuntimeError("Ensemble members do not share row order")
    return y, rows, zs


def binary_ensemble(prefix, seeds, out_name):
    y, rows, zs = _load(prefix, seeds)
    p = np.mean([z["p"].astype(np.float64) for z in zs], axis=0)
    pred = (p > 0.5).astype(int)
    k = max(1, int(0.01 * len(y)))
    top = np.argsort(-p)[:k]
    metrics = {
        "model": out_name,
        "seeds": list(seeds),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "precision_at_topk": round(float(y[top].mean()), 4),
        "topk_k": int(k),
        "n_test": int(len(y)),
        "n_pos_test": int(y.sum()),
    }
    np.savez_compressed(os.path.join(PRED, f"{out_name}.npz"),
                        y=y, p=p, rows=rows)
    with open(os.path.join(RES, f"{out_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def direction_ensemble():
    y, rows, zs = _load("graph_ecfp_esm_direction_row", SEEDS)
    p = np.mean([
        softmax(z["p"].astype(np.float64), axis=1) for z in zs
    ], axis=0)
    pred = p.argmax(axis=1)
    metrics = {
        "model": "graph_ecfp_esm_direction_ensemble4_row_s42",
        "seeds": list(SEEDS),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "macro_f1": round(float(f1_score(y, pred, average="macro")), 4),
        "cm": confusion_matrix(y, pred).tolist(),
        "n_test": int(len(y)),
    }
    name = "graph_ecfp_esm_direction_ensemble4_row_s42"
    np.savez_compressed(os.path.join(PRED, f"{name}.npz"),
                        y=y, p=p, rows=rows)
    with open(os.path.join(RES, f"{name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return metrics


def main():
    print(binary_ensemble(
        "graph_ecfp_esm_row", SEEDS,
        "graph_ecfp_esm_ensemble4_row_s42",
    ))
    print(binary_ensemble(
        "graph_ecfp_esm_strict", (0, 1, 42),
        "graph_ecfp_esm_ensemble3_strict_s42",
    ))
    print(direction_ensemble())


if __name__ == "__main__":
    main()
