#!/usr/bin/env python3
"""Post-hoc uncertainty evaluation for MetaboGene-Graph predictions."""
import json
import os

import numpy as np
from sklearn.metrics import roc_auc_score

RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results")
PRED = os.path.join(RES, "sota_preds")


def entropy(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1 - 1e-12)
    return -(p * np.log(p) + (1 - p) * np.log(1 - p))


def ece(y, p, bins=15):
    conf = np.maximum(p, 1 - p)
    pred = (p >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            value += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(value)


def bootstrap_auc(errors, uncertainty, seed=42, n_boot=500):
    rng = np.random.RandomState(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(errors), len(errors))
        if len(np.unique(errors[idx])) < 2:
            continue
        vals.append(roc_auc_score(errors[idx], uncertainty[idx]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def evaluate(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=np.float64)
    pred = (p >= 0.5).astype(int)
    err = (pred != y).astype(int)
    u = entropy(p)
    order = np.argsort(u)
    k = max(1, int(0.05 * len(y)))
    confident = order[:k]
    uncertain = order[-k:]
    bins = np.array_split(order, 20)
    return {
        "n": int(len(y)),
        "accuracy": round(float((pred == y).mean()), 4),
        "error_rate": round(float(err.mean()), 4),
        "entropy_mean": round(float(u.mean()), 4),
        "entropy_error_auc": round(float(roc_auc_score(err, u)), 4),
        "entropy_error_auc_95ci": [
            round(v, 4) for v in bootstrap_auc(err, u)
        ],
        "mcp_error_auc": round(
            float(roc_auc_score(err, np.maximum(p, 1 - p))), 4
        ),
        "brier": round(float(np.mean((p - y) ** 2)), 4),
        "ece_15bin": round(ece(y, p), 4),
        "most_confident_5pct_accuracy": round(
            float((pred[confident] == y[confident]).mean()), 4
        ),
        "least_confident_5pct_accuracy": round(
            float((pred[uncertain] == y[uncertain]).mean()), 4
        ),
        "bin_accuracy": [
            round(float((pred[b] == y[b]).mean()), 4) for b in bins
        ],
    }


def main():
    out = {"model": "MetaboGene-Graph", "metrics": {}}
    row_single = np.load(os.path.join(
        PRED, "graph_ecfp_esm_row_s42_b512.npz"
    ))
    row_e4 = np.load(os.path.join(
        PRED, "graph_ecfp_esm_ensemble4_row_s42.npz"
    ))
    strict_e3 = np.load(os.path.join(
        PRED, "graph_ecfp_esm_ensemble3_strict_s42.npz"
    ))
    out["metrics"]["row_single_s42"] = evaluate(
        row_single["y"], row_single["p"]
    )
    out["metrics"]["row_e4"] = evaluate(row_e4["y"], row_e4["p"])
    out["metrics"]["strict_e3"] = evaluate(
        strict_e3["y"], strict_e3["p"]
    )

    ext = np.load(os.path.join(
        RES, "metabogene_graph_e4_external_metrics_probs.npz"
    ))
    labels = {
        "chembl:esm": "chembl",
        "hmdb_assoc:esm": "hmdb_assoc",
        "drugbank_in_hmdb:esm": "drugbank_in_hmdb",
        "drugbank:esm": "drugbank",
        "pdb:esm": "pdb",
    }
    for key, name in labels.items():
        out["metrics"][f"external_{name}"] = evaluate(
            ext[key + ":y"], ext[key + ":p"]
        )

    path = os.path.join(RES, "metabogene_graph_uncertainty_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
