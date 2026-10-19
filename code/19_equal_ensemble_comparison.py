#!/usr/bin/env python3
"""Compare MetaboGene-Graph with SOTA controls under equal seed aggregation.

This is a read-only evaluation over saved test predictions. It does not train
models or tune ensemble weights. The four-seed ensembles are equal arithmetic
means of predicted probabilities, matching the construction of Graph-E4.
"""

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import norm, ttest_rel
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    roc_auc_score,
)


SEEDS = (0, 1, 2, 42)
ROOT = Path(__file__).resolve().parent.parent
PRED_DIR = ROOT / "results" / "sota_preds"
OUT_JSON = ROOT / "results" / "sota_equal_ensemble_comparison.json"
OUT_CSV = (
    ROOT.parent
    / "writing"
    / "plan-C"
    / "supplementary_tables"
    / "S18_equal_aggregation_comparison.csv"
)

MODELS = {
    "DeepDTA+ESM": {
        "prefix": "deepdta_row",
        "suffix": "own_esm",
        "display": "DeepDTA+ESM-E4",
    },
    "GraphDTA+ESM": {
        "prefix": "graphdta_row",
        "suffix": "own_esm",
        "display": "GraphDTA+ESM-E4",
    },
    "MetaboGene-Graph": {
        "prefix": "graph_ecfp_esm_row",
        "suffix": "b512",
        "display": "MetaboGene-Graph-E4",
    },
}


def _midrank(values):
    order = np.argsort(values)
    sorted_values = values[order]
    ranks = np.zeros(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i
        while j < len(values) and sorted_values[j] == sorted_values[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(len(values), dtype=np.float64)
    out[order] = ranks
    return out


def delong_paired(y, p1, p2):
    """Return paired DeLong AUCs, their difference, CI, z and p value."""
    y = np.asarray(y).astype(int)
    pos = y == 1
    neg = y == 0
    m = int(pos.sum())
    n = int(neg.sum())
    if m == 0 or n == 0:
        raise ValueError("Both classes are required")

    tz1, tz2 = _midrank(p1), _midrank(p2)
    tx1, tx2 = _midrank(p1[pos]), _midrank(p2[pos])
    ty1, ty2 = _midrank(p1[neg]), _midrank(p2[neg])
    auc1 = (tz1[pos].sum() - m * (m + 1) / 2) / (m * n)
    auc2 = (tz2[pos].sum() - m * (m + 1) / 2) / (m * n)

    d1p, d2p = tz1[pos] - tx1, tz2[pos] - tx2
    d1n, d2n = tz1[neg] - ty1, tz2[neg] - ty2
    s10_1 = float(np.var(d1p, ddof=1)) / (n**2)
    s01_1 = float(np.var(d1n, ddof=1)) / (m**2)
    s10_2 = float(np.var(d2p, ddof=1)) / (n**2)
    s01_2 = float(np.var(d2n, ddof=1)) / (m**2)
    cov10 = float(np.cov(d1p, d2p, ddof=1)[0, 1]) / (n**2)
    cov01 = float(np.cov(d1n, d2n, ddof=1)[0, 1]) / (m**2)
    variance = (
        s10_1 / m
        + s01_1 / n
        + s10_2 / m
        + s01_2 / n
        - 2 * (cov10 / m + cov01 / n)
    )
    se = float(np.sqrt(max(variance, 1e-18)))
    delta = float(auc1 - auc2)
    z = delta / se if se > 0 else 0.0
    p_value = float(2 * (1 - norm.cdf(abs(z))))
    return {
        "auc_model": float(auc1),
        "auc_comparator": float(auc2),
        "delta_roc_auc": delta,
        "ci95_low": float(delta - 1.96 * se),
        "ci95_high": float(delta + 1.96 * se),
        "z": float(z),
        "p_value": p_value,
    }


def load_predictions(prefix, suffix):
    arrays = []
    for seed in SEEDS:
        path = PRED_DIR / f"{prefix}_s{seed}_{suffix}.npz"
        with np.load(path, allow_pickle=True) as z:
            arrays.append(
                {
                    "seed": seed,
                    "y": z["y"].copy(),
                    "p": z["p"].astype(np.float64).copy(),
                    "rows": z["rows"].copy(),
                }
            )

    reference = arrays[0]
    for item in arrays[1:]:
        if not np.array_equal(item["y"], reference["y"]):
            raise ValueError(f"Label order differs for {prefix}")
        if not np.array_equal(item["rows"], reference["rows"]):
            raise ValueError(f"Test-row order differs for {prefix}")
    return reference["y"], reference["rows"], arrays


def metrics(y, p):
    prediction = (p >= 0.5).astype(int)
    top_k = max(1, int(0.01 * len(y)))
    top = np.argsort(-p)[:top_k]
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, prediction)),
        "precision_at_top_1pct": float(np.mean(y[top])),
        "top_k": int(top_k),
    }


def main():
    results = {
        "comparison": "equal_seed_probability_aggregation",
        "seeds": list(SEEDS),
        "aggregation": "equal arithmetic mean of predicted probabilities",
        "selection": "no ensemble weight or threshold was tuned on the test set",
        "training_schedule_note": (
            "Saved predictions were reused without retraining. The Graph models "
            "use patience 10; the DeepDTA/GraphDTA+ESM controls were produced "
            "with patience 6. The equal-aggregation comparison removes the "
            "seed-count mismatch, but it does not erase that schedule difference."
        ),
        "models": {},
    }

    global_y = None
    global_rows = None
    ensemble_predictions = {}
    seed_aucs = {}

    for name, spec in MODELS.items():
        y, rows, arrays = load_predictions(spec["prefix"], spec["suffix"])
        if global_y is None:
            global_y, global_rows = y, rows
        elif not np.array_equal(y, global_y) or not np.array_equal(rows, global_rows):
            raise ValueError(f"Test rows differ between model families: {name}")

        probs = np.stack([item["p"] for item in arrays], axis=0)
        ensemble_p = probs.mean(axis=0)
        seed_metrics = [metrics(y, item["p"]) for item in arrays]
        seed_aucs[name] = [item["roc_auc"] for item in seed_metrics]
        ensemble_predictions[name] = ensemble_p
        results["models"][name] = {
            "display_name": spec["display"],
            "single_model": {
                "roc_auc_mean": float(np.mean(seed_aucs[name])),
                "roc_auc_sd": float(np.std(seed_aucs[name], ddof=1)),
                "per_seed": {
                    str(seed): value
                    for seed, value in zip(SEEDS, seed_metrics, strict=True)
                },
            },
            "ensemble": metrics(y, ensemble_p),
        }

    graph_name = "MetaboGene-Graph"
    graph_p = ensemble_predictions[graph_name]
    results["comparisons"] = {}
    for name in ("DeepDTA+ESM", "GraphDTA+ESM"):
        results["comparisons"][name] = delong_paired(
            global_y, graph_p, ensemble_predictions[name]
        )

    results["seed_matched_single_model"] = {}
    for name in ("DeepDTA+ESM", "GraphDTA+ESM"):
        graph_seed = np.load(
            PRED_DIR / "graph_ecfp_esm_row_s42_b512.npz", allow_pickle=True
        )
        baseline_seed = np.load(
            PRED_DIR / f"{MODELS[name]['prefix']}_s42_own_esm.npz",
            allow_pickle=True,
        )
        results["seed_matched_single_model"][name] = delong_paired(
            graph_seed["y"],
            graph_seed["p"].astype(np.float64),
            baseline_seed["p"].astype(np.float64),
        )

    graph_seed_aucs = np.asarray(seed_aucs[graph_name], dtype=np.float64)
    for name in ("DeepDTA+ESM", "GraphDTA+ESM"):
        baseline_aucs = np.asarray(seed_aucs[name], dtype=np.float64)
        test = ttest_rel(graph_seed_aucs, baseline_aucs)
        results["single_model_four_seed_paired_auc_test"] = (
            results.get("single_model_four_seed_paired_auc_test", {})
        )
        results["single_model_four_seed_paired_auc_test"][name] = {
            "mean_delta_roc_auc": float(np.mean(graph_seed_aucs - baseline_aucs)),
            "t_statistic": float(test.statistic),
            "p_value": float(test.pvalue),
            "all_graph_seeds_higher": bool(np.all(graph_seed_aucs > baseline_aucs)),
        }

    results["n_test"] = int(len(global_y))
    results["n_positive"] = int(global_y.sum())
    OUT_JSON.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "model",
        "seeds",
        "aggregation",
        "n_test",
        "roc_auc",
        "pr_auc",
        "f1",
        "precision_at_top_1pct",
        "single_model_roc_mean",
        "single_model_roc_sd",
        "delta_vs_graph_e4",
        "ci95_low",
        "ci95_high",
        "p_value",
    ]
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, spec in MODELS.items():
            row = results["models"][name]
            comparison = results["comparisons"].get(name)
            writer.writerow(
                {
                    "model": spec["display"],
                    "seeds": "0/1/2/42",
                    "aggregation": "equal mean probability",
                    "n_test": results["n_test"],
                    "roc_auc": f"{row['ensemble']['roc_auc']:.6f}",
                    "pr_auc": f"{row['ensemble']['pr_auc']:.6f}",
                    "f1": f"{row['ensemble']['f1']:.6f}",
                    "precision_at_top_1pct": (
                        f"{row['ensemble']['precision_at_top_1pct']:.6f}"
                    ),
                    "single_model_roc_mean": (
                        f"{row['single_model']['roc_auc_mean']:.6f}"
                    ),
                    "single_model_roc_sd": (
                        f"{row['single_model']['roc_auc_sd']:.6f}"
                    ),
                    "delta_vs_graph_e4": (
                        f"{comparison['delta_roc_auc']:+.6f}" if comparison else ""
                    ),
                    "ci95_low": (
                        f"{comparison['ci95_low']:+.6f}" if comparison else ""
                    ),
                    "ci95_high": (
                        f"{comparison['ci95_high']:+.6f}" if comparison else ""
                    ),
                    "p_value": (
                        f"{comparison['p_value']:.6g}" if comparison else ""
                    ),
                }
            )

    print(json.dumps(results, indent=2))
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
