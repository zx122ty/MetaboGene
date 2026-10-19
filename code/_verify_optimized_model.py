#!/usr/bin/env python3
"""Read-only checks for the graph-augmented SOTA artifacts."""
import glob
import json
import os

import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

res = "results"
runs = f"{res}/sota_runs"
preds = f"{res}/sota_preds"
docs = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "writing", "plan-C")
)


def load(pattern):
    out = []
    for path in sorted(glob.glob(os.path.join(runs, pattern))):
        with open(path) as f:
            out.append(json.load(f))
    return out


checks = []

row = load("graph_ecfp_esm_row_s*_b512.json")
checks.append(("row seeds", sorted(r["seed"] for r in row) == [0, 1, 2, 42]))
checks.append((
    "row mean",
    abs(np.mean([r["roc_auc"] for r in row]) - 0.934125) < 1e-6,
))

strict = load("graph_ecfp_esm_strict_s*_b128.json")
checks.append((
    "strict seeds",
    sorted(r["seed"] for r in strict) == [0, 1, 42],
))
checks.append((
    "strict mean",
    abs(np.mean([r["roc_auc"] for r in strict]) - 0.9189) < 1e-4,
))

direction = load("graph_ecfp_esm_direction_row_s*_b128.json")
checks.append((
    "direction seeds",
    sorted(r["seed"] for r in direction) == [0, 1, 2, 42],
))
checks.append((
    "direction mean",
    abs(np.mean([r["accuracy"] for r in direction]) - 0.836925) < 1e-6,
))

for name, expected, metric in [
    ("graph_ecfp_esm_ensemble4_row_s42.npz", 0.9449, "roc"),
    ("graph_ecfp_esm_ensemble3_strict_s42.npz", 0.9307, "roc"),
    ("graph_ecfp_esm_direction_ensemble4_row_s42.npz", 0.8492, "acc"),
]:
    z = np.load(os.path.join(preds, name))
    score = (
        roc_auc_score(z["y"], z["p"])
        if metric == "roc"
        else accuracy_score(z["y"], z["p"].argmax(axis=1))
    )
    checks.append((name, abs(score - expected) < 1e-4))

with open(os.path.join(res, "sota_runs", "moltrans_original_row_s42_own.json")) as f:
    mt = json.load(f)
checks.append(("MolTrans official > floor", mt["roc_auc"] > 0.8387))

ext_path = os.path.join(res, "metabogene_graph_e4_external_metrics.json")
checks.append(("external E4 metrics", os.path.exists(ext_path)))
if os.path.exists(ext_path):
    with open(ext_path) as f:
        ext = json.load(f)
    checks.append((
        "external E4 ChEMBL",
        abs(ext["sources"]["chembl:esm"]["roc_auc"] - 0.8837) < 1e-4,
    ))
    checks.append((
        "external E4 HMDB",
        abs(ext["sources"]["hmdb_assoc:esm"]["roc_auc"] - 0.8658) < 1e-4,
    ))

unc_path = os.path.join(res, "metabogene_graph_uncertainty_metrics.json")
checks.append(("graph uncertainty metrics", os.path.exists(unc_path)))
if os.path.exists(unc_path):
    with open(unc_path) as f:
        unc = json.load(f)
    checks.append((
        "graph E4 uncertainty",
        abs(unc["metrics"]["row_e4"]["entropy_error_auc"] - 0.8111) < 1e-4,
    ))

fair_path = os.path.join(res, "sota_equal_ensemble_comparison.json")
checks.append(("equal-aggregation comparison", os.path.exists(fair_path)))
if os.path.exists(fair_path):
    with open(fair_path) as f:
        fair = json.load(f)
    graphdta_comparison = fair["comparisons"]["GraphDTA+ESM"]
    checks.append((
        "equal-aggregation GraphDTA+ESM",
        abs(graphdta_comparison["delta_roc_auc"] - 0.000575) < 1e-5
        and graphdta_comparison["p_value"] > 0.05,
    ))

fig3_files = [
    os.path.join(docs, "figures", f"fig3_direction_confusion.{ext}")
    for ext in ("png", "pdf", "svg")
]
if os.path.isdir(docs):
    checks.append(("Fig. 3 deployed-model files", all(map(os.path.exists, fig3_files))))

failed = [name for name, ok in checks if not ok]
for name, ok in checks:
    print(f"{'✓' if ok else '✗'} {name}")
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
raise SystemExit(1 if failed else 0)
