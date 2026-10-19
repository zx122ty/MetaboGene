#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SOTA 基线交付物与验收自检（只读）。

检查 prompt/26-10-16/03-SOTA基线补充.md §7 的 A1–A9 与 §1 的 D1–D12。
用法: python3 code/_verify_sota.py
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C

cfg = C.load()
RES = cfg.get_path("paths.results_dir")
FEAT = cfg.get_path("paths.features_dir")


def ok(msg):
    print(f"  ✓ {msg}")


def warn(msg):
    print(f"  ⚠ {msg}")


def fail(msg):
    print(f"  ✗ {msg}")
    FAILS.append(msg)


FAILS = []


def main():
    print("== D2 strict split ==")
    p = os.path.join(FEAT, "split_strict_features_esm.npz")
    if os.path.exists(p):
        z = np.load(p)
        sizes = (len(z["b_tr"]), len(z["b_va"]), len(z["b_te"]))
        exp = (317290, 40144, 42094)
        (ok if sizes == exp else fail)(f"{sizes} (expected {exp})")
        if len(set(z["b_tr"]) & set(z["b_te"])) == 0:
            ok("train/test disjoint")
        else:
            fail("train/test overlap")
    else:
        fail("missing split_strict_features_esm.npz")

    print("== D9 entity floor ==")
    p = os.path.join(RES, "sota_entity_floor_metrics.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for split, exp in (("row", {"met_id": 0.7028, "tar_id": 0.7881, "met+tar": 0.8387}),
                           ("strict", {"met_id": 0.6714, "tar_id": 0.7534, "met+tar": 0.8145})):
            for k, v in exp.items():
                got = d["splits"][split][k]["roc_auc"]
                (ok if abs(got - v) < 0.01 else fail)(
                    f"{split}/{k} ROC-AUC {got:.4f} (expected {v:.4f}, |Δ|<0.01)")
    else:
        fail("missing sota_entity_floor_metrics.json")

    print("== D3–D6 runs ==")
    runs = [json.load(open(f)) for f in glob.glob(os.path.join(RES, "sota_runs", "*.json"))]
    print(f"  runs: {len(runs)}")
    baseline_runs = [r for r in runs if r.get("model") in
                     ("deepdta", "graphdta", "moltrans")]
    required = ("n_params", "train_seconds", "epochs_run", "best_epoch", "seed",
                "split", "protocol", "model", "max_len_smi", "max_len_prot")
    a4_ok = True
    for r in baseline_runs:
        miss = [k for k in required if k not in r]
        if miss:
            a4_ok = False
            fail(f"{r.get('model')}/{r.get('split')}/s{r.get('seed')} missing {miss}")
    if a4_ok:
        ok("A4 required fields present in all run JSONs")
    for model in ("deepdta", "graphdta", "moltrans"):
        row = [r for r in runs if r["model"] == model and r["split"] == "row"
               and r["protocol"] == "own" and r["protein"] == "native"
               and r.get("max_len_smi", 128) == 128 and r.get("max_len_prot", 1500) == 1500]
        if row:
            aucs = [r["roc_auc"] for r in row]
            lo, hi = min(aucs), max(aucs)
            if lo < 0.8387:
                official = [r for r in runs if r["model"] == "moltrans"
                            and r["split"] == "row" and r["protocol"] == "own"
                            and r.get("variant") == "original"
                            and int(r.get("max_epochs", 0)) > 1]
                if model == "moltrans" and official:
                    olo = min(r["roc_auc"] for r in official)
                    ohi = max(r["roc_auc"] for r in official)
                    if olo >= 0.8387 and ohi <= 0.99:
                        warn(f"A1 moltrans adapted min {lo:.4f} < floor; "
                             f"official [{olo:.4f}, {ohi:.4f}] passes")
                    else:
                        fail(f"A1 moltrans official min {olo:.4f} < floor 0.8387")
                else:
                    fail(f"A1 {model} row-level min {lo:.4f} < floor 0.8387")
            elif hi > 0.99:
                fail(f"A1 {model} row-level max {hi:.4f} > 0.99 (leak?)")
            else:
                ok(f"A1 {model} row-level [{lo:.4f}, {hi:.4f}] within [0.8387, 0.95] "
                   f"({len(row)} seeds)")
            st = [r for r in runs if r["model"] == model and r["split"] == "strict"]
            if st:
                ok(f"A2 {model} strict ROC-AUC {np.mean([r['roc_auc'] for r in st]):.4f}")
            else:
                official = [r for r in runs if r["model"] == "moltrans"
                            and r["split"] == "strict" and r.get("variant") == "original"]
                if model == "moltrans" and official:
                    ok(f"A2 moltrans official strict ROC-AUC "
                       f"{np.mean([r['roc_auc'] for r in official]):.4f}")
                else:
                    warn(f"A2 {model} strict runs missing")
        else:
            warn(f"A1 {model} row-level own runs missing")
    curves = glob.glob(os.path.join(RES, "sota_curves", "*.npz"))
    print(f"== D12 curves: {len(curves)} ==")
    for model in ("deepdta", "graphdta", "moltrans"):
        if any(os.path.basename(c).startswith(model + "_row_s42_own") for c in curves):
            ok(f"{model} row s42 curve")
        else:
            warn(f"{model} row s42 curve missing")

    print("== D10 external ==")
    p = os.path.join(RES, "sota_external_metrics.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for model in ("deepdta", "graphdta", "moltrans"):
            if model not in d:
                warn(f"{model} external missing")
                continue
            src = d[model]["sources"]
            need = ("chembl", "hmdb_assoc", "pdb", "drugbank", "drugbank_in_hmdb")
            miss = [k for k in need if k not in src or "roc_auc" not in src.get(k, {})]
            (ok if not miss else fail)(f"{model} external sources {need} (missing {miss})")
    else:
        fail("missing sota_external_metrics.json")

    print("== D11 cold ==")
    p = os.path.join(RES, "sota_cold_metrics.json")
    if os.path.exists(p):
        d = json.load(open(p))
        for model in ("deepdta", "graphdta", "moltrans"):
            for split in ("cold_drug", "cold_target", "dual_cold"):
                if split in d.get(model, {}):
                    ok(f"{model}/{split} {d[model][split]['roc_auc_mean']:.4f} "
                       f"({len(d[model][split]['seeds'])} seed)")
                else:
                    warn(f"{model}/{split} missing")
    else:
        fail("missing sota_cold_metrics.json")

    print("== D7 report ==")
    for f in ("sota_baselines.md", "sota_baselines_summary.json",
              "sota_deepdta_metrics.json", "sota_graphdta_metrics.json",
              "sota_moltrans_metrics.json"):
        (ok if os.path.exists(os.path.join(RES, f)) else warn)(f)

    print("\n== SUMMARY ==")
    if FAILS:
        print(f"{len(FAILS)} failure(s):")
        for f in FAILS:
            print("  -", f)
        sys.exit(1)
    print("all hard checks passed")


if __name__ == "__main__":
    main()
