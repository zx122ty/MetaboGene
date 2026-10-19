#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 15b — 实体记忆地板线（entity-memory floor）。

目的（prompt/26-10-16/03-SOTA基线补充.md §4.4）：
  回答“0.9277 意味着什么”。一个完全不看化学、不看序列，只用实体 ID 的
  线性模型能拿到多少？这是三基线验收 A1 的下界锚点，也是本文模型
  “学到了真实结构信号”的干净参照系。

特征（稀疏 one-hot）：
  met_id       : one-hot(row_met)
  tar_id       : one-hot(row_tar)
  met+tar      : hstack[one-hot(row_met), one-hot(row_tar)]
  tar_hotness  : log1p(该靶点在训练集中的正样本数)（可选单特征）
  met_hotness  : log1p(该代谢物在训练集中的正样本数)（可选单特征）

模型：sklearn LogisticRegression(C ∈ {0.3, 1.0}, max_iter=300)，
      C 只在验证集上选择（禁止用 test 挑）。

两个划分：row-level（split_features_esm.npz）与 strict pair-disjoint
（split_strict_features_esm.npz，shared strict_split）。

产物：
  results/sota_entity_floor_metrics.json
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import scipy.sparse as sp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import strict_split

cfg = C.load()
RES_DIR = cfg.get_path("paths.results_dir")
os.makedirs(RES_DIR, exist_ok=True)
SEED = int(cfg.get("training.seed"))


def log(m):
    print(f"[15b] {m}", flush=True)


def _onehot(values, universe):
    """values → csr one-hot (n, len(universe))；universe 必须已排序。"""
    cols = np.searchsorted(universe, values)
    if not (universe[cols] == values).all():
        raise ValueError("one-hot 编码出现 universe 之外的实体")
    n = len(values)
    return sp.csr_matrix((np.ones(n, dtype=np.float32), (np.arange(n), cols)),
                         shape=(n, len(universe)))


def _hotness(values, universe, train_values, train_positive_mask):
    """log1p(训练集该实体的正样本数)；训练集未出现的实体记 0。"""
    pos = np.zeros(len(universe), dtype=np.float32)
    tv = train_values[train_positive_mask]
    cols, counts = np.unique(tv, return_counts=True)
    idx = np.searchsorted(universe, cols)
    pos[idx] = counts
    return np.log1p(pos[np.searchsorted(universe, values)]).reshape(-1, 1)


def compute_metrics(y, p, topk_frac=0.01):
    pred = (p > 0.5).astype(int)
    k = max(1, int(topk_frac * len(y)))
    top = np.argsort(-p)[:k]
    out = {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "precision_at_topk": round(float(y[top].mean()), 4),
        "topk_k": int(k),
        "n_test": int(len(y)),
        "n_pos_test": int(y.sum()),
    }
    return out


def fit_select_eval(X, y, tr, va, te, Cs, seed):
    """在 train 上拟合、val 上选 C、test 上评估。返回 (result, selected_C)。"""
    Xtr, Xva, Xte = X[tr], X[va], X[te]
    ytr = y[tr]
    best_c, best_va, val_curve, fits = None, -1.0, {}, {}
    t_all = 0.0
    for c in Cs:
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(C=c, max_iter=300, solver="lbfgs",
                                     random_state=seed)
            clf.fit(Xtr, ytr)
        dt = time.time() - t0
        va_auc = float(roc_auc_score(y[va], clf.predict_proba(Xva)[:, 1]))
        val_curve[str(c)] = round(va_auc, 4)
        fits[str(c)] = clf
        t_all += dt
        log(f"    C={c}: val ROC-AUC {va_auc:.4f} ({dt:.1f}s)")
        if va_auc > best_va:
            best_va, best_c = va_auc, c
    clf = fits[str(best_c)]
    p_te = clf.predict_proba(Xte)[:, 1]
    res = compute_metrics(y[te], p_te)
    res.update({"selected_C": best_c, "val_roc_auc": round(best_va, 4),
                "val_roc_auc_by_C": val_curve,
                "train_seconds": round(t_all, 1),
                "n_features": int(X.shape[1])})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["row", "strict", "both"], default="both")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--Cs", type=float, nargs="+", default=[0.3, 1.0])
    ap.add_argument("--no-hotness", action="store_true")
    args = ap.parse_args()

    feat_name = cfg.get("training.feat_file_map")["esm"]
    feat_dir = cfg.get_path("paths.features_dir")
    d = np.load(os.path.join(feat_dir, feat_name), allow_pickle=True)
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))
    row_met = d["row_met"].astype(np.int64)
    row_tar = d["row_tar"].astype(np.int64)
    n_pos = int(len(d["dir_met"]))
    y = l["y_binary"].astype(np.int64)
    log(f"rows={len(y):,} positives={int(y.sum()):,} n_pos_block={n_pos:,}")

    # 排序后的实体宇宙：列顺序在 row/strict 两个划分间保持一致
    met_univ = np.unique(row_met)
    tar_univ = np.unique(row_tar)
    log(f"unique metabolites={len(met_univ):,} targets={len(tar_univ):,}")

    X_met = _onehot(row_met, met_univ)
    X_tar = _onehot(row_tar, tar_univ)
    X_both = sp.hstack([X_met, X_tar], format="csr")
    feats = {"met_id": X_met, "tar_id": X_tar, "met+tar": X_both}
    if not args.no_hotness:
        feats["tar_hotness"] = None  # 依赖 train，稍后构造
        feats["met_hotness"] = None

    splits = {}
    if args.split in ("row", "both"):
        splits["row"] = {"b_tr": s["b_tr"], "b_va": s["b_va"], "b_te": s["b_te"]}
    if args.split in ("strict", "both"):
        sp_strict = strict_split.load_or_build(feat_dir, feat_name,
                                               row_met=row_met, row_tar=row_tar,
                                               n_pos=n_pos, seed=args.seed)
        splits["strict"] = sp_strict

    out = {
        "description": "entity-memory floor: ID-only logistic regression",
        "seed": args.seed,
        "n_rows": int(len(y)),
        "n_unique_met": int(len(met_univ)),
        "n_unique_tar": int(len(tar_univ)),
        "topk_frac": float(cfg.get("training.topk_frac")),
        "Cs": args.Cs,
        "splits": {},
    }

    for split_name, idx in splits.items():
        tr, va, te = idx["b_tr"], idx["b_va"], idx["b_te"]
        log(f"=== split={split_name}: train {len(tr):,} / val {len(va):,} / test {len(te):,}")
        # 热度特征只用训练集标签统计（防止 test 泄漏）
        local_feats = dict(feats)
        if not args.no_hotness:
            pos_mask = y == 1
            # 在训练行中取正样本
            tr_pos = np.zeros(len(y), dtype=bool)
            tr_pos[tr] = True
            train_pos_mask = pos_mask & tr_pos
            local_feats["tar_hotness"] = _hotness(row_tar, tar_univ, row_tar, train_pos_mask)
            local_feats["met_hotness"] = _hotness(row_met, met_univ, row_met, train_pos_mask)

        split_res = {}
        for name in ["met_id", "tar_id", "met+tar", "tar_hotness", "met_hotness"]:
            if name not in local_feats or local_feats[name] is None:
                continue
            log(f"  -- {name} (dim={local_feats[name].shape[1]})")
            split_res[name] = fit_select_eval(local_feats[name], y, tr, va, te,
                                              args.Cs, args.seed)
            log(f"     test: {split_res[name]}")
        out["splits"][split_name] = split_res

    path = os.path.join(RES_DIR, "sota_entity_floor_metrics.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    log(f"saved {path}")


if __name__ == "__main__":
    main()
