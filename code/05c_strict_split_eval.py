#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5c — Strict pair-disjoint split evaluation (leakage-controlled re-evaluation)

Motivation: the default row-level 80/10/10 split (split.npz) allows the same
metabolite-target pair to appear in multiple splits (duplicate literature rows).
To control for this, we build a strict split at the PAIR level:
  * positive rows are grouped by unique (met, tar) pairs; each pair is assigned
    to exactly one of train/val/test (80/10/10, seed=42);
  * negative samples are unique pairs by construction and are assigned randomly
    in the same 80/10/10 ratio (1:1 pos:neg per split).
The model architecture/hyperparameters are identical to Step 5
(DualTaskMLP, ESM-2 protein features with Linear projection), so results are
directly comparable to step5_esm_proj*_metrics.json.

Deliverables:
  results/step5c_strict_pair_metrics.json            (--fusion concat, 默认, 行为不变)
  results/step5c_strict_pair_<fusion>_metrics.json   (--fusion <kind>, 26-10-14 扩展)
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

import config as C
import strict_split

cfg = C.load()
FEAT_DIR = cfg.get_path("paths.features_dir")
RESULT_DIR = cfg.get_path("results.results_dir".replace("results.", "results")) if False else cfg.get_path("paths.results_dir")
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = cfg.get("training.seed")
BS = cfg.get("training.batch_size")
EPOCHS = cfg.get("training.epochs")
LR = cfg.get("training.lr")
PATIENCE = cfg.get("training.patience")
HIDDEN = cfg.get("training.hidden_dim")
DROPOUT = cfg.get("training.dropout")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def _load_train05():
    """导入 05_train_evaluate.DualTaskMLP 作为唯一模型实现（含 fusion 支持），
    保证 strict 复评与 Step5 主实验架构严格同源（原 05c 本地类已由它替代）。"""
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "05_train_evaluate.py")
    spec = importlib.util.spec_from_file_location("train_eval_mod_05c", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


mod05 = _load_train05()


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed()
    ap = argparse.ArgumentParser()
    ap.add_argument("--fusion",
                    choices=["concat", "bilinear", "gated", "xattn",
                             "bilinear_raw", "gated_raw", "xattn_raw"],
                    default="concat",
                    help="融合方式（默认 concat=原行为；模型实现来自 05_train_evaluate.DualTaskMLP）")
    args = ap.parse_args()
    print(f"[05c] fusion={args.fusion}")
    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(FEAT_DIR, feat_name))
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    X = torch.tensor(d["X_binary"]).float().to(DEVICE)
    yb = torch.tensor(l["y_binary"]).to(DEVICE)
    yd = torch.tensor(l["y_direction"]).to(DEVICE)

    # 行实体表（缺陷#4）：04 剔除缺失模态并下采样后，X_binary 行序 = [被保留的正样本 |
    # 被保留的负样本]，不再是「全部 positives 在前」。优先用 04 落盘的 row_met/row_tar。
    if "row_met" in d:
        row_met = d["row_met"].astype("int64")
        row_tar = d["row_tar"].astype("int64")
        n_pos = int(len(d["dir_met"]))
    else:  # 兼容旧特征文件（未含实体表）
        pos = pd.read_parquet(cfg.get_path("paths.positives"))[["met", "tar"]]
        neg = pd.read_parquet(cfg.get_path("paths.negatives"))[["met", "tar"]]
        row_met = np.concatenate([pos["met"].astype("int64").values,
                                  neg["met"].astype("int64").values])
        row_tar = np.concatenate([pos["tar"].astype("int64").values,
                                  neg["tar"].astype("int64").values])
        n_pos = len(pos)
    # 只取正样本行建 pair 分组（负样本对按构造唯一，另行随机分配）
    pairs = list(zip(row_met[:n_pos], row_tar[:n_pos]))
    pair_of_row = {i: pairs[i] for i in range(n_pos)}
    print(f"[05c] 行实体表: 正样本 {n_pos:,} / 负样本 {len(yb)-n_pos:,}")

    # ---- strict pair-level split over positive rows ----
    # 共享实现（26-10-16/03 §3.3）：必须复用调用方 rng，保持后续 batch shuffle
    # 的 RNG 流位置与旧内联实现逐位一致（A5）。
    rng = np.random.RandomState(SEED)
    sp = strict_split.build_strict_split(row_met, row_tar, n_pos, seed=SEED, rng=rng)
    tr, va, te = sp["b_tr"], sp["b_va"], sp["b_te"]
    pos_tr = tr[tr < n_pos]
    pos_va = va[va < n_pos]
    pos_te = te[te < n_pos]
    print(f"[05c] strict pair split: train {len(tr):,} (pos {len(pos_tr):,}) "
          f"val {len(va):,} (pos {len(pos_va):,}) test {len(te):,} (pos {len(pos_te):,})")
    # verify disjointness
    assert not (set(pair_of_row[i] for i in pos_te) & set(pair_of_row[i] for i in pos_tr))
    assert not (set(pair_of_row[i] for i in pos_va) & set(pair_of_row[i] for i in pos_tr))

    fp_dim = 2048  # ECFP4 block of features_esm.npz
    protein_dim = X.shape[1] - fp_dim

    # 统一构造器：与 05 主实验同一 DualTaskMLP（proj_esm=1024 与基线一致）
    def build(in_dim):
        return mod05.DualTaskMLP(in_dim, fp_dim=fp_dim, protein_dim=protein_dim,
                                 proj_esm=1024, fusion=args.fusion).to(DEVICE)

    results = {"split": "pair-disjoint (strict)", "seed": SEED, "feat": feat_name,
               "fusion": args.fusion}

    # ================= binary task =================
    model = build(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    best_va, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        order = tr.copy(); rng.shuffle(order)
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            opt.zero_grad()
            out, _ = model(X[b])
            loss = crit(out, yb[b].float())
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ova, _ = model(X[va])
            va_acc = ((ova.sigmoid() > 0.5).float() == yb[va]).float().mean().item()
        if va_acc > best_va:
            best_va, bad = va_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        outb, _ = model(X[te])
    prob = outb.sigmoid().cpu().numpy()
    yt = yb[te].cpu().numpy()
    pred = (prob > 0.5).astype(int)
    # Top-K 筛选指标（缺陷#5：原命名 topk_recall 实为 Precision@K）
    k = max(1, int(cfg.get("training.topk_frac") * len(yt)))
    n_pos_te = int(yt.sum())
    topk_idx = np.argsort(-prob)[:k]
    precision_at_topk = float(yt[topk_idx].mean())
    recall_at_topk = (float(yt[topk_idx].sum()) / n_pos_te) if n_pos_te else 0.0
    base_rate = (n_pos_te / len(yt)) if len(yt) else 0.0
    enrichment_factor = (precision_at_topk / base_rate) if base_rate else 0.0
    results["binary"] = {
        "roc_auc": round(float(roc_auc_score(yt, prob)), 4),
        "pr_auc": round(float(average_precision_score(yt, prob)), 4),
        "f1": round(float(f1_score(yt, pred)), 4),
        "precision": round(float(precision_score(yt, pred)), 4),
        "recall": round(float(recall_score(yt, pred)), 4),
        "precision_at_topk": round(precision_at_topk, 4),
        "recall_at_topk": round(recall_at_topk, 4),
        "enrichment_factor": round(enrichment_factor, 2),
        "topk_k": k,
        "accuracy": round(float(accuracy_score(yt, pred)), 4),
    }
    print(f"[05c] STRICT binary: {results['binary']}")

    # ================= direction task (positive rows only) =================
    d_tr = np.array(sorted(pos_tr), dtype=np.int64)
    d_va = np.array(sorted(pos_va), dtype=np.int64)
    d_te = np.array(sorted(pos_te), dtype=np.int64)
    Xd = torch.tensor(d["X_direction"]).float()   # CPU：与 05 同款显存策略，按批搬运
    model_d = build(Xd.shape[1])
    opt2 = torch.optim.Adam(model_d.parameters(), lr=LR)
    crit2 = nn.CrossEntropyLoss()
    best_da, best_dstate, bad2 = -1, None, 0
    for ep in range(min(EPOCHS, cfg.get("training.direction_max_epochs"))):
        model_d.train()
        order = d_tr.copy(); rng.shuffle(order)
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            opt2.zero_grad()
            _, dout = model_d(Xd[b].to(DEVICE))
            loss = crit2(dout, yd[b])
            loss.backward(); opt2.step()
        model_d.eval()
        with torch.no_grad():
            _, dva_ = model_d(Xd[d_va].to(DEVICE))
            da = (dva_.argmax(1) == yd[d_va]).float().mean().item()
        if da > best_da:
            best_da, bad2 = da, 0
            best_dstate = {k: v.clone() for k, v in model_d.state_dict().items()}
        else:
            bad2 += 1
            if bad2 >= PATIENCE:
                break
    model_d.load_state_dict(best_dstate)
    model_d.eval()
    with torch.no_grad():
        _, dlog = model_d(Xd[d_te].to(DEVICE))
    ydt = yd[d_te].cpu().numpy()
    dpred = dlog.argmax(1).cpu().numpy()
    cm = confusion_matrix(ydt, dpred)
    results["direction"] = {
        "accuracy": round(float(accuracy_score(ydt, dpred)), 4),
        "macro_f1": round(float(f1_score(ydt, dpred, average="macro")), 4),
        "confusion_matrix_label_order": cfg.direction_label_names(),
        "cm": cm.tolist(),
    }
    print(f"[05c] STRICT direction: {results['direction']}")

    suffix = "" if args.fusion == "concat" else f"_{args.fusion}"  # concat 保持原文件名不覆盖
    out = os.path.join(RESULT_DIR, f"step5c_strict_pair{suffix}_metrics.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[05c] saved {out}")


if __name__ == "__main__":
    main()
