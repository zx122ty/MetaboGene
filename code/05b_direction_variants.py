#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5b — 方向分类优化实验

目标: 把方向三分类 accuracy 从 0.663 → ≥0.70。

三个变体：
  V1: 3类 更深MLP + BatchNorm（基线复现，验证是否只是模型容量不足）
  V2: activation vs inhibition 二分类（剔除 neutral 噪声，测试「方向判向」本身的可分性上限）
  V3: 含 neutral 的三分类，但只对物理结合子集(pure) 方向 + 更长训练 —— 测标签纯净度影响

数据: Step4 的 X_direction / y_direction + d_tr/d_va/d_te。
支持蛋白表征切换：
  --feat default → features.npz（dde 421 维，即 Baseline）
  --feat esm     → features_esm.npz（ESM-3B 2560 维）
  --variant <名> 让输出文件名带后缀，避免覆盖。
交付: results/step5b_direction_variants[_<variant>].json
参数统一来自 config.yaml (paths.* / direction_variants.*)。
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import config as C

cfg = C.load()

FEAT = cfg.get_path("paths.features_dir")
RES = cfg.get_path("paths.results_dir")
os.makedirs(RES, exist_ok=True)

BS = cfg.get("direction_variants.batch_size")
LR = cfg.get("direction_variants.lr")
EPOCHS = cfg.get("direction_variants.epochs")
PATIENCE = cfg.get("direction_variants.patience")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=None):
    import random
    if seed is None:
        seed = cfg.get("direction_variants.seed")
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


class DirMLP(nn.Module):
    def __init__(self, in_dim, nout=3, hidden=None):
        super().__init__()
        if hidden is None:
            hidden = cfg.get("direction_variants.hidden_dims")
        dropout = cfg.get("direction_variants.dropout")
        h1, h2 = hidden[0], hidden[1]
        self.net = nn.Sequential(
            nn.Linear(in_dim, h1), nn.BatchNorm1d(h1), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h1, h2), nn.BatchNorm1d(h2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(h2, nout),
        )

    def forward(self, x):
        return self.net(x)


def train_val(model, Xtr, ytr, Xva, yva):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    best_acc, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        perm = np.random.permutation(len(Xtr))
        for i in range(0, len(perm), BS):
            idx = perm[i:i + BS]
            opt.zero_grad()
            out = model(Xtr[idx])
            loss = crit(out, ytr[idx])
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xva).argmax(1) == yva).float().mean().item()
        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return best_acc


def evaluate(model, X, y):
    model.eval()
    with torch.no_grad():
        pred = model(X).argmax(1).cpu().numpy()
    y = y.cpu().numpy()
    return accuracy_score(y, pred), confusion_matrix(y, pred), f1_score(y, pred, average='macro')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=["default", "esm"], default="default",
                    help="default=dde/Baseline 特征(features.npz); esm=ESM-3B 特征(features_esm.npz)")
    ap.add_argument("--variant", default=None, help="输出文件后缀；默认 default 用原命名，esm 用 _esm")
    args = ap.parse_args()
    set_seed()
    print(f"[05b] device={DEVICE}")
    feat_map = cfg.get("training.feat_file_map")
    feat_name = feat_map[args.feat]
    suffix = args.variant or ("esm" if args.feat == "esm" else "")
    # 【缺陷#6】原实现把 suffix 算出来却没用，out_path 只由 --feat 决定，
    # 于是任何一次带 --variant 的运行都会覆盖上一次的输出，--variant 形同虚设。
    out_path = os.path.join(RES, f"step5b_direction_variants{f'_{suffix}' if suffix else ''}.json")
    print(f"[05b] 输出文件: {os.path.basename(out_path)}")

    d = np.load(os.path.join(FEAT, feat_name))
    # 配套的 labels/split（见 cfg.sidecar_npz）
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))
    print(f"[05b] 特征文件: {feat_name}  X_direction_dim={d['X_direction'].shape[1]}")

    X_dir = torch.tensor(d["X_direction"]).float().to(DEVICE)
    y_dir = torch.tensor(l["y_direction"]).to(DEVICE)
    dtr, dva, dte = s["d_tr"], s["d_va"], s["d_te"]

    out = {}

    # ===== V1: 3类 MLP+BN（更好地拟合全量方向）=====
    print("\n===== V1: 3类 MLP+BN =====")
    Xtr, Xva, Xte = X_dir[dtr], X_dir[dva], X_dir[dte]
    ytr, yva, yte = y_dir[dtr], y_dir[dva], y_dir[dte]
    m1 = DirMLP(Xtr.shape[1], nout=3).to(DEVICE)
    train_val(m1, Xtr, ytr, Xva, yva)
    acc1, cm1, f1_1 = evaluate(m1, Xte, yte)
    out['V1_3class_BN'] = {"acc": round(float(acc1), 4), "macro_f1": round(float(f1_1), 4),
                           "cm": cm1.tolist()}
    print(f"[05b] V1 Test Acc={acc1:.4f} macro-F1={f1_1:.4f}\n{cm1}")

    # ===== V2: activation vs inhibition 二分类（剔 neutral）=====
    print("\n===== V2: activation vs inhibition =====")
    seed = cfg.get("direction_variants.seed")
    # 方向索引唯一真相来自 config（缺陷#1），不再硬编码 0/2
    dmap = cfg.direction_label_map()
    i_neu, i_act = dmap["neutral"], dmap["activation"]
    mask = (y_dir != i_neu).cpu().numpy()
    X2 = X_dir[mask]
    y2 = (y_dir[mask] == i_act).long()      # activation=1, inhibition=0
    idx2 = np.arange(len(X2))
    tr2, te2 = train_test_split(idx2, test_size=cfg.get("direction_variants.v2_test_size"), random_state=seed)
    tr2v, va2 = train_test_split(tr2, test_size=cfg.get("direction_variants.v2_val_frac"), random_state=seed)
    m2 = DirMLP(X2.shape[1], nout=2).to(DEVICE)
    _ = train_val(m2, X2[tr2v], y2[tr2v], X2[va2], y2[va2])
    acc2, cm2, f12 = evaluate(m2, X2[te2], y2[te2])
    out['V2_act_vs_inh'] = {"acc": round(float(acc2), 4), "macro_f1": round(float(f12), 4),
                            "cm": cm2.tolist()}
    print(f"[05b] V2 act-vs-inh Acc={acc2:.4f} macro-F1={f12:.4f} (可分性上限参考)\n{cm2}")

    out['_feat'] = feat_name
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n===== 汇总（已存 {out_path}）=====")
    for k, v in out.items():
        if isinstance(v, dict) and 'acc' in v:
            print(f"  {k}: acc={v['acc']:.4f}")


if __name__ == "__main__":
    main()
