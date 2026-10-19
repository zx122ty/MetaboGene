#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 14 — "不确定度"有效性统计分析（开发文档: prompt/26-10-15/01-...md）

参考: You, Le & Chua, Results in Engineering (2026) —— EviDTI 不确定性检验。
核心口径:
  * 多个独立训练 run（BCE 5 seeds; cold 划分各 5 seeds; 证据 5 seeds）,
    不确定性统计在 run 内计算、跨 run 报 mean±std 与符号一致性计数;
  * TP/TN/FP/FN 四分中位数 + 箱线图（FP>TP, FN>TN 判据）;
  * Mann-Whitney U（对/错两组 u）+ Cliff's δ（=2×errAUROC−1）+ BH 校正;
  * 20 区间置信分层准确率 + Spearman 单调性;
  * error-detection AUROC per run（mean±std, >0.5 计数, bootstrap 95% CI）;
  * UQ 方法对比（证据熵/BCE熵/MCP/MC-Dropout/ensemble/温度缩放/u_dist 引用 13 号）
    + **DeLong 配对检验**（相关 AUROC 比较）;
  * 校准: ECE(10 bins, [0.5,1.0]) + Brier + 逐 bin 表 + 温度缩放前后。

阶段:
  --stage train-runs  训练全部 run 并落盘逐 run 预测（unc/runs/*.npz, 完成后写 RUNS_DONE）
  --stage analyze     统计分析 → results/step14_uncertainty_stats.json + step14_*.png
  --groups            逗号分隔: bce_rand,evid_rand,cold_drug,cold_target,dual_cold,cold_evid
"""
import argparse
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

import config as C

cfg = C.load()
EXT_DIR = cfg.get_path("paths.external_dir")
UNC_DIR = os.path.join(EXT_DIR, "unified_ext", "unc")
RUNS_DIR = os.path.join(UNC_DIR, "runs")
os.makedirs(RUNS_DIR, exist_ok=True)
RES_DIR = cfg.get_path("paths.results_dir")
SEED0 = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FP_DIM, PROT_DIM, PROJ = 2048, 2560, 1024
MC_PASSES = 30          # 论文口径: 30 次前向取 std
BOOT_N = 1000


def log(m):
    print(f"[14] {m}", flush=True)


class BinMLP(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.esm_proj = nn.Linear(PROT_DIM, PROJ)
        self.trunk = nn.Sequential(
            nn.Linear(FP_DIM + PROJ, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.binary_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.trunk(torch.cat([x[:, :FP_DIM], self.esm_proj(x[:, FP_DIM:])], dim=1))
        return self.binary_head(h).squeeze(-1)


class EvidBinMLP(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.esm_proj = nn.Linear(PROT_DIM, PROJ)
        self.trunk = nn.Sequential(
            nn.Linear(FP_DIM + PROJ, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.binary_head = nn.Linear(hidden, 2)

    def forward(self, x):
        h = self.trunk(torch.cat([x[:, :FP_DIM], self.esm_proj(x[:, FP_DIM:])], dim=1))
        return self.binary_head(h)


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================ 划分
def load_row_entities():
    """X_binary 行序 → (met, entrez) 数组（与 05c/12/13 一致）。

    缺陷#4 后 04 会剔除缺失模态行并按少数类下采样，行序 = [被保留的正样本 | 被保留的负样本]，
    不再是「全部 positives | 全部 negatives」。优先读 04 落盘的实体表 row_met/row_tar。
    """
    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(cfg.get_path("paths.features_dir"), feat_name))
    if "row_met" in d:
        return d["row_met"].astype("int64"), d["row_tar"].astype("int64")
    gold = pd.read_parquet(cfg.get_path("paths.positives"))[["met", "tar"]]
    neg = pd.read_parquet(cfg.get_path("paths.negatives"))[["met", "tar"]]
    met = np.concatenate([gold["met"].astype("int64").values,
                          neg["met"].astype("int64").values])
    tar = np.concatenate([gold["tar"].astype("int64").values,
                          neg["tar"].astype("int64").values])
    return met, tar


def entity_split(values, rng, te=0.10, va=0.10):
    vals = np.array(sorted(set(values.tolist())))
    rng.shuffle(vals)
    n_te = int(len(vals) * te)
    n_va = int(len(vals) * va)
    return (set(vals[n_te + n_va:].tolist()),
            set(vals[n_te:n_te + n_va].tolist()),
            set(vals[:n_te].tolist()))


def cold_split_indices(met, tar, kind, rng):
    """返回 (tr, va, te) 行索引; dual_cold 丢弃跨桶 pair 并报告丢弃率。"""
    if kind == "cold_drug":
        tr_c, va_c, te_c = entity_split(met, rng)
        tr = np.where(np.isin(met, list(tr_c)))[0]
        va = np.where(np.isin(met, list(va_c)))[0]
        te = np.where(np.isin(met, list(te_c)))[0]
        return tr, va, te, 0.0
    if kind == "cold_target":
        tr_t, va_t, te_t = entity_split(tar, rng)
        tr = np.where(np.isin(tar, list(tr_t)))[0]
        va = np.where(np.isin(tar, list(va_t)))[0]
        te = np.where(np.isin(tar, list(te_t)))[0]
        return tr, va, te, 0.0
    # dual_cold: 两端同桶才保留
    tr_c, va_c, te_c = entity_split(met, rng)
    tr_t, va_t, te_t = entity_split(tar, rng)
    cb = np.where(np.isin(met, list(te_c)), 2,
                  np.where(np.isin(met, list(va_c)), 1, 0))
    tb = np.where(np.isin(tar, list(te_t)), 2,
                  np.where(np.isin(tar, list(va_t)), 1, 0))
    keep_tr = (cb == 0) & (tb == 0)
    keep_va = (cb == 1) & (tb == 1)
    keep_te = (cb == 2) & (tb == 2)
    idx = np.arange(len(met))
    keep = keep_tr | keep_va | keep_te
    drop = 1 - keep.mean()
    return (idx[keep_tr], idx[keep_va], idx[keep_te], float(drop))


# ============================================================ 训练单 run
def _train_split(X, y, tr, va, seed, mode="bce", kl_w=0.1, anneal=10,
                 epochs=None, patience=None):
    """在给定行索引上训练, 返回 (model, Xva_tensor)。X 为 CPU float32 全矩阵。"""
    import random as _r
    _r.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    hidden = cfg.get("training.hidden_dim"); dropout = cfg.get("training.dropout")
    BS = cfg.get("training.batch_size"); LR = cfg.get("training.lr")
    EPOCHS = epochs or cfg.get("training.epochs"); PATIENCE = patience or cfg.get("training.patience")
    Xtr = torch.tensor(X[tr]).float().to(DEVICE)
    ytr = torch.tensor(y[tr]).float().to(DEVICE)
    Xva = torch.tensor(X[va]).float().to(DEVICE)
    yva = torch.tensor(y[va]).to(DEVICE)
    if mode == "bce":
        model = BinMLP(hidden, dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        crit = nn.BCEWithLogitsLoss()
        best_va, best_st, bad = -1, None, 0
        for ep in range(EPOCHS):
            model.train()
            order = np.random.permutation(len(Xtr))
            for i in range(0, len(order), BS):
                b = order[i:i + BS]
                opt.zero_grad()
                loss = crit(model(Xtr[b]), ytr[b])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred = (model(Xva).sigmoid() > 0.5).float()
                va_ = float((pred == yva).float().mean())
            if va_ > best_va:
                best_va, bad = va_, 0
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_st)
    else:
        from importlib import import_module
        m13 = _load13()
        model = m13.EvidBinMLP(hidden, dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        ytr_oh = F.one_hot(ytr.long(), 2).float()
        best_va, best_st, bad = -1, None, 0
        for ep in range(EPOCHS):
            lam = kl_w * min(1.0, (ep + 1) / anneal)
            model.train()
            order = np.random.permutation(len(Xtr))
            for i in range(0, len(order), BS):
                b = order[i:i + BS]
                opt.zero_grad()
                loss = m13.evidential_mse_loss(model(Xtr[b]), ytr_oh[b], 2, lam)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                p, _, _ = m13.evidential_probs(model(Xva))
                va_ = float((p.argmax(1) == yva).float().mean())
            if va_ > best_va:
                best_va, bad = va_, 0
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_st)
    model.eval()
    del Xtr, Xva, ytr, yva
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


def _load13():
    import importlib.util as ilu
    spec = ilu.spec_from_file_location(
        "m13", os.path.join(os.path.dirname(os.path.abspath(__file__)), "13_uncertainty_eval.py"))
    m13 = ilu.module_from_spec(spec)
    spec.loader.exec_module(m13)
    return m13


def predict_run(model, evidential, Xte, yte, Xva, yva):
    """测试+验证预测与不确定性（含 MC-Dropout std, 30 passes）。"""
    with torch.no_grad():
        Xt = torch.tensor(Xte).float().to(DEVICE)
        if evidential:
            p, _, S = m13_global.evidential_probs(model(Xt))
            p_te = p[:, 1].cpu().numpy()
            u_evi = (2.0 / S.squeeze(1)).cpu().numpy()
        else:
            z = model(Xt)
            p_te = z.sigmoid().cpu().numpy()
            u_evi = None
        logits_te = model(Xt) if not evidential else None
        # evidential 的"logit"用 evidence 平方根无意义, 温度缩放仅对 BCE 做
        ent = []
        model.train()
        ps = []
        for _ in range(MC_PASSES):
            if evidential:
                pp, _, _ = m13_global.evidential_probs(model(Xt))
            else:
                zz = model(Xt).sigmoid()
                pp = torch.stack([1 - zz, zz], dim=1)
            ps.append(pp)
            ent.append(-(pp * torch.log(pp.clamp_min(1e-12))).sum(1))
        P_mc = torch.stack(ps)  # [T, N, 2]
        u_mc = P_mc.std(0)[:, 1].cpu().numpy()
        ent = torch.stack(ent).mean(0).cpu().numpy()
        model.eval()
        u_ent = ent
        del Xt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # val
    with torch.no_grad():
        Xv = torch.tensor(Xva).float().to(DEVICE)
        if evidential:
            pv, _, _ = m13_global.evidential_probs(model(Xv))
            p_val = pv[:, 1].cpu().numpy()
            logits_val = None
        else:
            logits_val = model(Xv).cpu().numpy()
            p_val = torch.sigmoid(model(Xv)).cpu().numpy()
        del Xv
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"p": p_te, "y": yte, "u_ent": u_ent, "u_mc": u_mc, "u_evi": u_evi,
            "logits_te": logits_te.cpu().numpy() if logits_te is not None else None,
            "p_val": p_val, "y_val": yva, "logits_val": logits_val}


m13_global = None


def stage_train_runs(groups, seeds):
    global m13_global
    m13_global = _load13()
    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(cfg.get_path("paths.features_dir"), feat_name))
    X = d["X_binary"]
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    y_all = l["y_binary"].astype(np.int64)
    s = np.load(cfg.sidecar_npz("split", feat_name))
    met, tar = load_row_entities()
    hidden = cfg.get("training.hidden_dim"); dropout = cfg.get("training.dropout")

    todo = []
    if "bce_rand" in groups:
        for sd in seeds:
            todo.append(("bce_rand", sd, 1, "bce"))
    if "evid_rand" in groups:
        for sd in seeds:
            todo.append(("evid_rand", sd, 1, "evid"))
    for g in ("cold_drug", "cold_target", "dual_cold", "cold_evid"):
        if g in groups:
            for sd in seeds:
                todo.append((g, sd, 1, "evid" if g == "cold_evid" else "bce"))

    log(f"待训练 run: {len(todo)}")
    for gi, (kind, seed, rep, mode) in enumerate(todo):
        out = os.path.join(RUNS_DIR, f"{kind}_s{seed}_r{rep}.npz")
        if os.path.exists(out):
            log(f"跳过（已存在）: {os.path.basename(out)}")
            continue
        log(f"({gi+1}/{len(todo)}) {kind} seed={seed} mode={mode}")
        evidential = (mode == "evid")
        kl_w = 0.1; anneal = 10  # 13 号结论: 证据头需低 KL
        if kind in ("bce_rand", "evid_rand"):
            tr, va, te = s["b_tr"], s["b_va"], s["b_te"]
            drop = 0.0
        else:
            rng = np.random.RandomState(SEED0 + seed)  # cold 划分随 seed 变化（论文口径: 每 seed 新划分）
            tr, va, te, drop = cold_split_indices(met, tar, kind, rng)
        model = _train_split(X, y_all, tr, va, seed, mode=mode, kl_w=kl_w, anneal=anneal)
        outs = predict_run(model, evidential, X[te], y_all[te], X[va], y_all[va])
        outs["drop_rate"] = drop
        outs["n_tr"] = int(len(tr)); outs["n_va"] = int(len(va)); outs["n_te"] = int(len(te))
        np.savez_compressed(out, **outs)
        log(f"  saved {os.path.basename(out)} (test n={len(te):,}, drop={drop:.1%})")
    done_marker = os.path.join(RUNS_DIR, "RUNS_DONE")
    with open(done_marker, "w") as f:
        f.write("ok\n")
    log(f"RUNS_DONE 标记已写入: {done_marker}")


# ============================================================ 统计分析
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_paired_test(y, p1, p2):
    """配对 DeLong 检验: H0 AUROC(p1)=AUROC(p2)。返回 (auc1, auc2, z, p)。"""
    from scipy.stats import norm
    pos = y == 1; neg = y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return float("nan"), float("nan"), 0.0, 1.0
    tz1, tz2 = _midrank(p1), _midrank(p2)
    tx1, tx2 = _midrank(p1[pos]), _midrank(p2[pos])
    ty1, ty2 = _midrank(p1[neg]), _midrank(p2[neg])
    auc1 = (tz1[pos].sum() - m * (m + 1) / 2) / (m * n)
    auc2 = (tz2[pos].sum() - m * (m + 1) / 2) / (m * n)
    v01_1 = (tz1[pos] - tx1) / n; v10_1 = 1 - (tz1[neg] - ty1) / m
    v01_2 = (tz2[pos] - tx2) / n; v10_2 = 1 - (tz2[neg] - ty2) / m
    s01 = np.cov(v01_1, v01_2, ddof=1)
    s10 = np.cov(v10_1, v10_2, ddof=1)
    S = np.array([[s01[0, 0] / m + s10[0, 0] / n, s01[0, 1] / m + s10[0, 1] / n],
                  [s01[1, 0] / m + s10[1, 0] / n, s01[1, 1] / m + s10[1, 1] / n]])
    d = auc1 - auc2
    var_d = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    z = d / np.sqrt(var_d) if var_d > 0 else 0.0
    pval = float(2 * (1 - norm.cdf(abs(z))))
    return round(auc1, 4), round(auc2, 4), round(z, 3), round(pval, 5)


def bootstrap_ci(u, err, n_boot=BOOT_N, seed=0):
    rng = np.random.RandomState(seed)
    aucs = []
    n = len(u)
    for _ in range(n_boot):
        idx = rng.randint(0, n, n)
        e, uu = err[idx], u[idx]
        if e.sum() in (0, len(e)):
            continue
        from sklearn.metrics import roc_auc_score
        aucs.append(roc_auc_score(e, uu))
    if not aucs:
        return float("nan"), float("nan")
    return round(float(np.percentile(aucs, 2.5)), 4), round(float(np.percentile(aucs, 97.5)), 4)


def bh_adjust(pvals):
    """Benjamini–Hochberg。"""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    prev = 1.0
    for rank, i in enumerate(order[::-1]):
        k = m - rank
        prev = min(prev, p[i] * m / k)
        adj[i] = prev
    return adj


def stage_analyze():
    from scipy.stats import mannwhitneyu, spearmanr

    runs = {}
    for f in sorted(os.listdir(RUNS_DIR)):
        if f.endswith(".npz"):
            key = f[:-4]
            runs[key] = np.load(os.path.join(RUNS_DIR, f))

    def group_runs(prefix):
        return {k: v for k, v in runs.items() if k.startswith(prefix)}

    def fourfold(p, y, thr=0.5):
        pred = (p > thr).astype(int)
        tp = (pred == 1) & (y == 1); fp = (pred == 1) & (y == 0)
        fn = (pred == 0) & (y == 1); tn = (pred == 0) & (y == 0)
        return tp, tn, fp, fn

    def analyze_group(prefix, u_key="u_ent", label=""):
        rs = group_runs(prefix)
        if not rs:
            return None
        med = {k: [] for k in ("TP", "TN", "FP", "FN")}
        aucs, mwu_p = [], []
        fp_gt_tp, fn_gt_tn, auc_gt_half = 0, 0, 0
        eces, briers = [], []
        pooled_u, pooled_err = [], []
        n = None
        for k, z in sorted(rs.items()):
            p, y = z["p"], z["y"].astype(int)
            u = z[u_key]
            tp, tn, fp, fn = fourfold(p, y)
            if tp.sum():
                med["TP"].append(float(np.median(u[tp])))
            if tn.sum():
                med["TN"].append(float(np.median(u[tn])))
            if fp.sum():
                med["FP"].append(float(np.median(u[fp])))
            if fn.sum():
                med["FN"].append(float(np.median(u[fn])))
            err = ((p > 0.5).astype(int) != y)
            cor = ~err
            if err.sum() and cor.sum():
                from scipy.stats import mannwhitneyu as mwu_
                stat = mwu_(u[err], u[cor], alternative="greater")
                mwu_p.append(float(stat.pvalue))
            from sklearn.metrics import roc_auc_score
            if err.sum() and (len(err) - err.sum()):
                aucs.append(float(roc_auc_score(err, u)))
            auc_gt_half += (aucs[-1] > 0.5) if aucs else 0
            fp_gt_tp += (med["FP"][-1] > med["TP"][-1]) if med["FP"] and med["TP"] else 0
            fn_gt_tn += (med["FN"][-1] > med["TN"][-1]) if med["FN"] and med["TN"] else 0
            eces.append(ece_binary(p, y))
            briers.append(float(((p - y) ** 2).mean()))
            pooled_u.append(u); pooled_err.append(err)
            n = len(y)
        pooled_u = np.concatenate(pooled_u); pooled_err = np.concatenate(pooled_err)
        lo, hi = bootstrap_ci(pooled_u, pooled_err)
        from sklearn.metrics import roc_auc_score
        pooled_auc = round(float(roc_auc_score(pooled_err, pooled_u)), 4)
        out = {
            "n_runs": len(rs), "n_per_run": n,
            "median_u": {k: (round(float(np.median(v)), 4) if v else None)
                          for k, v in med.items()},
            "median_u_iqr": {k: [round(float(np.percentile(v, 25)), 4),
                                  round(float(np.percentile(v, 75)), 4)]
                              for k, v in med.items() if v},
            "consistency": {"FP>TP": f"{fp_gt_tp}/{len(rs)}", "FN>TN": f"{fn_gt_tn}/{len(rs)}",
                            "errAUROC>0.5": f"{auc_gt_half}/{len(rs)}"},
            "err_auroc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
            "err_auroc_std": round(float(np.std(aucs)), 4) if aucs else None,
            "err_auroc_pooled": pooled_auc,
            "err_auroc_boot_ci": [lo, hi],
            "mwu_p_runs": [round(v, 5) for v in mwu_p],
            "mwu_p_median": round(float(np.median(mwu_p)), 6) if mwu_p else None,
            "ece_mean": round(float(np.mean(eces)), 4), "brier_mean": round(float(np.mean(briers)), 4),
        }
        return out

    results = {"groups": {}}
    for prefix, label in (("bce_rand", "random"), ("cold_drug", "cold-drug"),
                          ("cold_target", "cold-target"), ("dual_cold", "dual-cold")):
        r = analyze_group(prefix)
        if r:
            results["groups"][label] = r
            log(f"{label}: {json.dumps(r)}")

    # ---------- B: 分层准确率（bce_rand 代表 run, seed42）+ H3 单调性 ----------
    key42 = "bce_rand_s42_r1"
    if key42 in runs:
        z = runs[key42]
        p, y = z["p"], z["y"].astype(int)
        u = z["u_ent"]
        order = np.argsort(u)
        accs = []
        nb = 20
        for b in range(nb):
            idx = order[b * len(y) // nb:(b + 1) * len(y) // nb]
            pred = (p[idx] > 0.5).astype(int)
            accs.append(round(float((pred == y[idx]).mean()), 4))
        rho, sp_p = spearmanr(np.arange(nb), accs)
        top5 = round(float(( (p[order[:len(y)//20]] > 0.5).astype(int) == y[order[:len(y)//20]] ).mean()), 4)
        results["stratified"] = {"decile_acc_20": accs,
                                 "spearman_bin_acc": round(float(rho), 3),
                                 "spearman_p": round(float(sp_p), 5),
                                 "top5pct_most_confident_acc": top5}
        log(f"分层: spearman={rho:.3f} (p={sp_p:.5}), top5% acc={top5}")

    # ---------- D: 方法对比 + DeLong（bce_rand 与 evid_rand, seed 42 配对） ----------
    kb, ke = "bce_rand_s42_r1", "evid_rand_s42_r1"
    if kb in runs and ke in runs:
        b, e = runs[kb], runs[ke]
        y = b["y"].astype(int)
        p_e, u_evi = e["p"], (e["u_evi"] if e["u_evi"] is not None else e["u_ent"])
        err_e = (p_e > 0.5).astype(int) != y
        ent_e = e["u_ent"]
        p_b, u_mc = b["p"], b["u_mc"]
        err_b = (p_b > 0.5).astype(int) != y
        ent_b = b["u_ent"]
        # 证据 ensemble 分歧（3 个证据 run 的 std）
        evid_ps = [runs[k]["p"] for k in ("evid_rand_s42_r1", "evid_rand_s43_r1", "evid_rand_s44_r1")
                   if k in runs]
        dis_e = np.std(np.stack(evid_ps), axis=0) if len(evid_ps) >= 2 else None  # 各模型 p 的逐样本 std
        # BCE ensemble 熵
        bce_ps = [runs[k]["p"] for k in ("bce_rand_s42_r1", "bce_rand_s43_r1", "bce_rand_s44_r1")
                  if k in runs]
        p_mean = np.mean(np.stack(bce_ps), axis=0)
        p2 = np.stack([1 - p_mean, p_mean], 1)
        u_ens = -(p2 * np.log(p2 + 1e-12)).sum(1)
        # 温度缩放（val 拟合 T）
        T_opt = 1.0
        if b["logits_val"] is not None:
            from scipy.optimize import minimize_scalar
            zval, yval = b["logits_val"], b["y_val"].astype(int)
            def nll(t):
                pp = 1 / (1 + np.exp(-zval / t))
                pp = np.clip(pp, 1e-12, 1 - 1e-12)
                return -np.mean(yval * np.log(pp) + (1 - yval) * np.log(1 - pp))
            res = minimize_scalar(nll, bounds=(0.05, 20.0), method="bounded")
            T_opt = round(float(res.x), 4)
        zte = b["logits_te"]
        p_ts = 1 / (1 + np.exp(-zte / T_opt)) if zte is not None else p_b
        methods = {}
        m_err = {"u_ent(evid)": (ent_e, err_e), "u_evi": (u_evi, err_e),
                 "u_mcp(bce)": np.maximum(p_b, 1 - p_b), "u_ent(bce)": ent_b,
                 "u_mc_dropout": u_mc, "u_temp_scaled(bce)": np.maximum(p_ts, 1 - p_ts)}
        if dis_e is not None:
            m_err["evid_ensemble_disagree"] = (dis_e, err_e)
        if len(bce_ps) >= 2:
            m_err["bce_ensemble_entropy"] = (u_ens, err_b)
        for k, v in m_err.items():
            u_, e_ = (v if isinstance(v, tuple) else (v, err_b))
            methods[k] = err_auroc_of(u_, e_, y)
        results["methods_s42"] = methods
        # DeLong: 在 BCE 模型的错误集上, 配对比较各不确定性源的 error-AUROC
        results["delong_evid_ent_vs_bce_mcp"] = dict(zip(
            ("auc_evid_ent", "auc_bce_mcp", "z", "p"),
            delong_paired_test(err_b.astype(int), ent_e, np.maximum(p_b, 1 - p_b))))
        results["delong_ens_vs_bce_ent"] = dict(zip(
            ("auc_ens", "auc_bce_ent", "z", "p"),
            delong_paired_test(err_b.astype(int), u_ens, ent_b)))
        log(f"方法对比: {json.dumps(methods)}")
        log(f"DeLong: {json.dumps(results['delong_evid_ent_vs_bce_mcp'])} / "
            f"{json.dumps(results['delong_ens_vs_bce_ent'])}")

    # ---------- F: 显著性总表（BH 校正, 二分类 rand 组 MWU p 序列） ----------
    if results["groups"].get("random", {}).get("mwu_p_runs"):
        ps = results["groups"]["random"]["mwu_p_runs"]
        adj = bh_adjust(ps)
        results["groups"]["random"]["mwu_p_bh_adjusted"] = [round(float(x), 5) for x in adj]
        results["groups"]["random"]["mwu_sig_runs"] = f"{int((np.array(ps) < 0.05).sum())}/{len(ps)}"

    with open(os.path.join(RES_DIR, "step14_uncertainty_stats.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("metrics -> results/step14_uncertainty_stats.json")


def err_auroc_of(u, e_, y):
    from sklearn.metrics import roc_auc_score
    if np.asarray(e_).sum() in (0, len(e_)):
        return float("nan")
    return round(float(roc_auc_score(np.asarray(e_).astype(int), np.asarray(u))), 4)


def ece_binary(p, y, bins=10):
    conf = np.maximum(p, 1 - p)
    correct = ((p > 0.5).astype(int) == y).astype(float)
    ece, n = 0.0, len(p)
    for b in range(bins):
        lo, hi = b / bins, (b + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            ece += m.sum() / n * abs(correct[m].mean() - conf[m].mean())
    return round(float(ece), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train-runs", "analyze", "all"], default="all")
    ap.add_argument("--groups", type=str,
                    default="bce_rand,evid_rand,cold_drug,cold_target,dual_cold")
    ap.add_argument("--seeds", type=str, default="42,43,44,45,46")
    args = ap.parse_args()
    groups = [g for g in args.groups.split(",") if g]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    if args.stage in ("train-runs", "all"):
        stage_train_runs(groups, seeds)
    if args.stage in ("analyze", "all"):
        stage_analyze()


if __name__ == "__main__":
    main()
