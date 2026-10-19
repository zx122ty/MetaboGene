#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 13 — 不确定性预测：证据深度学习 u_evi ⊕ 分布距离 u_dist → u_fused

方法（prompt/26-10-14/01-uncertain.md）:
  u_evi : 证据深度学习（Sensoy 式 Dirichlet, e=softplus(z), α=e+1, u=K/S）,
          损失 = MSE-II 型 + KL 退火；binary K=2（头改 2 单元, 取正类概率）、dir K=3。
  u_dist: 表征空间 kNN 距离 → 验证集统计量标准化 → sigmoid。双轨:
          entity（分子 ECFP 余弦 / 蛋白 ESM 余弦, α=0.5）与 pair（trunk 512 维隐层）。
  u_fused = 1-(1-u_evi)(1-u_dist)（noisy-OR, 分位数校准版）。

基线: MCP、熵、MC-Dropout×10、deep-ensemble(3×BCE)。
评估: error-detection AUROC（按各自模型的错误集）、ECE/Brier、覆盖率与主指标无损检查。
数据集: 内部 row test / strict pair-disjoint test / 外部三源（任务1）; 内部+strict+ctd+drugbank（任务2）。

阶段:
  --stage train  训练并缓存模型+评估集特征 (data/external/unified_ext/unc/)
  --stage eval   计算全部不确定性/融合/指标
"""
import argparse
import gzip
import importlib.util as ilu
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
import strict_split

cfg = C.load()
EXT_DIR = cfg.get_path("paths.external_dir")
UNC_DIR = os.path.join(EXT_DIR, "unified_ext", "unc")
os.makedirs(UNC_DIR, exist_ok=True)
RES_DIR = cfg.get_path("paths.results_dir")
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
KL_W = float(cfg.get("training.evidential_kl_weight", 1.0))
KL_ANNEAL = int(cfg.get("training.evidential_anneal_epochs", 5))
KNN_K = int(cfg.get("training.uncertainty_knn_k", 10))
DIST_ALPHA = float(cfg.get("training.uncertainty_dist_alpha", 0.5))
FP_DIM, PROT_DIM, PROJ = 2048, 2560, 1024


def log(m):
    print(f"[13] {m}", flush=True)


def load_enc():
    spec = ilu.spec_from_file_location(
        "enc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_encode_features.py"))
    enc = ilu.module_from_spec(spec)
    spec.loader.exec_module(enc)
    return enc


def _set_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================ 模型
class BinMLP(nn.Module):
    """标准 BCE 模型（与 07/10/11 相同）。"""
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

    def hidden(self, x):
        return self.trunk(torch.cat([x[:, :FP_DIM], self.esm_proj(x[:, FP_DIM:])], dim=1))


class EvidBinMLP(nn.Module):
    """证据版二分类: 头 2 单元, e=softplus(z)。"""
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

    def hidden(self, x):
        return self.trunk(torch.cat([x[:, :FP_DIM], self.esm_proj(x[:, FP_DIM:])], dim=1))


class DirMLP(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.esm_proj = nn.Linear(PROT_DIM, PROJ)
        self.trunk = nn.Sequential(
            nn.Linear(FP_DIM + PROJ, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.dir_head = nn.Linear(hidden, 3)

    def forward(self, x):
        h = self.trunk(torch.cat([x[:, :FP_DIM], self.esm_proj(x[:, FP_DIM:])], dim=1))
        return self.dir_head(h)


# ============================================================ 证据数学
def evidential_probs(e_logits):
    e = F.softplus(e_logits)
    alpha = e + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    return alpha / S, alpha, S


def kl_dirichlet(alpha, K):
    """KL(Dir(α) || Dir(1,...,1)), Sensoy 闭式。"""
    beta = torch.ones(1, K, device=alpha.device)
    S_alpha = alpha.sum(dim=1, keepdim=True)
    S_beta = beta.sum(dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.lgamma(alpha).sum(dim=1, keepdim=True)
    lnB_uni = torch.lgamma(S_beta) - torch.lgamma(beta).sum(dim=1, keepdim=True)
    dg0 = torch.digamma(S_beta)
    dg1 = torch.digamma(alpha)
    return ((alpha - beta) * (dg1 - dg0)).sum(dim=1, keepdim=True) + lnB - lnB_uni


def evidential_mse_loss(e_logits, y_onehot, K, lam):
    alpha = F.softplus(e_logits) + 1.0
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    err = ((y_onehot - p) ** 2).sum(dim=1, keepdim=True)
    var = (alpha * (S - alpha) / (S * S * (S + 1))).sum(dim=1, keepdim=True)
    alpha_t = y_onehot + (1.0 - y_onehot) * alpha
    return (err + var).mean() + lam * kl_dirichlet(alpha_t, K).mean()


# ============================================================ 训练
def train_binary(mode, seed, d, l, s):
    _set_seed(seed)
    hidden = cfg.get("training.hidden_dim"); dropout = cfg.get("training.dropout")
    BS = cfg.get("training.batch_size"); LR = cfg.get("training.lr")
    EPOCHS = cfg.get("training.epochs"); PATIENCE = cfg.get("training.patience")
    Xtr = torch.tensor(d["X_binary"][s["b_tr"]]).float().to(DEVICE)
    ytr = torch.tensor(l["y_binary"][s["b_tr"]]).to(DEVICE)
    Xva = torch.tensor(d["X_binary"][s["b_va"]]).float().to(DEVICE)
    yva = torch.tensor(l["y_binary"][s["b_va"]]).to(DEVICE)

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
                loss = crit(model(Xtr[b]), ytr[b].float())
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                pred = (model(Xva).sigmoid() > 0.5).float()
                va = float((pred == yva).float().mean())
            if va > best_va:
                best_va, bad = va, 0
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_st)
    else:
        model = EvidBinMLP(hidden, dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        ytr_oh = F.one_hot(ytr, 2).float()
        best_va, best_st, bad = -1, None, 0
        for ep in range(EPOCHS):
            lam = KL_W * min(1.0, (ep + 1) / KL_ANNEAL)
            model.train()
            order = np.random.permutation(len(Xtr))
            for i in range(0, len(order), BS):
                b = order[i:i + BS]
                opt.zero_grad()
                loss = evidential_mse_loss(model(Xtr[b]), ytr_oh[b], 2, lam)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                p, _, _ = evidential_probs(model(Xva))
                va = float((p.argmax(1) == yva).float().mean())
            if va > best_va:
                best_va, bad = va, 0
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


def train_direction(mode, d, l, s):
    _set_seed(SEED)
    hidden = cfg.get("training.hidden_dim"); dropout = cfg.get("training.dropout")
    BS = cfg.get("training.batch_size"); LR = cfg.get("training.lr")
    EPOCHS = min(cfg.get("training.epochs"), cfg.get("training.direction_max_epochs"))
    PATIENCE = cfg.get("training.patience")
    Xd = torch.tensor(d["X_direction"]).float().to(DEVICE)
    yd = torch.tensor(l["y_direction"]).to(DEVICE)
    d_tr, d_va = s["d_tr"], s["d_va"]

    if mode == "ce":
        model = DirMLP(hidden, dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        crit = nn.CrossEntropyLoss()
        best_va, best_st, bad = -1, None, 0
        for ep in range(EPOCHS):
            model.train()
            order = np.random.permutation(len(d_tr))
            for i in range(0, len(order), BS):
                b = order[i:i + BS]
                opt.zero_grad()
                loss = crit(model(Xd[d_tr][b]), yd[d_tr][b])
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va = float((model(Xd[d_va]).argmax(1) == yd[d_va]).float().mean())
            if va > best_va:
                best_va, bad = va, 0
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_st)
    else:
        model = DirMLP(hidden, dropout).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        y_oh = F.one_hot(yd, 3).float()
        best_va, best_st, bad = -1, None, 0
        for ep in range(EPOCHS):
            lam = KL_W * min(1.0, (ep + 1) / KL_ANNEAL)
            model.train()
            order = np.random.permutation(len(d_tr))
            for i in range(0, len(order), BS):
                b = order[i:i + BS]
                opt.zero_grad()
                loss = evidential_mse_loss(model(Xd[d_tr][b]), y_oh[d_tr][b], 3, lam)
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                p, _, _ = evidential_probs(model(Xd[d_va]))
                va = float((p.argmax(1) == yd[d_va]).float().mean())
            if va > best_va:
                best_va, bad = va, 0
                best_st = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_st)
    model.eval()
    del Xd, yd
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


# ============================================================ 预测+不确定性
def predict_binary_unc(model, X, mode, mc_passes=10):
    with torch.no_grad():
        Xt = torch.tensor(X).float().to(DEVICE)
        if mode == "evidential":
            p, _, S = evidential_probs(model(Xt))
            p_pos = p[:, 1].cpu().numpy()
            u_evi = (2.0 / S.squeeze(1)).cpu().numpy()
            ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1).cpu().numpy()
            u_mcp = np.maximum(p_pos, 1 - p_pos)
            u_mc = None
        else:
            z = model(Xt)
            p_pos = z.sigmoid().cpu().numpy()
            u_evi = None
            p2 = torch.stack([1 - z.sigmoid(), z.sigmoid()], dim=1)
            ent = -(p2 * torch.log(p2.clamp_min(1e-12))).sum(1).cpu().numpy()
            u_mcp = np.maximum(p_pos, 1 - p_pos)
            model.train()
            ents = []
            for _ in range(mc_passes):
                zz = model(Xt).sigmoid()
                p2m = torch.stack([1 - zz, zz], dim=1)
                ents.append(-(p2m * torch.log(p2m.clamp_min(1e-12))).sum(1))
            u_mc = torch.stack(ents).mean(0).cpu().numpy()
            model.eval()
        del Xt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return p_pos, u_evi, ent, u_mcp, u_mc


def predict_dir_unc(model, X, mode, mc_passes=10):
    with torch.no_grad():
        Xt = torch.tensor(X).float().to(DEVICE)
        if mode == "evidential":
            p, _, S = evidential_probs(model(Xt))
        else:
            p = torch.softmax(model(Xt), dim=1)
        pred = p.argmax(1).cpu().numpy()
        u_evi = ((p.shape[1] * 1.0) / p.sum(1)).cpu().numpy() if mode == "evidential" else None
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(1).cpu().numpy()
        model.train()
        ents = []
        for _ in range(mc_passes):
            pp = torch.softmax(model(Xt), dim=1) if mode == "evidential" \
                else torch.softmax(model(Xt), dim=1)
            ents.append(-(pp * torch.log(pp.clamp_min(1e-12))).sum(1))
        u_mc = torch.stack(ents).mean(0).cpu().numpy()
        model.eval()
        del Xt
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pred, u_evi, ent, u_mc


# ============================================================ 数据
def strict_split_rows(d=None):
    """复刻 05c 的 pair-disjoint 划分（确定性, seed 42）→ (tr, va, te)。

    缺陷#4 后 X_binary 行序 = [被保留的正样本 | 被保留的负样本]（04 已剔除缺失模态行
    并按少数类下采样），故正样本的 (met, tar) 取自 04 落盘的 row_met/row_tar 实体表，
    不再假设 positives.parquet 的全部行都在特征矩阵里。
    """
    if d is not None and "row_met" in d:
        row_met = d["row_met"].astype("int64")
        row_tar = d["row_tar"].astype("int64")
        n_pos = int(len(d["dir_met"]))
    else:
        pos = pd.read_parquet(cfg.get_path("paths.positives"))[["met", "tar"]]
        neg = pd.read_parquet(cfg.get_path("paths.negatives"))[["met", "tar"]]
        row_met = np.concatenate([pos["met"].astype("int64").values,
                                  neg["met"].astype("int64").values])
        row_tar = np.concatenate([pos["tar"].astype("int64").values,
                                  neg["tar"].astype("int64").values])
        n_pos = len(pos)
    sp = strict_split.build_strict_split(row_met, row_tar, n_pos, seed=SEED)
    return sp["b_tr"], sp["b_va"], sp["b_te"]


def _load_fasta():
    acc2seq, buf, acc = {}, [], None
    with gzip.open(cfg.get_path("paths.uniprot_fasta"), "rt") as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    acc2seq[acc] = "".join(buf)
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) >= 2 else None
                buf = []
            elif acc:
                buf.append(line.strip())
    if acc:
        acc2seq[acc] = "".join(buf)
    return acc2seq


def _esm_lookup():
    emb = np.load(os.path.join(cfg.get_path("paths.esm_cache_dir"), "embeddings.npy"))
    s2i = {r["seq"]: r["index"] for r in json.load(
        open(os.path.join(cfg.get_path("paths.esm_cache_dir"), "ids.json")))}
    return emb, s2i


def _cache_external(name, pair_arr, y, enc, tag="bin"):
    acc2seq = _load_fasta()
    smile_cache_path = os.path.join(EXT_DIR, "pubchem_smiles_ext.json")
    cache = json.load(open(smile_cache_path)) if os.path.exists(smile_cache_path) else {}
    cids = pair_arr[:, 0].astype("int64")
    smis = [cache.get(str(int(c))) for c in cids]
    seqs = [acc2seq.get(u) for u in pair_arr[:, 1]]
    X_mol, has_mol = enc.encode_smiles_fp(smis)
    emb, s2i = _esm_lookup()
    X_seq, has_seq = enc.encode_seq(seqs, protein_type="esm",
                                    esm_cache=(emb, s2i), l2_norm=True)
    X = np.hstack([X_mol, X_seq]).astype("float32")
    # 【缺陷#4 口径统一】与训练一致：零填充行构成「有无结构」的泄漏/噪声源，
    # 其预测也无意义，故只缓存特征完整的行（并记录剔除量）。
    keep = has_mol & has_seq
    y_arr = np.asarray(y)[keep]
    np.savez(os.path.join(UNC_DIR, f"{name}_{tag}.npz"), X=X[keep], y=y_arr,
             has_mol=has_mol[keep], has_seq=has_seq[keep], n_dropped=int((~keep).sum()))
    log(f"{name}_{tag}: {X[keep].shape}, 完整特征 {keep.mean():.1%} "
        f"(剔除 {int((~keep).sum())} 行)")


def stage_train():
    _set_seed(SEED)
    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(cfg.get_path("paths.features_dir"), feat_name))
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))

    out = os.path.join(UNC_DIR, "models.pt")
    hidden = cfg.get("training.hidden_dim")
    if os.path.exists(out):
        log("models.pt 已存在, 跳过训练")
    else:
        log("训练: 证据二分类 / BCE / 3×BCE ensemble / 证据方向 / CE方向")
        ev_bin = train_binary("evidential", SEED, d, l, s)
        bce_bin = train_binary("bce", SEED, d, l, s)
        ens = [train_binary("bce", SEED + 1 + i, d, l, s) for i in range(3)]
        ev_dir = train_direction("evidential", d, l, s)
        ce_dir = train_direction("ce", d, l, s)
        torch.save({"ev_bin": ev_bin.state_dict(), "bce_bin": bce_bin.state_dict(),
                    "ens": [m.state_dict() for m in ens],
                    "ev_dir": ev_dir.state_dict(), "ce_dir": ce_dir.state_dict(),
                    "hidden": hidden, "dropout": cfg.get("training.dropout")}, out)
        log(f"模型已保存 -> {out}")
    # 评估集特征缓存（train 矩阵太大不缓存, eval 阶段从 npz 现取）
    np.savez(os.path.join(UNC_DIR, "internal_bin.npz"),
             X=d["X_binary"][s["b_te"]], y=l["y_binary"][s["b_te"]])
    np.savez(os.path.join(UNC_DIR, "internal_dir.npz"),
             X=d["X_direction"][s["d_te"]], y=l["y_direction"][s["d_te"]])
    np.savez(os.path.join(UNC_DIR, "internal_val_bin.npz"),
             X=d["X_binary"][s["b_va"]], y=l["y_binary"][s["b_va"]])
    _, _, te = strict_split_rows(d)
    np.savez(os.path.join(UNC_DIR, "strict_bin.npz"),
             X=d["X_binary"][te], y=l["y_binary"][te])
    te_pos = te[te < d["X_direction"].shape[0]]  # 方向矩阵仅含正样本行
    np.savez(os.path.join(UNC_DIR, "strict_dir.npz"),
             X=d["X_direction"][te_pos], y=l["y_direction"][te_pos])
    log("内部/strict 特征缓存完成")

    enc = load_enc()
    hm = pd.read_parquet(os.path.join(EXT_DIR, "unified_ext", "hmdb_assoc_pairs.parquet"))
    a = hm[hm["is_active"] == 1]; n0 = hm[hm["is_active"] == 0]
    if len(a) > len(n0):
        a = a.sample(len(n0), random_state=SEED)
    hm_bal = pd.concat([a, n0], ignore_index=True)
    _cache_external("hmdb", hm_bal[["metabolite_pubchem_cid", "target_uniprot"]].values,
                    hm_bal["is_active"].values.astype(int), enc)
    for src in ("pdb", "drugbank"):
        p = pd.read_parquet(os.path.join(EXT_DIR, "unified_ext", f"{src}_pairs.parquet"))
        _cache_external(src, p[["metabolite_pubchem_cid", "target_uniprot"]].values,
                        p["is_active"].values.astype(int), enc)
    # 方向标签索引：唯一真相来自 config（缺陷#1，禁止手写字典）
    dir_map = cfg.direction_label_map()
    cq = pd.read_parquet(os.path.join(EXT_DIR, "unified_ext", "ctd_pairs.parquet"))
    _cache_external("ctd", cq[["metabolite_pubchem_cid", "target_uniprot"]].values,
                    cq["direction"].map(dir_map).values,
                    enc, tag="dir")
    dq = pd.read_parquet(os.path.join(EXT_DIR, "unified_ext", "drugbank_pairs.parquet"))
    dq = dq[dq["direction"].notna()]
    _cache_external("drugbank", dq[["metabolite_pubchem_cid", "target_uniprot"]].values,
                    dq["direction"].map(dir_map).values,
                    enc, tag="dir")
    log("外部特征缓存完成")


def stage_strict():
    """在 strict pair-disjoint split 上训练 binary/dir 的 BCE/CE 与证据版,
    供 'strict' 数据集使用合法的 strict 数字（row-level 模型评 strict 行是无效的）。"""
    import torch
    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(cfg.get_path("paths.features_dir"), feat_name))
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    tr, va, te = strict_split_rows(d)
    n_dir = d["X_direction"].shape[0]
    s_strict = {"b_tr": tr, "b_va": va, "d_tr": tr[tr < n_dir], "d_va": va[va < n_dir]}
    log("strict 模型训练: BCE / 证据二分类 / CE / 证据方向")
    bce_s = train_binary("bce", SEED, d, l, s_strict)
    ev_s = train_binary("evidential", SEED, d, l, s_strict)
    ce_s = train_direction("ce", d, l, s_strict)
    evd_s = train_direction("evidential", d, l, s_strict)
    ck_path = os.path.join(UNC_DIR, "models.pt")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ck.update({"bce_bin_strict": bce_s.state_dict(), "ev_bin_strict": ev_s.state_dict(),
               "ce_dir_strict": ce_s.state_dict(), "ev_dir_strict": evd_s.state_dict()})
    torch.save(ck, ck_path)
    log("strict 模型已并入 models.pt")


# ============================================================ eval
def hidden_of(model, X, bs=20000):
    hs = []
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i:i + bs]).float().to(DEVICE)
            hs.append(model.hidden(xb).cpu().numpy())
    return np.vstack(hs)


def knn_mean_dist(Q, R, k=10, bs=4096):
    """Q/R 均先 L2 归一化; 返回 Q 每行到 R 的 top-k 平均余弦距离 (1-cos)。"""
    Rt = torch.tensor(R).float().to(DEVICE)
    Rt = F.normalize(Rt, dim=1)
    out = np.zeros(len(Q), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(Q), bs):
            q = torch.tensor(Q[i:i + bs]).float().to(DEVICE)
            q = F.normalize(q, dim=1)
            sim = q @ Rt.T
            kk = min(k, sim.shape[1])
            top = torch.topk(sim, kk, dim=1).values
            out[i:i + bs] = (1 - top).mean(1).cpu().numpy()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


def err_auroc(u, err):
    from sklearn.metrics import roc_auc_score
    err = np.asarray(err).astype(int)
    if err.sum() in (0, len(err)):
        return float("nan")
    return round(float(roc_auc_score(err, u)), 4)


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


def rank01(x):
    from scipy.stats import rankdata
    return (rankdata(x) - 1) / max(1, len(x) - 1)


def sigmoid01(d, mu, sd):
    return 1 / (1 + np.exp(-(d - mu) / sd))


def stage_eval():
    hidden = cfg.get("training.hidden_dim"); dropout = cfg.get("training.dropout")
    ck = torch.load(os.path.join(UNC_DIR, "models.pt"), map_location=DEVICE, weights_only=False)
    ev_bin = EvidBinMLP(hidden, dropout).to(DEVICE); ev_bin.load_state_dict(ck["ev_bin"]); ev_bin.eval()
    bce_bin = BinMLP(hidden, dropout).to(DEVICE); bce_bin.load_state_dict(ck["bce_bin"]); bce_bin.eval()
    ens = []
    for st in ck["ens"]:
        m = BinMLP(hidden, dropout).to(DEVICE); m.load_state_dict(st); m.eval(); ens.append(m)
    ev_dir = DirMLP(hidden, dropout).to(DEVICE); ev_dir.load_state_dict(ck["ev_dir"]); ev_dir.eval()
    ce_dir = DirMLP(hidden, dropout).to(DEVICE); ce_dir.load_state_dict(ck["ce_dir"]); ce_dir.eval()
    strict_models = None
    if all(k in ck for k in ("bce_bin_strict", "ev_bin_strict", "ce_dir_strict", "ev_dir_strict")):
        bs_ = EvidBinMLP(hidden, dropout).to(DEVICE); bs_.load_state_dict(ck["ev_bin_strict"]); bs_.eval()
        bb_ = BinMLP(hidden, dropout).to(DEVICE); bb_.load_state_dict(ck["bce_bin_strict"]); bb_.eval()
        ds_ = DirMLP(hidden, dropout).to(DEVICE); ds_.load_state_dict(ck["ev_dir_strict"]); ds_.eval()
        dc_ = DirMLP(hidden, dropout).to(DEVICE); dc_.load_state_dict(ck["ce_dir_strict"]); dc_.eval()
        strict_models = {"ev_bin": bs_, "bce_bin": bb_, "ev_dir": ds_, "ce_dir": dc_}
        log("strict 数据集将使用 strict-split 训练的模型")

    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(cfg.get_path("paths.features_dir"), feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))
    train_X = d["X_binary"][s["b_tr"]]
    val_X = d["X_binary"][s["b_va"]]

    # u_dist-entity 参考集: 训练矩阵的唯一样式块
    def unique_blocks(X, sl):
        blk = X[:, sl]
        _, idx = np.unique(blk, axis=0, return_index=True)
        return blk[np.sort(idx)]

    mol_train = unique_blocks(train_X, slice(0, FP_DIM))
    prot_train = unique_blocks(train_X, slice(FP_DIM, FP_DIM + PROT_DIM))
    log(f"u_dist-entity 参考集: mol {len(mol_train):,} / prot {len(prot_train):,}")

    # 验证集统计量（逐行配对: 同一对的分子/蛋白距离加权）
    d_v = DIST_ALPHA * knn_mean_dist(val_X[:, :FP_DIM], mol_train, KNN_K) \
        + (1 - DIST_ALPHA) * knn_mean_dist(val_X[:, FP_DIM:FP_DIM + PROT_DIM], prot_train, KNN_K)
    mu_e, sd_e = float(d_v.mean()), float(d_v.std() + 1e-9)
    h_train = hidden_of(bce_bin, train_X)
    h_train_n = h_train / (np.linalg.norm(h_train, axis=1, keepdims=True) + 1e-9)
    d_v_h = knn_mean_dist(hidden_of(bce_bin, val_X), h_train_n, KNN_K)
    mu_h, sd_h = float(d_v_h.mean()), float(d_v_h.std() + 1e-9)
    log(f"u_dist 统计量(val): entity μ={mu_e:.4f} σ={sd_e:.4f}; pair-h μ={mu_h:.4f} σ={sd_h:.4f}")

    def dist_ent_of(X):
        d_m = knn_mean_dist(X[:, :FP_DIM], mol_train, KNN_K)
        d_p = knn_mean_dist(X[:, FP_DIM:FP_DIM + PROT_DIM], prot_train, KNN_K)
        return sigmoid01(DIST_ALPHA * d_m + (1 - DIST_ALPHA) * d_p, mu_e, sd_e)

    def dist_pair_of(X):
        h = hidden_of(bce_bin, X)
        h_n = h / (np.linalg.norm(h, axis=1, keepdims=True) + 1e-9)
        return sigmoid01(knn_mean_dist(h_n, h_train_n, KNN_K), mu_h, sd_h)

    from sklearn.metrics import roc_auc_score
    results = {"binary": {}, "direction": {},
               "config": {"knn_k": KNN_K, "dist_alpha": DIST_ALPHA,
                          "kl_weight": KL_W, "kl_anneal": KL_ANNEAL, "seed": SEED}}

    def eval_binary(name, path):
        z = np.load(path)
        X, y = z["X"], z["y"].astype(int)
        p_ev, u_evi, ent_ev, mcp_ev, _ = predict_binary_unc(ev_bin, X, "evidential")
        pred_ev = (p_ev > 0.5).astype(int); err_ev = (pred_ev != y)
        p_bc, _, ent_bc, mcp_bc, u_mc = predict_binary_unc(bce_bin, X, "bce")
        pred_bc = (p_bc > 0.5).astype(int); err_bc = (pred_bc != y)
        with torch.no_grad():
            ps = [m(torch.tensor(X).float().to(DEVICE)).sigmoid().cpu().numpy() for m in ens]
        p_ens = np.mean(ps, axis=0)
        u_ens = -(np.stack([1 - p_ens, p_ens], 1) *
                  np.log(np.stack([1 - p_ens, p_ens], 1) + 1e-12)).sum(1)
        err_ens = (p_ens > 0.5).astype(int) != y
        ud_ent = dist_ent_of(X)
        ud_pair = dist_pair_of(X)
        r_evi, r_de, r_dp = rank01(u_evi), rank01(ud_ent), rank01(ud_pair)
        res = {
            "n": int(len(y)),
            "acc_evidential": round(float((pred_ev == y).mean()), 4),
            "acc_bce": round(float((pred_bc == y).mean()), 4),
            "auc_evidential": round(float(roc_auc_score(y, p_ev)), 4),
            "auc_bce": round(float(roc_auc_score(y, p_bc)), 4),
            "err_rate_bce": round(float(err_bc.mean()), 4),
            "ece_bce": ece_binary(p_bc, y), "ece_evidential": ece_binary(p_ev, y),
            "brier_bce": round(float(((p_bc - y) ** 2).mean()), 4),
            "brier_evidential": round(float(((p_ev - y) ** 2).mean()), 4),
        }
        res["err_auroc_evidential"] = {
            "u_evi": err_auroc(u_evi, err_ev), "u_mcp": err_auroc(mcp_ev, err_ev),
            "u_ent": err_auroc(ent_ev, err_ev),
            "u_fused(evi+dist_ent)": err_auroc(1 - (1 - r_evi) * (1 - r_de), err_ev),
            "u_fused(evi+dist_pair)": err_auroc(1 - (1 - r_evi) * (1 - r_dp), err_ev)}
        res["err_auroc_bce"] = {
            "u_mcp": err_auroc(mcp_bc, err_bc), "u_ent": err_auroc(ent_bc, err_bc),
            "u_mc_dropout": err_auroc(u_mc, err_bc), "u_ens3": err_auroc(u_ens, err_ens),
            "u_dist_entity": err_auroc(ud_ent, err_bc),
            "u_dist_pair": err_auroc(ud_pair, err_bc),
            "u_fused(mcp+dist_pair)": err_auroc(1 - (1 - rank01(mcp_bc)) * (1 - r_dp), err_bc)}
        # 十分位准确率（基线模型 × fused-pair）
        order = np.argsort(1 - (1 - rank01(mcp_bc)) * (1 - r_dp))
        dec = []
        for b in range(10):
            idx = order[b * len(y) // 10:(b + 1) * len(y) // 10]
            dec.append(round(float((pred_bc[idx] == y[idx]).mean()), 4))
        res["decile_acc_fused"] = dec
        results["binary"][name] = res
        log(f"binary/{name}: err_auroc_evi={res['err_auroc_evidential']} "
            f"err_auroc_bce={res['err_auroc_bce']}")

    def eval_direction(name, path):
        z = np.load(path)
        X, y = z["X"], z["y"].astype(int)
        pred_ev, u_evi, ent_ev, _ = predict_dir_unc(ev_dir, X, "evidential")
        err_ev = (pred_ev != y)
        pred_ce, _, ent_ce, u_mc = predict_dir_unc(ce_dir, X, "ce")
        err_ce = (pred_ce != y)
        ud_ent = dist_ent_of(X)
        ud_pair = dist_pair_of(X)
        r_evi, r_dp = rank01(u_evi), rank01(ud_pair)
        res = {"n": int(len(y)),
               "acc_evidential": round(float((pred_ev == y).mean()), 4),
               "acc_ce": round(float((pred_ce == y).mean()), 4)}
        res["err_auroc_evidential"] = {
            "u_evi": err_auroc(u_evi, err_ev), "u_ent": err_auroc(ent_ev, err_ev),
            "u_fused(evi+dist_pair)": err_auroc(1 - (1 - r_evi) * (1 - r_dp), err_ev)}
        res["err_auroc_ce"] = {
            "u_ent": err_auroc(ent_ce, err_ce), "u_mc_dropout": err_auroc(u_mc, err_ce),
            "u_dist_entity": err_auroc(ud_ent, err_ce),
            "u_dist_pair": err_auroc(ud_pair, err_ce)}
        results["direction"][name] = res
        log(f"direction/{name}: {json.dumps(res)}")

    eval_binary("internal", os.path.join(UNC_DIR, "internal_bin.npz"))
    if strict_models is not None:
        _ev_bin_keep, _bce_bin_keep = ev_bin, bce_bin
        ev_bin, bce_bin = strict_models["ev_bin"], strict_models["bce_bin"]
        eval_binary("strict", os.path.join(UNC_DIR, "strict_bin.npz"))
        ev_bin, bce_bin = _ev_bin_keep, _bce_bin_keep
    else:
        eval_binary("strict", os.path.join(UNC_DIR, "strict_bin.npz"))
    eval_binary("hmdb", os.path.join(UNC_DIR, "hmdb_bin.npz"))
    eval_binary("pdb", os.path.join(UNC_DIR, "pdb_bin.npz"))
    eval_binary("drugbank", os.path.join(UNC_DIR, "drugbank_bin.npz"))
    eval_direction("internal", os.path.join(UNC_DIR, "internal_dir.npz"))
    if strict_models is not None:
        _ev_dir_keep, _ce_dir_keep = ev_dir, ce_dir
        ev_dir, ce_dir = strict_models["ev_dir"], strict_models["ce_dir"]
        eval_direction("strict", os.path.join(UNC_DIR, "strict_dir.npz"))
        ev_dir, ce_dir = _ev_dir_keep, _ce_dir_keep
    else:
        eval_direction("strict", os.path.join(UNC_DIR, "strict_dir.npz"))
    eval_direction("ctd", os.path.join(UNC_DIR, "ctd_dir.npz"))
    eval_direction("drugbank", os.path.join(UNC_DIR, "drugbank_dir.npz"))

    with open(os.path.join(RES_DIR, "step13_uncertainty_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("metrics -> results/step13_uncertainty_metrics.json")


def main():
    global KL_W, KL_ANNEAL
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["train", "eval", "all", "strict"], default="all")
    ap.add_argument("--kl-w", type=float, default=None, help="覆盖 evidential KL 权重（抢救实验用）")
    ap.add_argument("--anneal", type=int, default=None, help="覆盖 KL 退火轮数")
    args = ap.parse_args()
    if args.kl_w is not None:
        KL_W = args.kl_w
    if args.anneal is not None:
        KL_ANNEAL = args.anneal
    if args.stage in ("train", "all"):
        stage_train()
    if args.stage == "strict":
        stage_strict()
    if args.stage in ("eval", "all"):
        stage_eval()


if __name__ == "__main__":
    main()
