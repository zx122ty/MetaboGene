#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 11 — 扩展训练宇宙：HMDB 关联对作为弱监督扩展训练数据（互作任务）

动机: 10 号脚本产出 80.4 万 HMDB 关联正例（与内部 gold 重叠仅 0.41%）。
本实验检验"扩展训练宇宙"能否进一步提升内部测试性能。任务定义警示:
酶/转运体关联 ≠ 直接互作, 故扩展集仅作弱监督, 并做两个变体对比:

  sanity : 基线复训（仅内部训练集, 期望 ~0.9203）
  W      : 弱监督预训练（扩展集, pretrain_epochs=5）→ 内部训练集微调（早停同 Step5）
  J      : 加权联合（loss = BCE_内部 + 0.3 × BCE_扩展, ≤8 ep, 早停仍在内部 val）

扩展集构建纪律（防泄漏）: 剔除与内部 gold（全部 split）重叠对、与内部负样本冲突对、
无 entrez 映射对（uniprot→entrez 用 gold_with_uniprot 映射）。
评估: 内部 test 不变（同一 split/seed）。特征: ECFP4(CID→PubChem 缓存) + ESM 缓存查表。
扩展特征矩阵常驻 CPU, 逐批 H2D（GPU 放不下 800k×4608）。

产物: data/external/unified_ext/expansion_pairs.parquet
      results/step11_expanded_training_metrics.json
"""
import json
import os
import random
import sys

import numpy as np
import pandas as pd

import config as C

cfg = C.load()
EXT_DIR = cfg.get_path("paths.external_dir")
OUT_PAIRS = os.path.join(EXT_DIR, "unified_ext", "expansion_pairs.parquet")
RES_DIR = cfg.get_path("paths.results_dir")
HMDB_PARQUET = cfg.get_path("paths.hmdb_metabolites")
SEED = 42
W_JOINT = 0.3
PRETRAIN_EPOCHS = 5
JOINT_MAX_EPOCHS = 8


def log(m):
    print(f"[11] {m}", flush=True)


def build_expansion():
    acc2cid, _, _ = None, None, None
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "m10", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "10_external_sources_validation.py"))
    m10 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m10)  # 复用其装载函数（模块级只做配置/目录, 无副作用）
    acc2cid, cid_set, _ = m10.load_hmdb_full()
    gold_pairs, u2e, neg_pairs = m10.load_gold_overlap()
    fasta_accs, _ = m10.load_fasta_maps()

    pos = pd.read_parquet(os.path.join(EXT_DIR, "unified_ext", "hmdb_assoc_pairs.parquet"))
    pos = pos[pos["is_active"] == 1]
    log(f"扩展候选（10 号正例）: {len(pos):,}")
    rows, seen, n_no_e, n_gold, n_negconf = [], set(), 0, 0, 0
    for cid, acc in zip(pos["metabolite_pubchem_cid"], pos["target_uniprot"]):
        e = u2e.get(str(acc))
        if e is None or pd.isna(e):
            n_no_e += 1
            continue
        e = int(e)
        key = (int(cid), e)
        if key in seen:
            continue
        if key in gold_pairs:
            n_gold += 1
            continue
        if key in neg_pairs:
            n_negconf += 1
            continue
        seen.add(key)
        rows.append((int(cid), str(acc), e))
    log(f"扩展集: {len(rows):,} 对（无 entrez 映射 {n_no_e:,}, gold 重叠 {n_gold:,}, "
        f"负样本冲突 {n_negconf:,}）")
    pd.DataFrame(rows, columns=["metabolite_pubchem_cid", "target_uniprot",
                                "gene_entrez_id"]).to_parquet(OUT_PAIRS, index=False)
    return rows


def main():
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, average_precision_score
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "enc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_encode_features.py"))
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)

    if not os.path.exists(OUT_PAIRS):
        rows = build_expansion()
    else:
        rows = [(int(a), str(b), int(c)) for a, b, c in
                pd.read_parquet(OUT_PAIRS).itertuples(index=False)]
        log(f"扩展集缓存: {len(rows):,} 对")

    # ---------------- 特征 ----------------
    cids = [r[0] for r in rows]
    ups = [r[1] for r in rows]
    acc2seq, _ = (None, None)
    import importlib.util as ilu
    spec10 = ilu.spec_from_file_location(
        "m10", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "10_external_sources_validation.py"))
    m10 = ilu.module_from_spec(spec10)
    spec10.loader.exec_module(m10)
    acc2seq, _ = m10.load_fasta_maps()

    smile_cache_path = os.path.join(EXT_DIR, "pubchem_smiles_ext.json")
    smile_cache = json.load(open(smile_cache_path)) if os.path.exists(smile_cache_path) else {}
    uniq_cids = sorted({int(c) for c in cids})
    missing = [c for c in uniq_cids if str(c) not in smile_cache]
    if missing:
        import urllib.request
        UA = {"User-Agent": "Mozilla/5.0 (metabo-gene-research)"}
        log(f"PubChem 批量补 {len(missing):,} SMILES ...")
        for i in range(0, len(missing), 200):
            b = missing[i:i + 200]
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                   f"cid/{','.join(map(str, b))}/property/CanonicalSMILES/JSON")
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                            timeout=60) as r:
                    d = json.load(r)
                for rec in d.get("PropertyTable", {}).get("Properties", []):
                    smi = rec.get("CanonicalSMILES") or rec.get("ConnectivitySMILES")
                    if smi:
                        smile_cache[str(rec["CID"])] = smi
            except Exception as ex:
                log(f"WARN batch {i//200}: {ex}")
            json.dump(smile_cache, open(smile_cache_path, "w"))
    smis = [smile_cache.get(str(int(c))) for c in cids]
    seqs = [acc2seq.get(u) for u in ups]
    X_mol, has_mol = enc.encode_smiles_fp(smis)
    emb_path = os.path.join(cfg.get_path("paths.esm_cache_dir"), "embeddings.npy")
    ids_path = os.path.join(cfg.get_path("paths.esm_cache_dir"), "ids.json")
    esm_emb = np.load(emb_path)
    seq_to_idx = {r["seq"]: r["index"] for r in json.load(open(ids_path))}
    X_seq, has_seq = enc.encode_seq(seqs, protein_type="esm",
                                    esm_cache=(esm_emb, seq_to_idx), l2_norm=True)
    X_exp = np.hstack([X_mol, X_seq]).astype("float32")
    valid = has_mol & has_seq  # 仅用双模态齐全的扩展对做弱监督（防零向量污染）
    log(f"扩展特征: {X_exp.shape}, 双模态有效 {valid.mean():.1%}")
    X_exp = X_exp[valid]
    y_exp = np.ones(len(X_exp), dtype=np.float32)

    # ---------------- 内部数据 ----------------
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(cfg.get_path("paths.features_dir"), feat_name))
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))
    Xtr = torch.tensor(d["X_binary"][s["b_tr"]]).float().to(device)
    ytr = torch.tensor(l["y_binary"][s["b_tr"]]).to(device)
    Xva = torch.tensor(d["X_binary"][s["b_va"]]).float().to(device)
    yva = torch.tensor(l["y_binary"][s["b_va"]]).to(device)
    Xte = torch.tensor(d["X_binary"][s["b_te"]]).float().to(device)
    yte = l["y_binary"][s["b_te"]]
    X_exp_t = torch.tensor(X_exp)  # 常驻 CPU

    BS = cfg.get("training.batch_size"); LR = cfg.get("training.lr")
    PATIENCE = cfg.get("training.patience"); DROP = cfg.get("training.dropout")
    HIDDEN = cfg.get("training.hidden_dim")
    FP_DIM, PROT_DIM, PROJ = 2048, 2560, 1024
    npr = np.random

    class BinMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.esm_proj = nn.Linear(PROT_DIM, PROJ)
            self.trunk = nn.Sequential(
                nn.Linear(FP_DIM + PROJ, HIDDEN), nn.ReLU(), nn.Dropout(DROP),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROP))
            self.binary_head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            h = self.trunk(torch.cat([x[:, :FP_DIM], self.esm_proj(x[:, FP_DIM:])], dim=1))
            return self.binary_head(h).squeeze(-1)

    crit = nn.BCEWithLogitsLoss()

    def evaluate(model):
        model.eval()
        with torch.no_grad():
            p = model(Xte).sigmoid().cpu().numpy()
        return {"internal_test_roc_auc": round(float(roc_auc_score(yte, p)), 4),
                "internal_test_pr_auc": round(float(average_precision_score(yte, p)), 4)}

    def finetune(model, max_epochs=None, tag=""):
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        best_va, best_state, bad = -1, None, 0
        for ep in range(max_epochs or cfg.get("training.epochs")):
            model.train()
            order = npr.permutation(len(Xtr))
            for i in range(0, len(order), BS):
                b = order[i:i + BS]
                opt.zero_grad()
                loss = crit(model(Xtr[b]), ytr[b].float())
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                va = float(((model(Xva).sigmoid() > 0.5).float() == yva).float().mean())
            if va > best_va:
                best_va, bad = va, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        log(f"{tag} 微调完成（best val acc {best_va:.4f}）")
        return model

    results = {}

    # ---- sanity: 基线复训 ----
    m = finetune(BinMLP().to(device), tag="sanity(仅内部)")
    results["sanity_internal_only"] = evaluate(m)
    log(f"sanity: {results['sanity_internal_only']}")

    # ---- W: 弱监督预训练 → 微调 ----
    m = BinMLP().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    n_exp = len(X_exp_t)
    for ep in range(PRETRAIN_EPOCHS):
        m.train()
        order = npr.permutation(n_exp)
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            xb = X_exp_t[b].to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(m(xb), torch.ones(len(b), device=device))
            loss.backward(); opt.step()
        log(f"  W 预训练 ep{ep+1}/{PRETRAIN_EPOCHS} loss={loss.item():.4f}")
    m = finetune(m, tag="W(预训练→微调)")
    results["W_pretrain_finetune"] = evaluate(m)
    log(f"W: {results['W_pretrain_finetune']}")

    # ---- J: 加权联合 ----
    m = BinMLP().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    best_va, best_state, bad = -1, None, 0
    for ep in range(JOINT_MAX_EPOCHS):
        m.train()
        order = npr.permutation(len(Xtr))
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            eb = npr.randint(0, n_exp, size=len(b))  # 扩展负采样配对（同 batch 等量）
            xb_e = X_exp_t[eb].to(device, non_blocking=True)
            opt.zero_grad()
            loss = crit(m(Xtr[b]), ytr[b].float()) + W_JOINT * crit(m(xb_e), torch.ones(len(b), device=device))
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            va = float(((m(Xva).sigmoid() > 0.5).float() == yva).float().mean())
        log(f"  J ep{ep+1}/{JOINT_MAX_EPOCHS} val_acc={va:.4f}")
        if va > best_va:
            best_va, bad = va, 0
            best_state = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(best_state)
    results["J_weighted_joint"] = evaluate(m)
    log(f"J: {results['J_weighted_joint']}")

    results["n_expansion_pairs"] = int(n_exp)
    results["w_joint"] = W_JOINT
    results["pretrain_epochs"] = PRETRAIN_EPOCHS
    with open(os.path.join(RES_DIR, "step11_expanded_training_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)
    log(f"完成 -> results/step11_expanded_training_metrics.json")


if __name__ == "__main__":
    main()
