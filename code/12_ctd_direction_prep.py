#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 12 — CTD chemical–gene 方向外部集（构建 + 方向打分）

背景: 任务2 方向外部迁移失败（DrugBank 粗映射 0.595 / Reactome 不可用, §4.6.1）。
CTD 的 chemical–gene 相互作用带动作语义（如 increases^activity / inhibits^activity），
且 GeneID 即 NCBI Entrez —— 与内部金标宇宙直接同键（无需 UniProt 映射）。

化学侧映射: CTD CasRN → 本地 HMDB parquet cas_registry_number → PubChem CID
（只保留 HMDB 内源集, 与既有协议一致）。
方向解析（保守）: actions 中出现 "increases^activity"/"activates" → activation;
"inhibits"/"decreases^activity" → inhibition; 其余（表达、结合、代谢等）不计方向。
冲突对 (同 (CID,gene) 双向) 剔除并计数。

打分: 复训 Step5 方向模型（同协议）后在 CTD 对上评估 3 类 acc / act-vs-inh acc。

用法: python code/12_ctd_direction_prep.py [--skip-build]
产物: data/external/unified_ext/ctd_pairs.parquet
      results/step12_ctd_direction_metrics.json
"""
import argparse
import gzip
import json
import os
import sys

import numpy as np
import pandas as pd

import config as C

cfg = C.load()
EXT_DIR = cfg.get_path("paths.external_dir")
CTD_DIR = os.path.join(EXT_DIR, "ctd")
OUT_PARQUET = os.path.join(EXT_DIR, "unified_ext", "ctd_pairs.parquet")
RES_DIR = cfg.get_path("paths.results_dir")
HMDB_PARQUET = cfg.get_path("paths.hmdb_metabolites")
SEED = 42

ACT_INH = ("inhibits", "decreases^activity")
ACT_ACT = ("activates", "increases^activity")


def log(m):
    print(f"[12] {m}", flush=True)


def parse_actions(cell):
    """actions 形如 'increases^activity|affects^binding' → 'activation'/'inhibition'/None。
    同条目双向（既增又减 activity）视为冲突 → None。"""
    toks = set(str(cell).lower().split("|"))
    a = any(any(t == p or t.startswith(p) for p in ACT_ACT) for t in toks)
    i = any(any(t == p or t.startswith(p) for p in ACT_INH) for t in toks)
    if a and i:
        return "conflict"
    if a:
        return "activation"
    if i:
        return "inhibition"
    return None


def build():
    # HMDB: cas → (cid, accession)
    hm = pd.read_parquet(HMDB_PARQUET, columns=["accession", "pubchem_compound_id",
                                                "cas_registry_number"])
    hm["cid"] = pd.to_numeric(hm["pubchem_compound_id"].astype(str).str.strip(),
                              errors="coerce")
    hm = hm[hm["cid"].notna()]
    hm["cid"] = hm["cid"].astype("int64")
    cas2cid = {}
    for cas, cid in zip(hm["cas_registry_number"], hm["cid"]):
        if pd.isna(cas):
            continue
        for c in str(cas).split("|"):
            c = c.strip()
            if c:
                cas2cid.setdefault(c, cid)

    # 内部宇宙: entrez → uniprot（gold_with_uniprot 反查）
    up = cfg.get_path("paths.gold_with_uniprot")
    gu = pd.read_parquet(up, columns=["uniprot_id", "gene_entrez_id"]).dropna()
    e2u = {}
    for u, e in zip(gu["uniprot_id"].astype(str), gu["gene_entrez_id"].astype("int64")):
        e2u.setdefault(e, u)
    log(f"内部宇宙 entrez→uniprot: {len(e2u):,}; HMDB cas→cid: {len(cas2cid):,}")

    path = os.path.join(CTD_DIR, "CTD_chem_gene_ixns.csv.gz")
    usecols = None
    # 文件头部与表头均为 '#' 注释行 → 跳过并显式命名（见文件头 Fields 说明）
    cols = ["ChemicalName", "ChemicalID", "CasRN", "GeneSymbol", "GeneID", "GeneForms",
            "Organism", "OrganismID", "Interaction", "InteractionActions", "PubMedIDs"]
    act_col = "InteractionActions"
    it = pd.read_csv(path, names=cols, header=None, dtype=str,
                     compression="gzip", comment="#")
    log(f"CTD 行: {len(it):,}")
    it = it[it["OrganismID"] == "9606"]
    log(f"human: {len(it):,}")

    rows, n_no_chem, n_no_gene, n_conflict = [], 0, 0, 0
    seen = {}
    for cas, gid, act in zip(it["CasRN"], it["GeneID"], it[act_col]):
        cid = cas2cid.get(str(cas).strip()) if pd.notna(cas) else None
        if cid is None:
            n_no_chem += 1
            continue
        try:
            e = int(gid)
        except (ValueError, TypeError):
            continue
        u = e2u.get(e)
        if u is None:
            n_no_gene += 1
            continue
        d = parse_actions(act)
        if d == "conflict":
            n_conflict += 1
            continue
        if d is None:
            continue
        key = (int(cid), u)
        if key in seen:
            if seen[key] != d:
                seen[key] = "conflict"
                n_conflict += 1
            continue
        seen[key] = d
        rows.append((int(cid), u, d))
    df = pd.DataFrame(rows, columns=["metabolite_pubchem_cid", "target_uniprot",
                                     "direction"])
    # 二次过滤跨行冲突
    agg = df.groupby(["metabolite_pubchem_cid", "target_uniprot"])["direction"].nunique()
    bad = set(agg[agg > 1].index)
    df = df[~df.set_index(["metabolite_pubchem_cid", "target_uniprot"]).index.isin(bad)]
    df = df.drop_duplicates()
    log(f"CTD 方向对: {len(df):,}（无化学映射 {n_no_chem:,}, 无基因映射 {n_no_gene:,}, "
        f"方向冲突 {n_conflict:,}）")
    log(f"方向分布: {df['direction'].value_counts().to_dict()}")
    df.to_parquet(OUT_PARQUET, index=False)


def score():
    import torch
    import torch.nn as nn
    from sklearn.metrics import f1_score
    import importlib.util as ilu
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    spec = ilu.spec_from_file_location(
        "enc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_encode_features.py"))
    enc = ilu.module_from_spec(spec)
    spec.loader.exec_module(enc)

    ds = pd.read_parquet(OUT_PARQUET)
    acc2seq, _ = (None, None)
    spec10 = ilu.spec_from_file_location(
        "m10", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "10_external_sources_validation.py"))
    m10 = ilu.module_from_spec(spec10)
    spec10.loader.exec_module(m10)
    acc2seq, _ = m10.load_fasta_maps()
    seqs = [acc2seq.get(u) for u in ds["target_uniprot"]]
    smile_cache_path = os.path.join(EXT_DIR, "pubchem_smiles_ext.json")
    smile_cache = json.load(open(smile_cache_path)) if os.path.exists(smile_cache_path) else {}
    miss = sorted({int(c) for c in ds["metabolite_pubchem_cid"] if str(int(c)) not in smile_cache})
    if miss:
        import urllib.request, time
        UA = {"User-Agent": "Mozilla/5.0 (metabo-gene-research)"}
        log(f"PubChem 批量补 {len(miss):,} SMILES ...")
        for i in range(0, len(miss), 200):
            b = miss[i:i + 200]
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                   f"cid/{','.join(map(str, b))}/property/CanonicalSMILES/JSON")
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                    d = json.load(r)
                for rec in d.get("PropertyTable", {}).get("Properties", []):
                    smi = rec.get("CanonicalSMILES") or rec.get("ConnectivitySMILES")
                    if smi:
                        smile_cache[str(rec["CID"])] = smi
            except Exception as ex:
                log(f"WARN batch {i//200}: {ex}")
            json.dump(smile_cache, open(smile_cache_path, "w"))
    smis = [smile_cache.get(str(int(c))) for c in ds["metabolite_pubchem_cid"]]
    X_mol, has_mol = enc.encode_smiles_fp(smis)
    emb = np.load(os.path.join(cfg.get_path("paths.esm_cache_dir"), "embeddings.npy"))
    s2i = {r["seq"]: r["index"] for r in json.load(
        open(os.path.join(cfg.get_path("paths.esm_cache_dir"), "ids.json")))}
    X_seq, has_seq = enc.encode_seq(seqs, protein_type="esm",
                                    esm_cache=(emb, s2i), l2_norm=True)
    X = np.hstack([X_mol, X_seq]).astype("float32")
    valid = has_mol & has_seq

    # 复训方向模型（与 Step5/10 相同协议）
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    npr = np.random
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(cfg.get_path("paths.features_dir"), feat_name))
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))
    Xd = torch.tensor(d["X_direction"]).float().to(device)
    yd = torch.tensor(l["y_direction"]).to(device)
    BS = cfg.get("training.batch_size"); LR = cfg.get("training.lr")
    PATIENCE = cfg.get("training.patience"); DROP = cfg.get("training.dropout")
    HIDDEN = cfg.get("training.hidden_dim")
    PROT_DIM, PROJ, FP_DIM = 2560, 1024, 2048

    class DirMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.esm_proj = nn.Linear(PROT_DIM, PROJ)
            self.trunk = nn.Sequential(
                nn.Linear(FP_DIM + PROJ, HIDDEN), nn.ReLU(), nn.Dropout(DROP),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROP))
            self.dir_head = nn.Linear(HIDDEN, 3)

        def forward(self, x):
            h = self.trunk(torch.cat([x[:, :FP_DIM], self.esm_proj(x[:, FP_DIM:])], dim=1))
            return self.dir_head(h)

    m = DirMLP().to(device)
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    d_tr, d_va = s["d_tr"], s["d_va"]
    best_da, best_st, bad = -1, None, 0
    for ep in range(min(cfg.get("training.epochs"), cfg.get("training.direction_max_epochs"))):
        m.train()
        order = npr.permutation(len(d_tr))
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            opt.zero_grad()
            loss = crit(m(Xd[d_tr][b]), yd[d_tr][b])
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            da = float((m(Xd[d_va]).argmax(1) == yd[d_va]).float().mean())
        if da > best_da:
            best_da, bad = da, 0
            best_st = {k: v.clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    m.load_state_dict(best_st)
    m.eval()
    with torch.no_grad():
        int_pred = m(Xd[s["d_te"]]).argmax(1).cpu().numpy()
    dir_sanity = float((int_pred == yd[s["d_te"]].cpu().numpy()).mean())
    log(f"内部方向 acc sanity: {dir_sanity:.4f} (期望 ~0.777)")

    Xe = torch.tensor(X).float().to(device)
    with torch.no_grad():
        pred_all = m(Xe).argmax(1).cpu().numpy()
    # 方向标签索引：唯一真相来自 config（缺陷#1，禁止手写字典）
    label2idx = cfg.direction_label_map()
    # 全量审计落盘（含占位零向量行），便于追溯被剔除的行
    ds_out = ds.copy(); ds_out["pred_label_idx"] = pred_all
    ds_out["feature_complete"] = valid
    ds_out.to_csv(os.path.join(RES_DIR, "step12_ctd_direction_audit.csv"), index=False)
    # 【缺陷#4 口径统一】与训练一致：零填充行的预测无意义，主指标只在特征完整的行上计算
    n_before = int(len(ds))
    ds = ds[valid].reset_index(drop=True)
    pred = pred_all[valid]
    y = ds["direction"].map(label2idx).values
    res = {"smiles_cov": float(np.mean([x is not None for x in smis])),
           "seq_cov": float(np.mean([x is not None for x in seqs])),
           "n_pairs": int(len(ds)), "n_pairs_before_valid_filter": n_before,
           "valid_feature_frac": round(float(valid.mean()), 4),
           "internal_direction_acc_sanity": round(dir_sanity, 4)}
    r = res
    r["direction_acc"] = round(float((pred == y).mean()), 4)
    r["direction_macro_f1"] = round(float(f1_score(y, pred, labels=sorted(set(y)),
                                                   average="macro")), 4)
    # act-vs-inh 子集：直接按方向字符串选行（不再用硬编码索引，避免映射漂移）
    m2 = ds["direction"].isin({"activation", "inhibition"}).values
    if m2.sum() > 10:
        r["act_vs_inh_acc"] = round(float((pred[m2] == y[m2]).mean()), 4)
    with open(os.path.join(RES_DIR, "step12_ctd_direction_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    log(f"metrics: {json.dumps(res)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()
    if not args.skip_build:
        build()
    score()


if __name__ == "__main__":
    main()
