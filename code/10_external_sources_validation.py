#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 10 — 多源外部验证统一构建与打分（HMDB 关联 / PDB 共晶 / Reactome 方向 / DrugBank）

对齐 07 的 ChEMBL 协议: HMDB 内源集过滤 → 剔除与内部 gold 重叠 pair（报告剔除率,
同时检查与内部负样本的标签冲突）→ 同宇宙未报道负例 1:1 → 复用 04 特征编码
(ECFP4 + ESM 缓存查表) → 在内部 split/seed 重训 concat+proj1024 → 打分 + 审计表。

四个源（--stage build 产物 data/external/unified_ext/ 下）:
  hmdb_assoc   任务1。HMDB parquet protein_uniprot_ids（'|' 分隔）→ (CID, UniProt)。
  pdb          任务1。共晶: comp→HMDB accession→CID × 条目 UniProt（step9 产物）。
  reactome     任务2。chembi 方向边（step8 产物）→ (CID, UniProt, direction)。
  drugbank     任务1+2。药物卡 CSV(07b 回填 SMILES) × gene_name→UniProt(FASTA GN=);
               action 映射: agonist/activator/...→activation, inhibitor/antagonist/...→
               inhibition, binder/ligand/substrate/...→neutral（映射表见 ACTION_MAP）。
               注意: DrugBank 为"药物域外推", 不限制在 HMDB 内源集, 但输出 in_hmdb 列
               供二次 restricted 视图。

用法:
  python code/10_external_sources_validation.py --stage build
  python code/10_external_sources_validation.py --stage score
产物:
  data/external/unified_ext/<source>_pairs.parquet + build_stats.json
  results/step10_external_sources_metrics.json + step10_<source>_audit.csv
"""
import argparse
import gzip
import json
import os
import random

import numpy as np
import pandas as pd

import config as C

cfg = C.load()
EXT_DIR = cfg.get_path("paths.external_dir")
OUT_DIR = os.path.join(EXT_DIR, "unified_ext")
os.makedirs(OUT_DIR, exist_ok=True)
RES_DIR = cfg.get_path("paths.results_dir")

SEED = 42
HMDB_PARQUET = cfg.get_path("paths.hmdb_metabolites")
PDB_DIR = os.path.join(EXT_DIR, "pdb")
REACTOME_DIR = os.path.join(EXT_DIR, "reactome")
DRUGBANK_DIR = os.path.join(EXT_DIR, "drugbank")
DRUGBANK_CSV = os.path.join(DRUGBANK_DIR, "drugbank_final_result_with_smiles.csv")

# DrugBank action → 方向标签（训练标签序见 config.yaml training.direction_label_map:
# 0=neutral, 1=inhibition, 2=activation）
ACTION_MAP = {
    "agonist": "activation", "partial agonist": "activation", "activator": "activation",
    "positive allosteric modulator": "activation", "inducer": "activation",
    "potentiator": "activation", "stimulator": "activation", "coagonist": "activation",
    "inhibitor": "inhibition", "antagonist": "inhibition", "inverse agonist": "inhibition",
    "negative allosteric modulator": "inhibition", "blocker": "inhibition",
    "channel blocker": "inhibition", "inactivator": "inhibition",
    "disrupting agent": "inhibition", "suppressor": "inhibition",
    "binder": "neutral", "ligand": "neutral", "substrate": "neutral",
    "cofactor": "neutral", "antibody": "neutral", "adduct": "neutral",
    "chaperone": "neutral", "cleavage": "neutral", "oxidizer": "neutral",
    "reducing agent": "neutral", "immunotherapeutic?": "neutral",
}


def log(msg):
    print(f"[10] {msg}", flush=True)


# ------------------------------------------------------------------ 公共装载
def load_hmdb_full():
    """HMDB parquet → (acc2cid, cid_set, chebi2accs)。仅保留 CID 可解析的行。"""
    hm = pd.read_parquet(HMDB_PARQUET,
                         columns=["accession", "pubchem_compound_id", "chebi_id"])
    hm["cid"] = pd.to_numeric(hm["pubchem_compound_id"].astype(str).str.strip(),
                              errors="coerce")
    hm = hm[hm["cid"].notna()]
    hm["cid"] = hm["cid"].astype("int64")
    acc2cid = dict(zip(hm["accession"], hm["cid"]))
    cid_set = set(hm["cid"].unique())
    hm["chebi_num"] = hm["chebi_id"].astype(str).str.extract(r"(\d+)")[0]
    chebi2accs = (hm[hm["chebi_num"].notna()]
                  .groupby("chebi_num")["accession"].apply(lambda s: sorted(set(s)))
                  .to_dict())
    return acc2cid, cid_set, chebi2accs


def load_gold_overlap():
    """(cid, entrez) 金标对集合 + uniprot→entrez 映射 + 内部负样本对集合。"""
    gold = pd.read_parquet(cfg.get_path("paths.gold_positive"),
                           columns=["metabolite_pubchem_cid", "gene_entrez_id"])
    gold_pairs = set(zip(gold["metabolite_pubchem_cid"].astype("int64"),
                         gold["gene_entrez_id"].astype("int64")))
    u2e = {}
    up = cfg.get_path("paths.gold_with_uniprot")
    if os.path.exists(up):
        gu = pd.read_parquet(up, columns=["uniprot_id", "gene_entrez_id"]).dropna()
        u2e = dict(zip(gu["uniprot_id"].astype(str), gu["gene_entrez_id"].astype("int64")))
    neg_pairs = set()
    try:
        ng = pd.read_parquet(cfg.get_path("paths.negatives"))
        cols = {c.lower(): c for c in ng.columns}
        mcol, tcol = cols.get("met"), cols.get("tar")
        if mcol and tcol:
            ng2 = ng[[mcol, tcol]].dropna()
            neg_pairs = set(zip(ng2[mcol].astype("int64"), ng2[tcol].astype("int64")))
    except Exception as e:  # 内部负样本列名不确定时不阻塞，仅告警
        log(f"WARN: 读取内部负样本失败（跳过冲突检查）: {e}")
    return gold_pairs, u2e, neg_pairs


def load_fasta_maps():
    """本地 reviewed FASTA → (acc2seq, gene2accs)。gene 映射仅取 human (OX=9606)。"""
    acc2seq, gene2accs = {}, {}
    path = cfg.get_path("paths.uniprot_fasta")
    acc, seq, gn, is_human = None, [], None, False
    with gzip.open(path, "rt") as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    acc2seq[acc] = "".join(seq)
                h = line[1:].rstrip()
                parts = h.split("|")
                acc = parts[1] if len(parts) >= 2 else None
                gn, is_human, seq = None, False, []
                for tok in h.split():
                    if tok.startswith("GN="):
                        gn = tok[3:].split(";")[0]
                    if tok.startswith("OX=") and tok[3:].split(";")[0] == "9606":
                        is_human = True
                if is_human and gn and acc:
                    for g in gn.replace(" ", "").split(","):
                        if g and g not in ("-"):
                            gene2accs.setdefault(g.upper(), set()).add(acc)
            elif acc:
                seq.append(line.strip())
    if acc:
        acc2seq[acc] = "".join(seq)
    gene2accs = {g: sorted(a) for g, a in gene2accs.items()}
    log(f"FASTA: {len(acc2seq):,} seqs; human gene->acc: {len(gene2accs):,} genes")
    return acc2seq, gene2accs


def overlap_stats(pairs, gold_pairs, u2e, neg_pairs, tag):
    """pairs: list[(cid, uniprot)]。返回 (in_gold, in_neg, clean)。"""
    in_gold = in_neg = 0
    clean = []
    for cid, acc in pairs:
        e = u2e.get(str(acc))
        g = bool(pd.notna(e)) and (int(cid), int(e)) in gold_pairs
        n = bool(pd.notna(e)) and (int(cid), int(e)) in neg_pairs
        in_gold += g
        in_neg += n
        if not g:
            clean.append((cid, acc))
    log(f"{tag}: {len(pairs):,} pairs; gold 重叠剔除 {in_gold:,}; "
        f"与内部负样本冲突 {in_neg:,}（仅计数, 已随 gold 剔除逻辑保留原样）")
    return in_gold, in_neg, clean


def sample_negatives(rid2, met_pool, prot_pool, pos_pair_set, gold_pairs, u2e,
                     n_target, rng):
    """未报道负例: (met_pool × prot_pool) 中随机抽, 排除正例与 gold。"""
    met_pool = sorted(met_pool)
    prot_pool = sorted(prot_pool)
    out, seen, attempts = [], set(), 0
    limit = max(n_target * 60, 10000)
    while len(out) < n_target and attempts < limit:
        attempts += 1
        c = met_pool[rng.randrange(len(met_pool))]
        a = prot_pool[rng.randrange(len(prot_pool))]
        key = (int(c), str(a))
        if key in seen or key in pos_pair_set:
            continue
        e = u2e.get(key[1])
        if pd.notna(e) and (key[0], int(e)) in gold_pairs:
            continue
        seen.add(key)
        out.append(key)
    log(f"负例抽样: {len(out):,}（尝试 {attempts:,}）")
    return out


# ------------------------------------------------------------------ build
def build_hmdb_assoc(acc2cid, fasta_accs, gold_pairs, u2e, neg_pairs, rng):
    hm = pd.read_parquet(HMDB_PARQUET, columns=["accession", "protein_uniprot_ids"])
    pairs = []
    for acc_hm, ups in zip(hm["accession"], hm["protein_uniprot_ids"]):
        cid = acc2cid.get(acc_hm)
        if cid is None or pd.isna(ups):
            continue
        for u in str(ups).split("|"):
            u = u.strip()
            if u in fasta_accs:
                pairs.append((int(cid), u))
    pairs = sorted(set(pairs))
    log(f"HMDB-assoc 原始 (CID,UniProt) 对: {len(pairs):,}")
    in_gold, in_neg, clean = overlap_stats(pairs, gold_pairs, u2e, neg_pairs, "hmdb_assoc")
    prot_pool = {u for _, u in clean}
    met_pool = {c for c, _ in clean}
    negs = sample_negatives(None, met_pool, prot_pool, set(clean), gold_pairs, u2e,
                            min(len(clean), 50000), rng)
    df = pd.DataFrame(
        [(c, u, 1, "hmdb_assoc") for c, u in clean]
        + [(c, u, 0, "hmdb_assoc") for c, u in negs],
        columns=["metabolite_pubchem_cid", "target_uniprot", "is_active", "source"])
    stats = {"raw_pairs": len(pairs), "gold_overlap_removed": in_gold,
             "internal_neg_conflict": in_neg, "positives": len(clean),
             "negatives": len(negs)}
    return df, stats


def build_pdb(acc2cid, fasta_accs, gold_pairs, u2e, neg_pairs, rng):
    cand = pd.read_csv(os.path.join(PDB_DIR, "ccd_hmdb_candidates.tsv"), sep="\t")
    comp2accs = dict(zip(cand["comp_id"], cand["accession"]))
    pdb = pd.read_csv(os.path.join(PDB_DIR, "pdb_candidates.tsv"), sep="\t")
    pairs, n_entry = [], 0
    for comps, ups in zip(pdb["comp_ids"], pdb["uniprots"]):
        cids = set()
        for comp in str(comps).split(";"):
            for a in str(comp2accs.get(comp, "")).split(";"):
                if a in acc2cid:
                    cids.add(acc2cid[a])
        us = [u for u in str(ups).split(";") if u in fasta_accs]
        if cids and us:
            n_entry += 1
        for c in cids:
            for u in us:
                pairs.append((int(c), u))
    pairs = sorted(set(pairs))
    log(f"PDB 共晶原始 (CID,UniProt) 对: {len(pairs):,}（含候选配体条目 {n_entry:,}）")
    in_gold, in_neg, clean = overlap_stats(pairs, gold_pairs, u2e, neg_pairs, "pdb")
    prot_pool = set(pd.read_csv(os.path.join(PDB_DIR, "pdb_candidates.tsv"),
                                sep="\t")["uniprots"].map(
        lambda s: [u for u in str(s).split(";") if u in fasta_accs]).explode().dropna().astype(str))
    met_pool = {c for c, _ in clean}
    negs = sample_negatives(None, met_pool, prot_pool, set(clean), gold_pairs, u2e,
                            min(len(clean), 30000), rng)
    df = pd.DataFrame([(c, u, 1, "pdb") for c, u in clean]
                      + [(c, u, 0, "pdb") for c, u in negs],
                      columns=["metabolite_pubchem_cid", "target_uniprot", "is_active", "source"])
    stats = {"raw_pairs": len(pairs), "gold_overlap_removed": in_gold,
             "internal_neg_conflict": in_neg, "positives": len(clean),
             "negatives": len(negs), "entries_with_candidate_ligand": n_entry}
    return df, stats


def build_reactome(acc2cid, chebi2accs, fasta_accs, gold_pairs, u2e, neg_pairs):
    e = pd.read_csv(os.path.join(REACTOME_DIR, "regulation_edges.tsv"), sep="\t",
                    dtype={"chebi_id": str})
    rows, n_no_cid, conflicts = [], 0, 0
    for chebi, direction, reaction, ups in zip(e["chebi_id"], e["direction"],
                                               e["reaction"], e["uniprot_set"]):
        accs = chebi2accs.get(str(chebi), [])
        cids = [acc2cid[a] for a in accs if a in acc2cid]
        if not cids:
            n_no_cid += 1
            continue
        us = [u for u in str(ups).split(";") if u in fasta_accs]
        for c in cids:
            for u in us:
                rows.append((int(c), u, direction, reaction))
    df = pd.DataFrame(rows, columns=["metabolite_pubchem_cid", "target_uniprot",
                                     "direction", "reaction"])
    # 同 (cid,up) 方向冲突剔除
    agg = df.groupby(["metabolite_pubchem_cid", "target_uniprot"])["direction"].nunique()
    bad = set(agg[agg > 1].index)
    conflicts = int(df.set_index(["metabolite_pubchem_cid", "target_uniprot"]).index.isin(bad).sum())
    df = df[~df.set_index(["metabolite_pubchem_cid", "target_uniprot"]).index.isin(bad)]
    df = df.drop_duplicates(["metabolite_pubchem_cid", "target_uniprot", "direction"])
    log(f"Reactome 方向对: {len(df):,}（无 CID 映射剔除 {n_no_cid:,} 行; "
        f"方向冲突剔除 {conflicts:,} 行）")
    df.to_parquet(os.path.join(OUT_DIR, "reactome_pairs.parquet"), index=False)
    stats = {"edges": len(e), "direction_conflict_removed": conflicts,
             "no_cid_mapping_rows": n_no_cid,
             "pairs": len(df),
             "unique_pairs": int(df.groupby(["metabolite_pubchem_cid",
                                             "target_uniprot"]).ngroups),
             "direction_counts": df["direction"].value_counts().to_dict()}
    return stats


def build_drugbank(acc2cid, fasta_accs, gene2accs, gold_pairs, u2e, neg_pairs, rng):
    df0 = pd.read_csv(DRUGBANK_CSV, dtype=str, encoding="utf-8")
    log(f"DrugBank 行: {len(df0):,}; 有结构: {df0['pubchem_smiles'].notna().sum():,}")
    pos, n_no_gene, n_no_acc, n_acted = [], 0, 0, 0
    seen = set()
    for name, cid_s, genes, orgs, acts in zip(
            df0["name"], df0["pubchem_cid"], df0["targets_gene_name"],
            df0["targets_organism"], df0["targets_actions"]):
        if pd.isna(cid_s) or pd.isna(genes):
            continue
        try:
            cid = int(float(cid_s))
        except ValueError:
            continue
        gl = str(genes).split(";")
        ol = str(orgs).split(";") if pd.notna(orgs) else []
        al = str(acts).split(";") if pd.notna(acts) else []
        for i, g in enumerate(gl):
            g = g.strip()
            org = ol[i].strip() if i < len(ol) else ""
            act = al[i].strip().lower() if i < len(al) else ""
            if "human" not in org.lower():
                continue
            accs = gene2accs.get(g.upper(), [])
            if not accs:
                n_no_gene += 1
                continue
            acc = accs[0]
            if acc not in fasta_accs:
                n_no_acc += 1
                continue
            label = ACTION_MAP.get(act)
            if label:
                n_acted += 1
            key = (cid, acc)
            if key in seen:
                continue
            seen.add(key)
            pos.append((cid, acc, label))
    log(f"DrugBank (CID,UniProt) 对: {len(pos):,}（缺 gene 映射 {n_no_gene:,}, "
        f"acc 不在 FASTA {n_no_acc:,}; 有方向标签 {n_acted:,}）")
    pos_df = pd.DataFrame(pos, columns=["metabolite_pubchem_cid", "target_uniprot",
                                        "direction"])
    in_gold, in_neg, clean = overlap_stats(
        list(zip(pos_df["metabolite_pubchem_cid"], pos_df["target_uniprot"])),
        gold_pairs, u2e, neg_pairs, "drugbank")
    pos_df["is_gold_overlap"] = [
        (int(c), str(a)) not in set(map(tuple, clean)) for c, a in
        zip(pos_df["metabolite_pubchem_cid"], pos_df["target_uniprot"])]
    clean_pairs = set(map(tuple, clean))
    pos_df = pos_df[[ (int(c), str(a)) in clean_pairs
                      for c, a in zip(pos_df["metabolite_pubchem_cid"],
                                      pos_df["target_uniprot"])]].copy()
    pos_df["in_hmdb"] = pos_df["metabolite_pubchem_cid"].isin(acc2cid.values())
    drug_cids = sorted({int(c) for c in pos_df["metabolite_pubchem_cid"]})
    prot_pool = sorted({a for _, a in clean})
    negs = sample_negatives(None, set(drug_cids), set(prot_pool), clean_pairs,
                            gold_pairs, u2e, min(len(clean), 20000), rng)
    out = pd.DataFrame({
        "metabolite_pubchem_cid": [c for c, _ in clean] + [c for c, _ in negs],
        "target_uniprot": [a for _, a in clean] + [a for _, a in negs],
        "is_active": [1] * len(clean) + [0] * len(negs),
        "source": "drugbank"})
    out = out.merge(pos_df[["metabolite_pubchem_cid", "target_uniprot", "direction"]],
                    on=["metabolite_pubchem_cid", "target_uniprot"], how="left")
    out["in_hmdb"] = out["metabolite_pubchem_cid"].isin(acc2cid.values())
    out.to_parquet(os.path.join(OUT_DIR, "drugbank_pairs.parquet"), index=False)
    stats = {"raw_pairs": len(pos), "gold_overlap_removed": in_gold,
             "internal_neg_conflict": in_neg, "positives": len(clean),
             "negatives": len(negs), "with_direction_label": int(pos_df["direction"].notna().sum()),
             "direction_counts": pos_df["direction"].value_counts(dropna=False).to_dict(),
             "in_hmdb_subset": int(pos_df["in_hmdb"].sum())}
    return out, stats


def build_all():
    acc2cid, cid_set, chebi2accs = load_hmdb_full()
    log(f"HMDB: {len(acc2cid):,} accessions with CID; {len(chebi2accs):,} with ChEBI")
    fasta_accs, gene2accs = load_fasta_maps()
    gold_pairs, u2e, neg_pairs = load_gold_overlap()
    log(f"gold pairs: {len(gold_pairs):,}; internal neg pairs: {len(neg_pairs):,}")
    rng = random.Random(SEED)

    all_stats = {}
    df1, s1 = build_hmdb_assoc(acc2cid, fasta_accs, gold_pairs, u2e, neg_pairs, rng)
    df1.to_parquet(os.path.join(OUT_DIR, "hmdb_assoc_pairs.parquet"), index=False)
    all_stats["hmdb_assoc"] = s1
    df2, s2 = build_pdb(acc2cid, fasta_accs, gold_pairs, u2e, neg_pairs, rng)
    df2.to_parquet(os.path.join(OUT_DIR, "pdb_pairs.parquet"), index=False)
    all_stats["pdb"] = s2
    all_stats["reactome"] = build_reactome(acc2cid, chebi2accs, fasta_accs,
                                           gold_pairs, u2e, neg_pairs)
    df4, s4 = build_drugbank(acc2cid, fasta_accs, gene2accs, gold_pairs, u2e, neg_pairs, rng)
    all_stats["drugbank"] = s4
    all_stats["seed"] = SEED
    with open(os.path.join(OUT_DIR, "build_stats.json"), "w") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    log("build 完成 -> " + json.dumps(all_stats, ensure_ascii=False, indent=2))


# ------------------------------------------------------------------ score
def score_all():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "enc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_encode_features.py"))
    enc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(enc)

    # ---- 特征编码组件（与 07 相同）----
    acc2seq, _ = load_fasta_maps()
    smile_cache_path = os.path.join(EXT_DIR, "pubchem_smiles_ext.json")
    smile_cache = json.load(open(smile_cache_path)) if os.path.exists(smile_cache_path) else {}

    def smiles_for(cids, given=None):
        out = []
        miss = set()
        for c in cids:
            c = str(int(c))
            if given is not None:
                out.append(given)
                continue
            v = smile_cache.get(c)
            if v is None:
                miss.add(int(c))
            out.append(v)
        if miss:
            UA = {"User-Agent": "Mozilla/5.0 (metabo-gene-research)"}
            import urllib.request
            ml = sorted(miss)
            log(f"PubChem 批量补 {len(ml):,} 个 SMILES ...")
            for i in range(0, len(ml), 200):
                b = ml[i:i + 200]
                url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                       f"cid/{','.join(map(str, b))}/property/CanonicalSMILES/JSON")
                for a in range(3):
                    try:
                        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                                    timeout=60) as r:
                            d = json.load(r)
                        for rec in d.get("PropertyTable", {}).get("Properties", []):
                            smi = rec.get("CanonicalSMILES") or rec.get("ConnectivitySMILES")
                            if smi:
                                smile_cache[str(rec["CID"])] = smi
                        break
                    except Exception as ex:
                        if a == 2 and "400" in str(ex):
                            # 整批 400 = 个别坏 CID, 逐个回退
                            for c1 in b:
                                u1 = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                                      f"cid/{c1}/property/CanonicalSMILES/JSON")
                                try:
                                    with urllib.request.urlopen(urllib.request.Request(u1, headers=UA),
                                                                timeout=60) as r1:
                                        d1 = json.load(r1)
                                    for rec in d1.get("PropertyTable", {}).get("Properties", []):
                                        smi = rec.get("CanonicalSMILES") or rec.get("ConnectivitySMILES")
                                        if smi:
                                            smile_cache[str(rec["CID"])] = smi
                                except Exception:
                                    smile_cache[str(c1)] = None
                                smile_cache[str(c1)] = smile_cache.get(str(c1))
                        else:
                            log(f"WARN batch {i//200} attempt {a+1}: {ex}")
                json.dump(smile_cache, open(smile_cache_path, "w"))
            out = [smile_cache.get(str(int(c))) for c in cids]
        return out

    def encode_pairs(cid_list, up_list):
        smis = smiles_for(cid_list)
        seqs = [acc2seq.get(u) for u in up_list]
        X_mol, has_mol = enc.encode_smiles_fp(smis)
        emb_path = os.path.join(cfg.get_path("paths.esm_cache_dir"), "embeddings.npy")
        ids_path = os.path.join(cfg.get_path("paths.esm_cache_dir"), "ids.json")
        esm_emb = np.load(emb_path)
        seq_to_idx = {r["seq"]: r["index"] for r in json.load(open(ids_path))}
        X_seq, has_seq = enc.encode_seq(seqs, protein_type="esm",
                                        esm_cache=(esm_emb, seq_to_idx), l2_norm=True)
        X = np.hstack([X_mol, X_seq]).astype("float32")
        cov = {"smiles_cov": float(np.mean([s is not None for s in smis])),
               "seq_cov": float(np.mean([s is not None for s in seqs])),
               "has_seq_mask": float(has_seq.mean())}
        return X, cov, has_mol & has_seq

    # ---- 重训二分类模型（与 07 相同）----
    import random as _r
    import numpy.random as npr
    _r.seed(SEED); npr.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(cfg.get_path("paths.features_dir"), feat_name))
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))
    Xtr = torch.tensor(d["X_binary"][s["b_tr"]]).float().to(device)
    ytr = torch.tensor(l["y_binary"][s["b_tr"]]).to(device)
    Xva = torch.tensor(d["X_binary"][s["b_va"]]).float().to(device)
    yva = torch.tensor(l["y_binary"][s["b_va"]]).to(device)

    BS = cfg.get("training.batch_size"); LR = cfg.get("training.lr")
    EPOCHS = cfg.get("training.epochs"); PATIENCE = cfg.get("training.patience")
    HIDDEN = cfg.get("training.hidden_dim"); DROP = cfg.get("training.dropout")
    FP_DIM, PROT_DIM, PROJ = 2048, 2560, 1024

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

    def train_binary():
        model = BinMLP().to(device)
        opt = torch.optim.Adam(model.parameters(), lr=LR)
        crit = nn.BCEWithLogitsLoss()
        best_va, best_state, bad = -1, None, 0
        for ep in range(EPOCHS):
            model.train()
            order = npr.permutation(len(Xtr))
            for i in range(0, len(order), BS):
                b = order[i:i + BS]
                opt.zero_grad()
                loss = crit(model(Xtr[b]), ytr[b].float())
                loss.backward(); opt.step()
            model.eval()
            with torch.no_grad():
                ova = model(Xva)
                va = float(((ova.sigmoid() > 0.5).float() == yva).float().mean())
            if va > best_va:
                best_va, bad = va, 0
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= PATIENCE:
                    break
        model.load_state_dict(best_state)
        model.eval()
        return model

    bin_model = train_binary()
    Xte = torch.tensor(d["X_binary"][s["b_te"]]).float().to(device)
    yte = torch.tensor(l["y_binary"][s["b_te"]]).cpu().numpy()
    with torch.no_grad():
        p_int = bin_model(Xte).sigmoid().cpu().numpy()
    from sklearn.metrics import roc_auc_score, average_precision_score
    int_roc = roc_auc_score(yte, p_int)
    log(f"内部测试 ROC-AUC sanity: {int_roc:.4f} (期望 ~0.9217)")
    del Xtr, Xva, Xte, ytr, yva
    torch.cuda.empty_cache()

    # ---- 重训方向模型（与 Step5 任务2 相同协议; 3 类: 0=neutral 1=inhibition 2=activation）----
    X_dir_all = torch.tensor(d["X_direction"]).float().to(device)
    y_dir_all = torch.tensor(l["y_direction"]).to(device)
    dir_model = BinMLP().to(device)
    # 复用同一架构，但用 dir_head: 直接重建带 dir_head 的版本
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

    dir_model = DirMLP().to(device)
    opt2 = torch.optim.Adam(dir_model.parameters(), lr=LR)
    crit2 = nn.CrossEntropyLoss()
    d_tr, d_va = s["d_tr"], s["d_va"]
    Xd_tr, yd_tr = X_dir_all[d_tr], y_dir_all[d_tr]
    Xd_va, yd_va = X_dir_all[d_va], y_dir_all[d_va]
    max_ep = min(EPOCHS, cfg.get("training.direction_max_epochs"))
    best_da, best_ds, bad2 = -1, None, 0
    for ep in range(max_ep):
        dir_model.train()
        order = npr.permutation(len(Xd_tr))
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            opt2.zero_grad()
            loss = crit2(dir_model(Xd_tr[b]), yd_tr[b])
            loss.backward(); opt2.step()
        dir_model.eval()
        with torch.no_grad():
            da = float((dir_model(Xd_va).argmax(1) == yd_va).float().mean())
        if da > best_da:
            best_da, bad2 = da, 0
            best_ds = {k: v.clone() for k, v in dir_model.state_dict().items()}
        else:
            bad2 += 1
            if bad2 >= PATIENCE:
                break
    dir_model.load_state_dict(best_ds)
    dir_model.eval()
    with torch.no_grad():
        dte_pred = dir_model(X_dir_all[s["d_te"]]).argmax(1).cpu().numpy()
    dir_sanity = float((dte_pred == y_dir_all[s["d_te"]].cpu().numpy()).mean())
    log(f"内部方向 acc sanity: {dir_sanity:.4f} (期望 ~0.777)")
    del X_dir_all, Xd_tr, Xd_va, yd_tr, yd_va, y_dir_all
    torch.cuda.empty_cache()

    # ---- 各源打分 ----
    # 方向标签索引：唯一真相来自 config（缺陷#1，禁止手写字典）
    label2idx = cfg.direction_label_map()
    results = {"internal_test_roc_auc_sanity": round(float(int_roc), 4),
               "internal_direction_acc_sanity": round(float(dir_sanity), 4)}

    def score_task1(source, parquet, balance=False):
        ds = pd.read_parquet(parquet)
        if balance:
            act0 = ds[ds["is_active"] == 1]
            neg0 = ds[ds["is_active"] == 0]
            if len(act0) > len(neg0):
                act0 = act0.sample(len(neg0), random_state=SEED)
            ds = pd.concat([act0, neg0], ignore_index=True)
            log(f"{source}: 平衡视图 正例下采样至 {len(act0):,}")
        X, cov, valid_mask = encode_pairs(ds["metabolite_pubchem_cid"].tolist(),
                                          ds["target_uniprot"].tolist())
        Xe = torch.tensor(X).float().to(device)
        with torch.no_grad():
            p = bin_model(Xe).sigmoid().cpu().numpy()
        del Xe
        torch.cuda.empty_cache()
        ds["pred_prob"] = p
        act = ds[ds["is_active"] == 1]
        neg = ds[ds["is_active"] == 0]
        r = dict(cov, n_pos=int(len(act)), n_neg=int(len(neg)))
        y = ds["is_active"].values
        r["external_roc_auc"] = round(float(roc_auc_score(y, p)), 4)
        r["external_pr_auc"] = round(float(average_precision_score(y, p)), 4)
        # 特征完整子集上的指标（与训练口径一致：训练已剔除缺失模态行）
        r["n_valid_features"] = int(valid_mask.sum())
        if valid_mask.sum() > 20 and len(np.unique(y[valid_mask])) > 1:
            r["external_roc_auc_complete_only"] = round(
                float(roc_auc_score(y[valid_mask], p[valid_mask])), 4)
            r["external_pr_auc_complete_only"] = round(
                float(average_precision_score(y[valid_mask], p[valid_mask])), 4)
        med = float(np.median(neg["pred_prob"].values)) if len(neg) else 0.0
        r["frac_pos_above_median_neg"] = round(float((act["pred_prob"].values > med).mean()), 4)
        for kf in (0.01, 0.05, 0.10):
            k = max(1, int(kf * len(act)))
            top = ds.nlargest(k, "pred_prob")
            r[f"precision_at_top{int(kf*100)}pct_of_pos"] = round(float(top["is_active"].mean()), 4)
        audit = pd.concat([act.nlargest(50, "pred_prob"), neg.sample(min(20, len(neg)),
                                                                     random_state=SEED)],
                          ignore_index=True)
        audit.to_csv(os.path.join(RES_DIR, f"step10_{source}_audit.csv"), index=False)
        return r, ds

    def score_direction(source, ds):
        ds = ds[ds["direction"].notna()].copy()
        if ds.empty:
            return {"n_pairs": 0}
        X, cov, valid_mask = encode_pairs(ds["metabolite_pubchem_cid"].tolist(),
                                          ds["target_uniprot"].tolist())
        Xe = torch.tensor(X).float().to(device)
        with torch.no_grad():
            pred_all = dir_model(Xe).argmax(1).cpu().numpy()
        del Xe
        torch.cuda.empty_cache()
        # 全量审计落盘（含占位零向量行），便于追溯被剔除的行
        ds_all = ds.copy()
        ds_all["pred_label_idx"] = pred_all
        ds_all["feature_complete"] = valid_mask
        ds_all.to_csv(os.path.join(RES_DIR, f"step10_{source}_direction_audit.csv"),
                      index=False)
        # 【缺陷#4 口径统一】与训练一致：零填充行的预测无意义，主指标只在特征完整的行上计算
        n_before = int(len(ds))
        ds = ds[valid_mask]
        pred = pred_all[valid_mask]
        if ds.empty:
            return dict(cov, n_pairs=0, n_pairs_before_valid_filter=n_before)
        y = ds["direction"].map(label2idx).values
        r = dict(cov, n_pairs=int(len(ds)), n_pairs_before_valid_filter=n_before,
                 valid_feature_frac=round(float(valid_mask.mean()), 4))
        lab_all = sorted(set(y.tolist()))
        r["direction_acc"] = round(float((pred == y).mean()), 4)
        r["direction_macro_f1"] = round(float(f1_score(y, pred, labels=lab_all,
                                                       average="macro")), 4)
        # act-vs-inh 子集：直接按方向字符串选行（不再用硬编码索引，避免映射漂移）
        act_inh = {"activation", "inhibition"}
        m2 = ds["direction"].isin(act_inh).values
        if m2.sum() > 10:
            r["act_vs_inh_acc"] = round(float((pred[m2] == y[m2]).mean()), 4)
        return r

    for source in ("hmdb_assoc", "pdb"):
        pq = os.path.join(OUT_DIR, f"{source}_pairs.parquet")
        if os.path.exists(pq):
            r, _ = score_task1(source, pq, balance=(source == "hmdb_assoc"))
            results[source] = r
            log(f"{source}: {json.dumps(r)}")
    dqp = os.path.join(OUT_DIR, "drugbank_pairs.parquet")
    if os.path.exists(dqp):
        r, ds = score_task1("drugbank", dqp)
        results["drugbank_task1"] = r
        log(f"drugbank task1: {json.dumps(r)}")
        results["drugbank_direction"] = score_direction("drugbank", ds)
        results["drugbank_task1_in_hmdb"] = None
        sub = ds[ds["in_hmdb"] == True] if "in_hmdb" in ds.columns else ds.iloc[0:0]
        na = sub[sub["is_active"] == 1]
        ne = sub[sub["is_active"] == 0]
        if len(na) > len(ne):
            na = na.sample(len(ne), random_state=SEED)
        sub = pd.concat([na, ne])
        if len(sub) > 20:
            import tempfile
            tmp = os.path.join(OUT_DIR, "drugbank_in_hmdb_tmp.parquet")
            sub[["metabolite_pubchem_cid", "target_uniprot", "is_active"]].to_parquet(tmp)
            results["drugbank_task1_in_hmdb"], _ = score_task1("drugbank_in_hmdb", tmp)
        if os.path.exists(os.path.join(OUT_DIR, "reactome_pairs.parquet")):
            rq = pd.read_parquet(os.path.join(OUT_DIR, "reactome_pairs.parquet"))
            results["reactome_direction"] = score_direction("reactome", rq)
    with open(os.path.join(RES_DIR, "step10_external_sources_metrics.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log("metrics -> results/step10_external_sources_metrics.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "score"], required=True)
    args = ap.parse_args()
    if args.stage == "build":
        build_all()
    else:
        score_all()


if __name__ == "__main__":
    main()
