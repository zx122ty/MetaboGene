#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3b — 给负样本补全结构特征（SMILES / protein_seq）

理由（反制审稿攻击点：模态不平衡）：正样本有 SMILES/序列、负样本无，会导致模型
靠「有无结构」当分类信号。故用与正样本相同的数据源（PubChem SMILES 缓存 + 本地 FASTA）
给负样本按 met/tar 补全结构。

复用：
  data/cache/pubchem_smiles.json     # pubchem_cid -> SMILES
  paths.uniprot_fasta                                           # uniprot -> seq
  data/gold/gold_positive_with_uniprot.parquet  # met/tar -> uniprot_id 映射

交付物：
  data/neg/negatives.parquet   # 更新：smiles / protein_seq 补全
  data/neg/gen_struct_coverage.json
参数统一来自 config.yaml (paths.*)。
"""
import gzip
import json
import os

import pandas as pd

import config as C

cfg = C.load()


def load_fasta():
    seq = {}
    acc = None
    buf = []
    with gzip.open(cfg.get_path("paths.uniprot_fasta"), "rt") as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    seq[acc] = "".join(buf)
                acc = line[1:].split("|")[1]
                buf = []
            elif acc:
                buf.append(line.strip())
        if acc:
            seq[acc] = "".join(buf)
    return seq


def main():
    # SMILES 缓存（全 24420 已拉全）
    with open(cfg.get_path("paths.pubchem_smiles_cache")) as f:
        smiles = json.load(f)

    # FASTA
    print("[03b] 解析 FASTA ...")
    fasta = load_fasta()
    print(f"[03b] FASTA 条目: {len(fasta):,}")

    # met/tar -> uniprot_id 映射（用全源表 entrez->uniprot，负样本 tar 需能命中）
    uid = pd.read_parquet(
        cfg.get_path("paths.source_parquet"),
        columns=["gene_entrez_id", "uniprot_id"])
    uid = uid.dropna(subset=["gene_entrez_id"])
    uid = uid.drop_duplicates("gene_entrez_id")
    uid["tar"] = uid["gene_entrez_id"].astype("int64")
    uniprot_of = dict(zip(uid["tar"], uid["uniprot_id"]))

    out_neg = cfg.get_path("paths.negatives")
    neg = pd.read_parquet(out_neg)
    print(f"[03b] 负样本: {len(neg):,} 行")

    # 补 smiles
    neg["smiles"] = neg["met"].map(lambda c: smiles.get(str(c)))
    # 补 protein_seq（经 tar -> uniprot_id）
    neg["uniprot_id_tmp"] = neg["tar"].map(lambda t: uniprot_of.get(int(t)) if pd.notna(t) else None)
    neg["protein_seq"] = neg["uniprot_id_tmp"].map(lambda u: fasta.get(u) if u else None)
    neg = neg.drop(columns=["uniprot_id_tmp"])

    neg.to_parquet(out_neg, index=False)

    cov = {
        "n": int(len(neg)),
        "smiles_cov": round(float(neg["smiles"].notna().mean()), 4),
        "protein_seq_cov": round(float(neg["protein_seq"].notna().mean()), 4),
        "both_cov": round(float((neg["smiles"].notna() & neg["protein_seq"].notna()).mean()), 4),
    }
    out_cov = cfg.get_path("paths.neg_struct_coverage")
    with open(out_cov, "w") as f:
        json.dump(cov, f, indent=2)
    print(f"[03b] 负样本结构覆盖率: {cov}")
    print(f"[03b] 已保存: {out_neg}")


if __name__ == "__main__":
    main()
