#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2b — 蛋白序列全量填充（用本地 uniprot_sprot.fasta.gz 按 uniprot_id 匹配）

修复：之前 gold_with_struct.protein_seq 仅 6.2%（只取了 300 条）。
现用本地 FASTA 全量解析，按 uniprot_id 命中填到 positives / gold_with_struct。

交付物：
  data/gold/gold_with_struct_fullseq.parquet   # gold 全量，protein_seq 覆盖率 ~92%
  data/neg/positives.parquet                   # 更新：正样本 protein_seq 全量
参数统一来自 config.yaml (paths.*)。
"""
import gzip
import json
import os

import pandas as pd

import config as C

cfg = C.load()


def load_fasta():
    """解析本地 FASTA -> {uniprot_accession: sequence}"""
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
    print("[02b] 解析本地 FASTA ...")
    fasta = load_fasta()
    print(f"[02b] FASTA 条目: {len(fasta):,}")

    gold = pd.read_parquet(cfg.get_path("paths.gold_with_uniprot"), columns=["uniprot_id"])
    gold = gold["uniprot_id"].dropna()
    uni = set(gold.unique())
    hit = sum(1 for u in uni if u in fasta)
    print(f"[02b] gold 唯一 uniprot: {len(uni):,}  FASTA 命中: {hit:,} ({100*hit/len(uni):.1f}%)")

    # 构造 uniprot -> seq 映射（只保留命中）
    seq_map = {u: fasta[u] for u in uni if u in fasta}

    # 更新 gold_with_struct（按 met/tar 键回填）
    struct = pd.read_parquet(cfg.get_path("paths.gold_with_struct"))
    uid = pd.read_parquet(cfg.get_path("paths.gold_with_uniprot"),
                          columns=["metabolite_pubchem_cid", "gene_entrez_id", "uniprot_id"])

    struct["met"] = struct["metabolite_pubchem_cid"].astype("int64")
    struct["tar"] = struct["gene_entrez_id"].astype("int64")
    uid["met"] = uid["metabolite_pubchem_cid"].astype("int64")
    uid["tar"] = uid["gene_entrez_id"].astype("int64")
    uid = uid.drop_duplicates(["met", "tar"])
    struct = struct.merge(uid[["met", "tar", "uniprot_id"]], on=["met", "tar"], how="left")
    struct["protein_seq"] = struct["uniprot_id"].map(seq_map)
    struct = struct.drop(columns=["met", "tar", "uniprot_id"])
    out_struct = cfg.get_path("paths.gold_with_struct_fullseq")
    struct.to_parquet(out_struct, index=False)

    # 更新 positives（pos 已含 uniprot_id 列，直接映射）
    out_pos = cfg.get_path("paths.positives")
    pos = pd.read_parquet(out_pos)
    pos["protein_seq"] = pos["uniprot_id"].map(seq_map)
    pos = pos.drop(columns=["uniprot_id"])
    pos.to_parquet(out_pos, index=False)

    cov = {
        "struct_seq_coverage": round(float(struct["protein_seq"].notna().mean()), 4),
        "pos_seq_coverage": round(float(pos["protein_seq"].notna().mean()), 4),
        "pos_smiles_coverage": round(float(pos["smiles"].notna().mean()), 4),
    }
    with open(cfg.get_path("paths.seq_coverage"), "w") as f:
        json.dump(cov, f, indent=2)
    print(f"[02b] 覆盖率: {cov}")
    print(f"[02b] 已保存: {out_struct} / {out_pos}")


if __name__ == "__main__":
    main()
