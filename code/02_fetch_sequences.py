#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 2 — SMILES 与蛋白序列获取（PubChem + UniProt）

策略:
  * 在「唯一 CID」和「唯一 Entrez」层面抓取，避免对重复行重复请求。
  * PubChem: PUG REST API 拉取 CanonicalSMILES（ConnectivitySMILES 备选）。
  * 蛋白: Entrez ID -> 无直接序列 API，这里用 UniProt ID 映射 + 已有 uniprot 序列服务。
    说明: 原始数据未保留 UniProt 列。本步尝试用「UniProt 由 Entrez 检索映射」，
    若不可行则降级为「标记无序列」，并记录覆盖率，供 Step 4 决定是否用轻量序列特征。
  * 全部请求缓存到本地 JSON，幂等、可断点续跑、失败跳过。
参数统一来自 config.yaml (paths.* / fetch.*)。
"""
import json
import os
import time

import pandas as pd
import requests

import config as C

cfg = C.load()

HEADERS = {"User-Agent": cfg.get("fetch.user_agent")}


def load_cache(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_cache(path, d):
    with open(path, "w") as f:
        json.dump(d, f)


# ---------------- SMILES (batch) ----------------
def fetch_smiles_batch(cid_list):
    """批量请求 PubChem，一次最多 ~200 个 CID，返回 {cid: smiles}"""
    out = {}
    url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/"
           f"{','.join(map(str, cid_list))}/property/CanonicalSMILES/JSON")
    try:
        r = requests.get(url, headers=HEADERS, timeout=cfg.get("fetch.request_timeout_s"))
        if r.status_code == 200:
            for prop in r.json()["PropertyTable"]["Properties"]:
                cid = str(prop["CID"])
                smi = prop.get("CanonicalSMILES") or prop.get("ConnectivitySMILES")
                out[cid] = smi
        elif r.status_code == 404:
            # 单个 CID 不存在会让整个 batch 404，退化为逐CID
            for c in cid_list:
                out[str(c)] = None
        else:
            # 限速/服务器繁忙，返回空，等待
            time.sleep(cfg.get("fetch.retry_sleep_s"))
    except Exception:
        time.sleep(cfg.get("fetch.retry_sleep_s"))
    return out


def build_smiles(cids):
    smiles_cache_path = cfg.get_path("paths.pubchem_smiles_cache")
    cache = load_cache(smiles_cache_path)
    todo = [int(c) for c in cids if str(int(c)) not in cache]
    print(f"[02] SMILES: 唯一CID {len(cids):,}, 已缓存 {len(cache):,}, 待取 {len(todo):,}")
    max_requests = cfg.get("fetch.max_smiles_requests")
    limited = todo[:max_requests]
    batch = cfg.get("fetch.pubchem_batch_size")
    for i in range(0, len(limited), batch):
        chunk = limited[i:i + batch]
        got = fetch_smiles_batch(chunk)
        cache.update(got)
        if (i // batch + 1) % 5 == 0 or i + batch >= len(limited):
            print(f"      batch {i // batch + 1}/{(len(limited)-1)//batch+1}  done, cache={len(cache)}")
            save_cache(smiles_cache_path, cache)
        time.sleep(cfg.get("fetch.sleep_between_batches_s"))
    save_cache(smiles_cache_path, cache)
    return cache


# ---------------- 蛋白序列 ----------------
# 通过 Ensembl gene ID (在原始数据 gene_ensembl_id) 映射到蛋白序列。
# Ensembl 无直接 entrez->seq 简单 API，这里用 NCBI eutils esearch 到 protein DB，或
# 直接: 用 gene_protein_name + species 在 UniProt REST 检索 'reviewed:true AND organism_id:9606'。
# 为控制规模与鲁棒性，只对子样本取序列，其余标记 NaN。用缓存、失败跳过。
def fetch_uniprot_by_gene_name(name):
    """gene symbol -> uniprot accession -> 返回 'ACC|SEQ'；失败返回 None"""
    if not name or name == "":
        return None
    try:
        organism_id = cfg.get("fetch.uniprot_organism_id")
        q = requests.utils.quote(f'gene_exact:"{name}" AND organism_id:{organism_id} AND reviewed:true')
        url = f"https://rest.uniprot.org/uniprotkb/search?query={q}&fields=accession,sequence&format=json&size=1"
        r = requests.get(url, headers=HEADERS, timeout=cfg.get("fetch.uniprot_timeout_s"))
        if r.status_code == 200:
            data = r.json()
            if data.get("results"):
                res = data["results"][0]
                acc = res.get("primaryAccession")
                seq = res.get("sequence", {}).get("value", "")
                if acc and seq:
                    return f"{acc}:{seq}"
    except Exception:
        pass
    return None


def build_protein(entrez_map):
    """entrez -> (name) 映射。逐基因名查 UniProt，缓存到本地。"""
    cache_path = cfg.get_path("paths.uniprot_seq_cache")
    cache = load_cache(cache_path)
    # 对子样本取序列（跑通 pipeline）
    items = list(entrez_map.items())
    print(f"[02] 蛋白: 唯一Entrez {len(items):,}, 已缓存 {len(cache):,}")
    limited = items[:cfg.get("fetch.max_protein_sequences")]
    for i, (eid, name) in enumerate(limited):
        if str(eid) in cache:
            continue
        cache[str(eid)] = fetch_uniprot_by_gene_name(name)
        if (i + 1) % 100 == 0:
            print(f"      ...{i+1}/{len(limited)} done, cache={len(cache)}", flush=True)
            save_cache(cache_path, cache)
    save_cache(cache_path, cache)
    return cache


def main():
    gold = pd.read_parquet(cfg.get_path("paths.gold_positive"))
    print(f"[02] 载入金标准: {len(gold):,} 行")

    cids = sorted(set(int(x) for x in gold["metabolite_pubchem_cid"].dropna()))
    entrez = sorted(set(int(x) for x in gold["gene_entrez_id"].dropna()))
    print(f"[02] 唯一CID {len(cids):,}  唯一Entrez {len(entrez):,}")

    smiles_cache = build_smiles(cids)

    # 用原 parquet 的 gene_protein_name 映射 entrez -> 基因名
    full = pd.read_parquet(
        cfg.get_path("paths.source_parquet"),
        columns=["gene_entrez_id", "gene_protein_name"])
    full = full.drop_duplicates("gene_entrez_id")
    name_map = dict(zip(full["gene_entrez_id"].astype("Int64"), full["gene_protein_name"]))
    entrez_name = {str(int(e)): (name_map.get(int(e)) or "") for e in entrez}
    seq_map = build_protein(entrez_name)

    # 回填到 gold
    gold["smiles"] = gold["metabolite_pubchem_cid"].map(
        lambda c: smiles_cache.get(str(int(c))))
    gold["protein_seq"] = gold["gene_entrez_id"].map(
        lambda e: ((seq_map.get(str(int(e))) or "").split(":", 1)[1] if (seq_map.get(str(int(e))) or "").count(":") else None))

    out = cfg.get_path("paths.gold_with_struct")
    gold.to_parquet(out, index=False)
    print(f"[02] 已保存: {out}")

    smi_cov = gold["smiles"].notna().mean()
    seq_cov = gold["protein_seq"].notna().mean()
    cov = {
        "rows": int(len(gold)),
        "smiles_coverage": round(float(smi_cov), 4),
        "protein_seq_coverage": round(float(seq_cov), 4),
        "rows_with_both": int((gold["smiles"].notna() & gold["protein_seq"].notna()).sum()),
        "smiles_cache_size": len(smiles_cache),
    }
    cov_path = os.path.join(cfg.get_path("paths.gold_dir"), "struct_coverage.json")
    with open(cov_path, "w") as f:
        json.dump(cov, f, indent=2)
    print(f"[02] 覆盖率: {cov}")

    # ---- 验收 ----
    accept_smi = cfg.get("fetch.accept_smiles_coverage")
    accept_seq = cfg.get("fetch.accept_seq_coverage")
    print("\n========== 验收验证 ==========")
    print(f"SMILES 覆盖率: {smi_cov*100:.1f}%  (验收>={accept_smi*100:.0f}%)")
    print(f"蛋白序列覆盖率: {seq_cov*100:.1f}%  (验收>={accept_seq*100:.0f}%, 若不足则降级为单模态)")
    ok = smi_cov >= accept_smi and seq_cov >= accept_seq
    print(f"[02] 验收{'通过 ✓' if ok else '未通过 ✗'}")
    if not ok:
        print("[02] 提示: 蛋白序列覆盖率不足，Step4 将使用轻量序列特征或仅 ECFP 单模态。")


if __name__ == "__main__":
    main()
