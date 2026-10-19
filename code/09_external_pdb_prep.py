#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 9 — 外部数据准备：PDB 人源共晶（内源代谢物配体）

流程:
  1) gemmi 解析 components.cif.gz（wwPDB CCD）→ comp_id → InChIKey 表
  2) 与本地 HMDB parquet（训练同源宇宙, v5.0）InChIKey 求交 → 候选代谢物 comp_id
     （排除水/缓冲液/单原子离子等晶体伪影, 列表见 ARTIFACTS）
  3) RCSB search API: Homo sapiens + X-RAY(≤3.0Å) + 含候选 comp_id 的条目（分批 IN 检索）
  4) RCSB GraphQL 批量取条目详情: 分辨率 / 非聚合体 comp_id / 聚合体 UniProt

产物 (data/external/pdb/):
  ccd_inchikey.tsv          comp_id → InChIKey
  ccd_hmdb_candidates.tsv   comp_id → HMDB accession 列表（候选代谢物配体）
  pdb_candidates.tsv        entry \t resolution \t comp_ids \t uniprots

用法: python code/09_external_pdb_prep.py [--phase N]
"""
import argparse
import gzip
import json
import os
import time

import pandas as pd
import requests

import config as C

cfg = C.load()
OUT = os.path.join(cfg.get_path("paths.external_dir"), "pdb")
os.makedirs(OUT, exist_ok=True)
CCD_GZ = os.path.join(OUT, "components.cif.gz")
HMDB_PARQUET = cfg.get_path("paths.hmdb_metabolites")

# 晶体学伪影配体（水 / 缓冲液组分 / 单原子离子）——虽部分为 HMDB 内源成员,
# 但共晶出现多因结晶条件而非生物学相互作用, 一并排除（报告需写明）
ARTIFACTS = {
    "HOH", "DOD", "WAT", "NA", "CL", "K", "MG", "CA", "ZN", "FE", "FE2", "MN",
    "CU", "CU1", "IOD", "BR", "F", "LI", "CS", "RB", "BA", "SR", "CD", "HG",
    "AU", "AG", "CO", "NI", "AL", "SO4", "PO4", "POP", "GOL", "EDO", "PEG",
    "PGE", "PG4", "P6G", "1PE", "PE4", "PE5", "PE6", "ACT", "ACY", "MES",
    "TRS", "EPE", "BME", "DDT", "DTT", "DMS", "MPD", "CIT", "FLC", "NO3",
    "NH4", "OXY", "CO2", "SCN", "IUM", "NCO", "TAR", "MO6", "WO5", "SEO",
}

SESS = requests.Session()
SESS.headers.update({"User-Agent": "metabo-gene-research"})


def phase1_ccd():
    out = os.path.join(OUT, "ccd_inchikey.tsv")
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        print("[9-pdb] phase1 cached")
        return
    print("[9-pdb] phase1: parsing components.cif.gz (gemmi)...")
    import gemmi
    raw = gzip.open(CCD_GZ, "rb").read().decode("utf-8", errors="replace")
    doc = gemmi.cif.read_string(raw)
    n = 0
    with open(out, "w") as f:
        f.write("comp_id\tinchikey\n")
        for blk in doc:  # CCD: 一个组件一个 data block
            tab = blk.find("_pdbx_chem_comp_descriptor.", ["comp_id", "type", "descriptor"])
            for r in tab:
                if r.str(1) == "InChIKey" and "-" in r.str(2):
                    f.write(f"{r.str(0)}\t{r.str(2)}\n")
                    n += 1
    print(f"[9-pdb] phase1 done: {n:,} InChIKey rows -> {out}")


def phase2_candidates():
    out = os.path.join(OUT, "ccd_hmdb_candidates.tsv")
    ccd = pd.read_csv(os.path.join(OUT, "ccd_inchikey.tsv"), sep="\t").drop_duplicates()
    hm = pd.read_parquet(HMDB_PARQUET, columns=["accession", "inchikey"])
    hm = hm[hm["inchikey"].astype(str).str.len() > 0].drop_duplicates()
    joined = ccd.merge(hm, on="inchikey", how="inner")
    joined = joined[~joined["comp_id"].isin(ARTIFACTS)]
    g = joined.groupby("comp_id")["accession"].apply(lambda s: ";".join(sorted(set(s))))
    g.reset_index().to_csv(out, sep="\t", index=False)
    print(f"[9-pdb] phase2 done: {len(g):,} candidate metabolite ligands -> {out}")
    return list(g.index)


def phase3_search(cands, rows_per_page=5000, in_batch=150):
    cache_path = os.path.join(OUT, "entry_ids.json")
    if os.path.exists(cache_path):
        ids = json.load(open(cache_path))
        print(f"[9-pdb] phase3 cached: {len(ids):,} entries")
        return ids
    entry_ids = set()
    total = 0
    for b in range(0, len(cands), in_batch):
        batch = cands[b:b + in_batch]
        start = 0
        while True:
            q = {
                "query": {"type": "group", "logical_operator": "and", "nodes": [
                    {"type": "terminal", "service": "text", "parameters": {
                        "attribute": "rcsb_entity_source_organism.taxonomy_lineage.name",
                        "operator": "exact_match", "value": "Homo sapiens"}},
                    {"type": "terminal", "service": "text", "parameters": {
                        "attribute": "rcsb_nonpolymer_entity_instance_container_identifiers.comp_id",
                        "operator": "in", "value": batch}},
                    {"type": "terminal", "service": "text", "parameters": {
                        "attribute": "exptl.method", "operator": "exact_match",
                        "value": "X-RAY DIFFRACTION"}},
                    {"type": "terminal", "service": "text", "parameters": {
                        "attribute": "rcsb_entry_info.resolution_combined",
                        "operator": "less_or_equal", "value": 3.0}},
                ]},
                "return_type": "entry",
                "request_options": {"paginate": {"start": start, "rows": rows_per_page},
                                    "results_content_type": ["experimental"]},
            }
            for a in range(4):
                r = SESS.post("https://search.rcsb.org/rcsbsearch/v2/query", json=q, timeout=60)
                if r.status_code == 200:
                    break
                if r.status_code == 204:
                    r = None
                    break
                time.sleep(2 * (a + 1))
            if not r:
                break
            j = r.json()
            total = j["total_count"]
            ids = [x["identifier"] for x in j["result_set"]]
            entry_ids.update(ids)
            start += rows_per_page
            if start >= total or not ids:
                break
        print(f"[9-pdb] phase3 batch {b//in_batch + 1}/{(len(cands)+in_batch-1)//in_batch}: "
              f"cumulative entries {len(entry_ids):,}", flush=True)
    json.dump(sorted(entry_ids), open(cache_path, "w"))
    print(f"[9-pdb] phase3 done: {len(entry_ids):,} human X-ray entries with metabolite ligands")
    return sorted(entry_ids)


def phase4_details(entry_ids, batch=20):
    out = os.path.join(OUT, "pdb_candidates.tsv")
    t0 = time.time()
    with open(out, "w") as f:
        f.write("entry\tresolution\tcomp_ids\tuniprots\n")
        for b in range(0, len(entry_ids), batch):
            chunk = entry_ids[b:b + batch]
            parts, names = [], []
            for i, e in enumerate(chunk):
                parts.append(
                    f'e{i}: entry(entry_id:"{e}") {{ exptl{{method}} '
                    f'rcsb_entry_info{{resolution_combined}} '
                    f'nonpolymer_entities{{rcsb_nonpolymer_entity_container_identifiers{{nonpolymer_comp_id}}}} '
                    f'polymer_entities{{rcsb_polymer_entity_container_identifiers{{'
                    f'reference_sequence_identifiers{{database_name database_accession}}}}}} }}')
                names.append(f"e{i}")
            gql = "{" + " ".join(parts) + "}"
            for a in range(4):
                try:
                    r = SESS.post("https://data.rcsb.org/graphql", json={"query": gql}, timeout=60)
                    if r.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2 * (a + 1))
            else:
                print(f"  [warn] batch {b} failed")
                continue
            data = r.json().get("data") or {}
            for i, e in enumerate(chunk):
                d = data.get(names[i])
                if not d:
                    continue
                res = (d.get("rcsb_entry_info") or {}).get("resolution_combined")
                comps = sorted({ne["rcsb_nonpolymer_entity_container_identifiers"]["nonpolymer_comp_id"]
                                for ne in (d.get("nonpolymer_entities") or [])
                                if ne.get("rcsb_nonpolymer_entity_container_identifiers")})
                ups = set()
                for pe in (d.get("polymer_entities") or []):
                    for ref in (pe.get("rcsb_polymer_entity_container_identifiers") or {}
                                  ).get("reference_sequence_identifiers") or []:
                        if ref.get("database_name") == "UniProt" and ref.get("database_accession"):
                            ups.add(ref["database_accession"])
                if comps and ups:
                    f.write(f'{e}\t{res}\t{";".join(comps)}\t{";".join(sorted(ups))}\n')
            if (b // batch) % 25 == 0:
                print(f"  phase4 {b:,}/{len(entry_ids):,}  ({time.time()-t0:.0f}s)", flush=True)
    n = sum(1 for _ in open(out)) - 1
    print(f"[9-pdb] phase4 done: {n:,} entry rows -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", type=int, default=4, help="跑到第几阶段(1-4)")
    args = ap.parse_args()
    t0 = time.time()
    phase1_ccd()
    if args.phase >= 2:
        cands = phase2_candidates()
    else:
        return
    if args.phase >= 3:
        entry_ids = phase3_search(cands)
    else:
        return
    if args.phase >= 4:
        phase4_details(entry_ids)
    print(f"[9-pdb] all done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
