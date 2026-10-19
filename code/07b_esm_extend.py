#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7b — Extend the ESM-2 cache with NEW external-validation target sequences.

Reads data/external/chembl_ext_dataset.parquet, collects target_uniprot accessions
whose sequence (from the local UniProt FASTA) is not yet in the ESM cache, embeds
them with the IDENTICAL model/pooling as code/04b_esm_embed.py (full-sequence
weighted mean-pooling, segmented long sequences), and appends to
data/features/esm_3B/{embeddings.npy, ids.json}.

Idempotent: re-running after completion is a no-op.
"""
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

import config as C

cfg = C.load()
CODE = os.path.dirname(os.path.abspath(__file__))


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    esm_mod = _load_module("esm_embed_04b", os.path.join(CODE, "04b_esm_embed.py"))

    cache_dir = cfg.get_path("paths.esm_cache_dir")
    emb_f = os.path.join(cache_dir, "embeddings.npy")
    ids_f = os.path.join(cache_dir, "ids.json")
    emb = np.load(emb_f)
    ids = json.load(open(ids_f))
    cached_seqs = {r["seq"] for r in ids}
    print(f"[07b] existing cache: {len(cached_seqs):,} sequences ({emb.shape})")

    ds = pd.read_parquet(os.path.join(cfg.get_path("paths.external_dir"),
                                      "chembl_ext_dataset.parquet"))
    accs = sorted(set(ds["target_uniprot"].dropna().astype(str)))

    # local UniProt FASTA (gzip)
    import gzip
    fasta = {}
    acc, buf = None, []
    with gzip.open(cfg.get_path("paths.uniprot_fasta"), "rt") as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    fasta[acc] = "".join(buf)
                acc = line[1:].split("|")[1]
                buf = []
            elif acc:
                buf.append(line.strip())
    if acc:
        fasta[acc] = "".join(buf)
    print(f"[07b] FASTA entries: {len(fasta):,}")

    new_seqs = sorted({fasta[a] for a in accs if a in fasta} - cached_seqs)
    print(f"[07b] external targets: {len(accs):,}; NEW sequences to embed: {len(new_seqs)}")
    if not new_seqs:
        print("[07b] nothing to do.")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # --fp16：ESM-2-3B 以 FP32 加载约 12GB 权重，16GB 卡上会全量 OOM 兜底并把零向量
    # 静默写进缓存（历史上本脚本的 771 条新增即全部为零向量，污染了 ChEMBL 外部验证）。
    model, tokenizer = esm_mod.load_model(cfg.get_path("paths.esm_model_path"), device, fp16=True)
    t0 = time.time()
    vec_dict, n_seg, failures = esm_mod.run_in_batches(
        model, tokenizer, new_seqs, device, cfg.get("esm_embed.batch_size"))

    # 与 04b 同款硬闸门：拒绝把 OOM 兜底的零向量写入缓存
    if len(failures) > max(10, 0.01 * len(new_seqs)):
        raise RuntimeError(
            f"[07b] {len(failures)}/{len(new_seqs)} 条序列触发 OOM 兜底（>1%），已拒绝写入缓存。\n"
            f"  零向量会污染外部验证特征。请调小 esm_embed.max_aa_per_chunk 或换更大显存的卡。")

    new_emb = np.stack([vec_dict[i] for i in range(len(new_seqs))]).astype(np.float32)
    full_emb = np.vstack([emb, new_emb])
    full_ids = ids + [
        {"index": len(ids) + i, "hash": esm_mod.seq_hash(s), "seq": s}
        for i, s in enumerate(new_seqs)]
    # atomic-ish save (np.save appends .npy to names not ending in .npy)
    tmp_emb = emb_f + ".tmp.npy"
    np.save(tmp_emb, full_emb)
    os.replace(tmp_emb, emb_f)
    tmp_ids = ids_f + ".tmp.json"
    with open(tmp_ids, "w") as f:
        json.dump(full_ids, f, ensure_ascii=False)
    os.replace(tmp_ids, ids_f)

    stats = {
        "n_unique": int(len(full_ids)),
        "n_added_by_07b": int(len(new_seqs)),
        "n_segmented_added": int(n_seg),
        "dim": cfg.get("esm_embed.hidden_dim"),
        "elapsed_s": round(time.time() - t0, 1),
        "oom_fallbacks_added": len(failures),
    }
    with open(os.path.join(cache_dir, "esm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[07b] done: cache now {len(full_ids):,} sequences; added {len(new_seqs)} "
          f"in {stats['elapsed_s']}s; OOM fallbacks {len(failures)}")


if __name__ == "__main__":
    main()
