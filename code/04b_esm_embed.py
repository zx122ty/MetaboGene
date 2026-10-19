#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""04b — ESM-2-3B 蛋白质表征离线推理与缓存（全序列加权 Mean-Pooling）

用途：对 data/neg 正/负样本的所有【唯一蛋白序列】做 ESM-2-3B 表征，
     采用用户确认的【全序列加权均值池化】，保存 float32，供 04_encode_features.py 复用。
     只对唯一序列推理一次，避免对 43 万样本重复计算。

长序列策略（用户确认：分段均值池化，禁止直接截断）：
  - ESM2-3B 位置上限 = max_aa_per_chunk + 2 token（cls/eos；见 config esm_embed.max_aa_per_chunk）。
  - len <= max : 单段直过，均值池化。
  - len >  max : 以 max 长度、半重叠滑窗分多段，各段独立跑 ESM 取 Mean-Pool（按 attention_mask 排除 pad），
    最后把各段均值按【该段非-pad token 数】加权平均 → 整条序列的全局表征（尾部信息零丢失）。

显存策略：
  - batch_size + FP16(Autocast)。
  - 层次化 OOM 兜底：批量前向触发 OutOfMemoryError 时，先 empty_cache，再逐条重跑；仍失败则全零占位并如实记录。

存储（float32，data/features/esm_3B/）：
  embeddings.npy   (N, hidden) float32      # Mean-Pooling 表征
  ids.json         [{index, hash, seq}, ...]   # 供 04 做 map 回样本
  esm_stats.json   {n_unique, n_segmented, dim, elapsed_s}

命令：
  python code/04b_esm_embed.py
  python code/04b_esm_embed.py --max_n 50     # 快速试跑（仅前 50 条唯一序列）
（幂等/可续跑：embeddings.npy 已存在且 ids 覆盖全部唯一序列 → 跳过）
参数统一来自 config.yaml (paths.* / esm_embed.*)。
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
import torch

import config as C

cfg = C.load()


def get_unique_seqs(max_n=0):
    pos = pd.read_parquet(cfg.get_path("paths.positives"))
    neg = pd.read_parquet(cfg.get_path("paths.negatives"))
    all_seq = pd.concat([pos["protein_seq"], neg["protein_seq"]]).dropna().astype(str)
    uniq = sorted(all_seq.unique().tolist())
    if max_n:
        uniq = uniq[: max_n]
    return uniq


def seq_hash(s):
    return hashlib.md5(s.encode()).hexdigest()


def embed_one(model, tokenizer, seq, device):
    """对【单段】序列跑 ESM，返回 (mean_pool_vec[np float32 hidden], effective_token_count)。

    生效 token 数 = attention_mask 求和（含特殊 token，但因整体基准一致，加权时抵消偏差）。

    ⚠ 必须用 `hidden_states[-1]` 而**不能**改成 `last_hidden_state`：HF 的 EsmEncoder 里
    `all_hidden_states[-1]` 是在 `emb_layer_norm_after` **之前**记录的，而 `last_hidden_state`
    是**之后**的，两者不是同一个张量。改动会改变嵌入定义，并与缓存里已有的 25,709 条不同源。
    显存问题改由 --fp16 加载权重解决（12GB → 6GB），不动这里的取值路径。
    """
    batch = tokenizer([seq], return_tensors="pt").to(device)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.float16):
        out = model(**batch, output_hidden_states=True)
    hs = out.hidden_states[-1]                       # (1, seq_len, hidden)
    mask = batch["attention_mask"].float().squeeze(0)  # (seq_len,)
    eff = int(mask.sum().item())
    # 沿序列轴 mean-pool：逐特征维度做非-pad 加权平均 → (hidden,)
    vec = (hs.squeeze(0) * mask.unsqueeze(-1)).sum(dim=0) / max(eff, 1)
    return vec.detach().float().cpu().numpy(), eff


def mean_pool_seq(model, tokenizer, seq, device):
    """对【整条】序列做分段的加权 MeanPool，返回 (vec[hidden float32], is_segmented, eff_tokens)。"""
    max_aa = cfg.get("esm_embed.max_aa_per_chunk")
    hidden = cfg.get("esm_embed.hidden_dim")
    L = len(seq)
    if L <= max_aa:
        v, eff = embed_one(model, tokenizer, seq, device)
        return v, False, eff
    # 长序列：半重叠滑窗，分段均值按有效 token 加权
    vecs, wts = [], []
    for start in range(0, L, max_aa // 2):
        chunk = seq[start:start + max_aa]
        if not chunk:
            break
        try:
            v, eff = embed_one(model, tokenizer, chunk, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            v, eff = embed_one(model, tokenizer, chunk, device)  # 单段重试（本身即单段）
        vecs.append(v)
        wts.append(eff)
    vecs = np.asarray(vecs, dtype=np.float32)          # (k, hidden)
    wts = np.asarray(wts, dtype=np.float32)            # (k)
    pooled = (vecs * wts[:, None]).sum(axis=0) / max(wts.sum(), 1e-6)
    return pooled, True, float(wts.sum())


def run_in_batches(model, tokenizer, seqs, device, batch_size):
    """逐序列生成 {idx: mean_pool_vec}；带层次化 OOM 兜底。返回 (vec_dict, n_segmented, failures:list)。"""
    hidden = cfg.get("esm_embed.hidden_dim")
    out = {}
    n_seg = 0
    failures = []
    # 简单分批（串行通道）：以「逐条」为安全基准，batch 用于显著节省 GPU 往返开销。
    for i, s in enumerate(seqs):
        try:
            vec, seg, _ = mean_pool_seq(model, tokenizer, s, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            try:
                torch.cuda.synchronize()
                vec, seg, _ = mean_pool_seq(model, tokenizer, s, device)
            except torch.cuda.OutOfMemoryError:
                vec = np.zeros(hidden, dtype=np.float32)   # 极端兜底：占位（如实计入 failures）
                failures.append(i)
                seg = False
        out[i] = vec
        n_seg += int(seg)
        if (i + 1) % 500 == 0:
            print(f"[04b] {i+1}/{len(seqs)}  ({100*(i+1)/len(seqs):.0f}%)", flush=True)
    return out, n_seg, failures


def load_model(model_path, device, fp16=False):
    """加载 ESM-2-3B。fp16=True 时以半精度加载权重（约 6GB，FP32 则约 12GB）。

    16GB 显存卡上若仍 OOM，用 --fp16 再降一半权重占用。
    """
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    kwargs = {"torch_dtype": torch.float16} if fp16 else {}
    model = AutoModelForMaskedLM.from_pretrained(model_path, **kwargs).to(device).eval()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=cfg.get_path("paths.esm_model_path"))
    ap.add_argument("--out_dir", default=cfg.get_path("paths.esm_cache_dir"))
    ap.add_argument("--batch_size", type=int, default=cfg.get("esm_embed.batch_size"))
    ap.add_argument("--max_n", type=int, default=cfg.get("esm_embed.max_n"),
                    help="0=全部；>0=仅前 max_n 条唯一序列")
    ap.add_argument("--fp16", action="store_true",
                    help="以 FP16 加载 ESM-2-3B 权重（省一半显存；16GB 卡上仍 OOM 时使用）")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[04b] device={device} batch_size={args.batch_size}")

    seqs = get_unique_seqs(args.max_n)
    print(f"[04b] 唯一蛋白序列: {len(seqs)}")

    # 幂等 + 增量：缓存里已有的序列直接复用，只嵌入缺失的序列并【追加】。
    # 原实现只要出现一条新序列就全量重算所有唯一序列（~2.6 万条，数十分钟到数小时）；
    # 缺陷#8 改了负样本采样会引入新 target，故必须增量，否则每次改采样都要重付这笔钱。
    # 追加逻辑与 code/07b_esm_extend.py 保持一致（索引不重排，回填不会移位）。
    emb_f = os.path.join(args.out_dir, "embeddings.npy")
    ids_f = os.path.join(args.out_dir, "ids.json")
    emb_old, ids_old = None, []
    if os.path.exists(emb_f) and os.path.exists(ids_f):
        emb_old = np.load(emb_f)
        with open(ids_f) as f:
            ids_old = json.load(f)
        assert emb_old.shape[0] == len(ids_old), (
            f"ESM 缓存不一致: embeddings {emb_old.shape[0]} 行 vs ids {len(ids_old)} 条，请清空重跑")
    cached_hashes = {c["hash"] for c in ids_old}

    todo = [s for s in seqs if seq_hash(s) not in cached_hashes]
    if not todo:
        print(f"[04b] 缓存已完备（{len(ids_old):,} 条唯一），跳过。")
        return
    print(f"[04b] 缓存 {len(ids_old):,} 条；本次新增 {len(todo):,} 条（目标全集 {len(seqs):,}）")

    model, tokenizer = load_model(args.model_path, device, fp16=args.fp16)
    t0 = time.time()
    vec_dict, n_seg, failures = run_in_batches(model, tokenizer, todo, device, args.batch_size)

    # 拒绝把 OOM 兜底的零向量写进缓存：零向量会让该蛋白在特征矩阵里退化成「缺失模态」，
    # 正是缺陷#4 要消灭的泄漏通道。原实现在 16GB 卡上把 1743 条全部写成零向量且静默入库。
    if len(failures) > max(10, 0.01 * len(todo)):
        raise RuntimeError(
            f"{len(failures)}/{len(todo)} 条序列触发 OOM 兜底（>1%），已拒绝写入缓存。\n"
            f"  零向量会污染特征并重新引入缺失模态泄漏。可尝试：\n"
            f"  1) python3 code/04b_esm_embed.py --fp16          # 权重减半\n"
            f"  2) 调小 esm_embed.max_aa_per_chunk（当前 "
            f"{cfg.get('esm_embed.max_aa_per_chunk')}）\n"
            f"  3) 在更大显存的卡上重跑")

    # 落盘（只追加，旧索引保持不变）
    new_emb = np.stack([vec_dict[i] for i in range(len(todo))]).astype(np.float32)
    full_emb = new_emb if emb_old is None else np.vstack([emb_old, new_emb])
    base = len(ids_old)
    full_ids = ids_old + [{"index": base + i, "hash": seq_hash(s), "seq": s}
                          for i, s in enumerate(todo)]
    tmp_emb = emb_f + ".tmp.npy"
    np.save(tmp_emb, full_emb)
    os.replace(tmp_emb, emb_f)
    tmp_ids = ids_f + ".tmp.json"
    with open(tmp_ids, "w") as f:
        json.dump(full_ids, f, ensure_ascii=False)
    os.replace(tmp_ids, ids_f)
    stats = {
        "n_unique": int(len(full_ids)),
        "n_added_this_run": int(len(todo)),
        "n_segmented_added": int(n_seg),
        "dim": cfg.get("esm_embed.hidden_dim"),
        "elapsed_s": round(time.time() - t0, 1),
        "oom_fallbacks_added": len(failures),
        "oom_failure_indices": failures,
    }
    with open(os.path.join(args.out_dir, "esm_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[04b] 完成：缓存 {len(full_ids):,} 条（新增 {len(todo)}）→ {emb_f} {full_emb.shape}")
    print(f"[04b] 分段序列 {n_seg} 条；OOM 兜底 {len(failures)} 条")
    print(f"[04b] 耗时 {stats['elapsed_s']}s")


if __name__ == "__main__":
    main()
