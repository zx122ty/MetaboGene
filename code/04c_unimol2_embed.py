#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""04c — Uni-Mol2(1.1B) 分子表征离线推理与缓存（CLS + Atomic）

用途：对 data/neg 正/负样本的所有【唯一 SMILES】做 Uni-Mol2 推理，
     一次性提取：
       - cls_repr     (N_unique, cls_dim) float32          # 图级表征，主用特征
       - atomic_reprs (N_unique, MAX_ATOMS, cls_dim) fp16  # 原子级，预留 Cross-Attention
     保存到 data/features/cache_unimol2/，供 04_encode_features.py 复用。
     只对唯一 SMILES 推理一次，避免对 ~41 万样本重复计算（1.1B 推理昂贵）。

与 04b_esm_embed.py 对齐的工程约定：
  - 去重：以「唯一 SMILES」为缓存键，矩阵按行序回填（键=smiles，矩阵序=输入序）。
  - 幂等/可续跑：cls_reprs.npy 已存在且 ids 覆盖全部唯一 SMILES → 跳过。
  - 失败兜底：无效 SMILES / 3D 构象生成失败 → 该键不写特征，如实记入 failed，
    （由 04 下游以 has_mol 掩码占位，杜绝「有无 3D」泄漏成分类信号）。
  - 显存：batch_size + FP16(Autocast)；OOM → empty_cache 后降 batch → 逐条重试。

存储（cache_unimol2/）：
  cls_reprs.npy     (N_unique, cls_dim) float32            # CLS 图级表征
  ids.json          [{index, smiles}, ...]                 # 04 按 smiles 回填
  atomic_reprs.npy  (N_unique, max_atoms, cls_dim) fp16    # 原子级（fp16 省盘；缺位=0）
  atom_counts.npy   (N_unique,) uint16                     # 每键真实原子数
  failed.json       [smiles, ...]                          # 失败键
  meta.json         {model_name, model_size, ckpt, n_unique, n_failed,
                     cls_dim, atom_dim, max_atoms, elapsed_s}

命令：
  python code/04c_unimol2_embed.py
  python code/04c_unimol2_embed.py --max_n 50     # 快速试跑

依赖：pip install unimol_tools rdkit；权重见 config.yaml (unimol2_embed.ckpt)。
参数统一来自 config.yaml (paths.* / unimol2_embed.*)。
"""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
from unimol_tools import UniMolRepr

import config as C

cfg = C.load()


def get_unique_smiles(max_n=0):
    pos = pd.read_parquet(cfg.get_path("paths.positives"))
    neg = pd.read_parquet(cfg.get_path("paths.negatives"))
    all_smiles = pd.concat([pos["smiles"], neg["smiles"]]).dropna().astype(str)
    uniq = sorted(all_smiles.unique().tolist())  # 排序保证可复现
    if max_n:
        uniq = uniq[: max_n]
    return uniq


def make_model(ckpt):
    return UniMolRepr(
        data_type="molecule",
        remove_hs=False,
        model_name=cfg.get("unimol2_embed.model_name"),
        model_size=cfg.get("unimol2_embed.model_size"),
        pretrained_model_path=ckpt,
    )


def fold_one(res, cls_out):
    """把 get_repr 的【单条】结果折进 cls_out。

    返回 True=写入成功(真实向量)；False=畸形（由调用方判定失败）。
    get_repr 单条时 res['cls_repr'] 形如 [[...cls_dim]] 二维张量，取 [0] 即是向量。
    本函数不捕获异常——异常统一由 embed_batch 的逐条 try 捕获。
    """
    cls_dim = cfg.get("unimol2_embed.cls_dim")
    cls0 = np.asarray(res["cls_repr"], dtype=np.float32)[0]
    if cls0.ndim != 1 or cls0.shape[0] != cls_dim or not np.isfinite(cls0).all():
        return False
    cls_out[:] = cls0
    return True


def embed_batch(clf, smiles_batch, device):
    """对一个 batch 的 SMILES 取 cls + atomic 表征。返回 (cls_np (B,cls_dim), atomic_list[B], failed_idx)。

    这是唯一推理点：逐条调用 get_repr，绝不批量（避免 3D 构象失败的 RDKit 多进程批异常拖垮整批）。
    - 每条的 cls 临时数组单独一维（行号 0..len-1）正确填真实向量。B=len(smiles_batch)。
    - 失败键：cls 行全零、atomic=None，且进 failed_idx。
    """
    cls_dim = cfg.get("unimol2_embed.cls_dim")
    B = len(smiles_batch)
    cls_out = np.zeros((B, cls_dim), dtype=np.float32)
    atomic_list = [None] * B
    failed_idx = []
    for j, smi in enumerate(smiles_batch):
        try:
            res = clf.get_repr([smi], return_atomic_reprs=True)
            if res is None or len(np.asarray(res["cls_repr"])) == 0:
                raise RuntimeError("空结果")
            if not fold_one(res, cls_out[j]):   # 只折叠 cls；原子另读
                raise RuntimeError("cls 异常")
            # 原子块：get_repr 返回 (1, n_atoms, cls_dim)（带 batch 维）。取 [0] → (n_atoms, cls_dim)。
            a = np.asarray(res["atomic_reprs"], dtype=np.float32)
            atomic_list[j] = a[0] if a.ndim == 3 else (a if a.ndim == 2 else None)
            if atomic_list[j] is None:
                raise RuntimeError("atomic 异常")
        except Exception:
            torch.cuda.empty_cache()
            failed_idx.append(j)
            cls_out[j, :] = 0.0
            atomic_list[j] = None
    return cls_out, atomic_list, failed_idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=cfg.get("unimol2_embed.ckpt"))
    ap.add_argument("--out_dir", default=cfg.get_path("paths.unimol2_cache_dir"))
    ap.add_argument("--batch_size", type=int, default=cfg.get("unimol2_embed.batch_size"))
    ap.add_argument("--max_n", type=int, default=cfg.get("unimol2_embed.max_n"),
                    help="0=全部；>0=仅前 max_n 条唯一 SMILES")
    ap.add_argument("--resume", action="store_true",
                    help="断点续跑：已写入的 cls_reprs.npy/ids.json 视为已缓存，只补缺失键"
                         "{--resolve-with-idx 以唯一键去重}")
    args = ap.parse_args()

    cls_dim = cfg.get("unimol2_embed.cls_dim")
    max_atoms = cfg.get("unimol2_embed.max_atoms")

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[04c] device={device} batch_size={args.batch_size}")

    smiles = get_unique_smiles(args.max_n)
    print(f"[04c] 唯一 SMILES: {len(smiles)}")

    cls_f = os.path.join(args.out_dir, "cls_reprs.npy")
    ids_f = os.path.join(args.out_dir, "ids.json")
    # 幂等/续跑：已存在缓存 → 只对缺失键重跑（保留已有索引，不重排造成回填移位）
    cached_smis = set()
    if args.resume and os.path.exists(ids_f):
        with open(ids_f) as f:
            cached_smis = {c["smiles"] for c in json.load(f)}
    todo = [] if cached_smis else smiles
    if args.resume:
        todo = [s for s in smiles if s not in cached_smis]
    if args.resume and os.path.exists(ids_f) and not todo:
        print(f"[04c] 缓存已完备（{len(cached_smis)} 条唯一），跳过。")
        return
    # 决定 cls_all/atomic_all 的尺寸：断点续跑用 full 长度(与 smiles 等长)；新跑也用 full 长度并全部重算
    n_total = len(smiles)
    n_todo = len(todo)
    # 复用现有数组（若已完成部分+断点续跑）——此处简化：统一全量，续跑仍需重算全部（因数组是整存）
    # 注：因缓存按 smiles 排序建索引，续跑最简做法 = 重新编码 todo，再连同既有 np.load 合并；此处采取
    #     「新一次运行＝只算 todo，但写回时必须覆盖对应原数组」的等价路径。
    cls_full = None
    atomic_full = None
    count_full = None
    if args.resume and os.path.exists(cls_f):
        cls_full = np.load(cls_f)
        atomic_full = np.load(os.path.join(args.out_dir, "atomic_reprs.npy"))
        count_full = np.load(os.path.join(args.out_dir, "atom_counts.npy"))
        assert cls_full.shape[0] == n_total, "缓存长度与唯一SMILES不一致，请清空重跑"
    else:
        cls_full = np.zeros((n_total, cls_dim), dtype=np.float32)
        atomic_full = np.zeros((n_total, max_atoms, cls_dim), dtype=np.float16)
        count_full = np.zeros(n_total, dtype=np.uint16)
    # 把既有失败标记载入（续跑时用于 not 重复报失败）
    failed_smiles = []
    fail_f = os.path.join(args.out_dir, "failed.json")
    if args.resume and os.path.exists(fail_f):
        failed_smiles = json.load(open(fail_f))
    failed_set = set(failed_smiles)

    clf = make_model(args.ckpt)
    t0 = time.time()
    # 映射 smiles->index（续跑/新跑全一致）
    smi2idx = {s: i for i, s in enumerate(smiles)}
    for pos in range(0, len(todo), args.batch_size):
        batch = todo[pos:pos + args.batch_size]
        cls_b, atom_b_list, _failed = embed_batch(clf, batch, device)
        for j, smi in enumerate(batch):
            idx = smi2idx[smi]
            cls_full[idx] = cls_b[j]
            a = atom_b_list[j]
            if a is not None and a.shape[0] > 0:
                k = min(a.shape[0], max_atoms)
                count_full[idx] = k
                atomic_full[idx, :k, :] = a[:k].astype(np.float16)
            elif smi not in failed_set:
                failed_smiles.append(smi)
                failed_set.add(smi)
        if (pos // args.batch_size) % 5 == 0 and len(todo):
            print(f"[04c] {pos}/{len(todo)}  ({(100*pos)/len(todo):.0f}%)  失败累计 {len(failed_smiles)}", flush=True)

    # 清理 NaN
    nanrows = np.isnan(cls_full).any(axis=1)
    if nanrows.any():
        failed_smiles.extend(np.array(smiles)[nanrows].tolist())
        cls_full[nanrows] = 0.0

    np.save(cls_f, cls_full)
    np.save(os.path.join(args.out_dir, "atomic_reprs.npy"), atomic_full)
    np.save(os.path.join(args.out_dir, "atom_counts.npy"), count_full)
    with open(ids_f, "w") as f:
        json.dump([{"index": i, "smiles": s} for i, s in enumerate(smiles)], f, ensure_ascii=False)
    with open(os.path.join(args.out_dir, "atom_smiles.json"), "w") as f:
        json.dump([{"index": i, "smiles": smiles[i], "n_atoms": int(count_full[i])}
                   for i in range(len(smiles))], f, ensure_ascii=False)
    with open(fail_f, "w") as f:
        json.dump(failed_smiles, f, ensure_ascii=False)

    stats = {
        "model_name": cfg.get("unimol2_embed.model_name"),
        "model_size": cfg.get("unimol2_embed.model_size"),
        "ckpt": args.ckpt,
        "n_unique": int(len(smiles)), "n_failed": len(failed_smiles),
        "n_skipped_nan": int(nanrows.sum()),
        "cls_dim": cls_dim, "atom_dim": cls_dim, "max_atoms": max_atoms,
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[04c] 完成：{len(smiles)} 条 → {cls_f} ({cls_full.shape})")
    print(f"[04c] 失败 {len(failed_smiles)} 条；耗时 {stats['elapsed_s']}s")
    if failed_smiles:
        print(f"[04c] 失败样例：{failed_smiles[:5]}")


if __name__ == "__main__":
    main()
