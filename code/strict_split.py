#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""严格 pair-disjoint 划分的共享实现（唯一真相）。

背景（prompt/26-10-16/03-SOTA基线补充.md §3.3）：
  05c（strict 复评）、13（不确定性 strict 模型）与新增的 SOTA 基线脚本原本
  各自现场重建 strict 划分。任何一处 RandomState 消费顺序不同，划分就会漂移，
  所有对比数字静默不可比。

本模块把 05c 的原始逻辑原样抽取为 build_strict_split()：
  * 正样本按唯一 (met, tar) pair 分组，RandomState(seed).shuffle 后 80/10/10；
  * 负样本行随机 shuffle 后按各 split 的正样本数 1:1 截取；
  * 返回 row-level 索引 {"b_tr","b_va","b_te"}，语义与 split_features_esm.npz 一致。

★ RNG 流复用（坑 P1b）：
  05c 建完划分后继续用同一个 rng 对象 shuffle 每个 epoch 的 batch。若本函数
  内部新建 RandomState，调用方的 rng 会停在流位置 0，batch 顺序改变 → 权重不同
  → A5（step5c 指标数值不变）静默失败。故 rng=None 时内部新建，传入 rng 时复用。

落盘：
  data/features/split_strict_<feature_stem>.npz    （默认 features_esm.npz）
  键名 b_tr / b_va / b_te。

用法：
  python3 code/strict_split.py                 # 生成并校验默认 ESM 划分
  python3 code/strict_split.py --feat features_esm.npz
"""
import argparse
import os

import numpy as np

import config as C

# 05c 原始实现复现出的规模（稿件 Table 3 / §2.5 口径，用于自检）
EXPECTED_SIZES = {
    "train": (317290, 158645),
    "val": (40144, 20072),
    "test": (42094, 21047),
}


def build_strict_split(row_met, row_tar, n_pos, seed=42, rng=None):
    """严格 pair-disjoint 划分。

    Parameters
    ----------
    row_met, row_tar : array-like (n_rows,)
        每行的 (PubChem CID, Entrez)；行序唯一真相来自 features_esm.npz。
    n_pos : int
        正样本行数；行序约定为 [正样本块 | 负样本块]。
    seed : int
        rng=None 时内部新建 RandomState(seed) 的种子。
    rng : np.random.RandomState or None
        传入时复用调用方 RNG（保持其流位置，05c 的 batch shuffle 才能逐位复现）。

    Returns
    -------
    dict with int64 arrays b_tr / b_va / b_te（已排序、无交集）。
    """
    row_met = np.asarray(row_met, dtype=np.int64)
    row_tar = np.asarray(row_tar, dtype=np.int64)
    n_pos = int(n_pos)
    rng = np.random.RandomState(seed) if rng is None else rng

    # ---- strict pair-level split over positive rows（与 05c 逐行一致）----
    pairs = list(zip(row_met[:n_pos], row_tar[:n_pos]))
    unique_pairs = {}
    for i, pr in enumerate(pairs):
        if pr not in unique_pairs:
            unique_pairs[pr] = []
        unique_pairs[pr].append(i)

    all_pairs = list(unique_pairs.keys())
    rng.shuffle(all_pairs)
    n_pairs = len(all_pairs)
    n_te = int(n_pairs * 0.10)
    n_va = int(n_pairs * 0.10)
    te_pairs = set(all_pairs[:n_te])
    va_pairs = set(all_pairs[n_te:n_te + n_va])

    pos_tr, pos_va, pos_te = [], [], []
    for pr in all_pairs:
        rows = unique_pairs[pr]
        if pr in te_pairs:
            pos_te.extend(rows)
        elif pr in va_pairs:
            pos_va.extend(rows)
        else:
            pos_tr.extend(rows)

    # ---- negatives: unique pairs, random 80/10/10, 1:1 per split ----
    neg_rows = list(range(n_pos, len(row_met)))
    rng.shuffle(neg_rows)
    n_neg_tr, n_neg_va, n_neg_te = len(pos_tr), len(pos_va), len(pos_te)
    neg_tr = neg_rows[:n_neg_tr]
    neg_va = neg_rows[n_neg_tr:n_neg_tr + n_neg_va]
    neg_te = neg_rows[n_neg_tr + n_neg_va:n_neg_tr + n_neg_va + n_neg_te]

    tr = np.array(sorted(pos_tr + neg_tr), dtype=np.int64)
    va = np.array(sorted(pos_va + neg_va), dtype=np.int64)
    te = np.array(sorted(pos_te + neg_te), dtype=np.int64)

    assert not (set(pos_te) & set(pos_tr))
    assert not (set(pos_va) & set(pos_tr))
    assert len(tr) + len(va) + len(te) <= len(row_met)
    return {"b_tr": tr, "b_va": va, "b_te": te}


def strict_split_path(features_dir, feat_name):
    """与 features_esm.npz 配套的 strict 划分落盘路径。"""
    stem = os.path.splitext(os.path.basename(feat_name))[0]
    return os.path.join(features_dir, f"split_strict_{stem}.npz")


def load_row_entities(features_dir, feat_name):
    """从特征 npz 读 row_met/row_tar/n_pos（缺陷#4 后的行序唯一真相）。"""
    d = np.load(os.path.join(features_dir, feat_name), allow_pickle=True)
    if "row_met" not in d or "row_tar" not in d:
        raise KeyError(f"{feat_name} 缺少 row_met/row_tar，无法构建 strict 划分")
    return (d["row_met"].astype(np.int64), d["row_tar"].astype(np.int64),
            int(len(d["dir_met"])))


def load_or_build(features_dir, feat_name, row_met=None, row_tar=None,
                  n_pos=None, seed=42, force=False):
    """读落盘 strict 划分；缺失或 force=True 时重建并落盘。"""
    path = strict_split_path(features_dir, feat_name)
    if os.path.exists(path) and not force:
        z = np.load(path)
        return {k: z[k] for k in ("b_tr", "b_va", "b_te")}
    if row_met is None:
        row_met, row_tar, n_pos = load_row_entities(features_dir, feat_name)
    sp = build_strict_split(row_met, row_tar, n_pos, seed=seed)
    np.savez(path, **sp)
    return sp


def _check_sizes(sp, n_pos):
    print(f"[strict] train {len(sp['b_tr']):,} (pos {int((sp['b_tr'] < n_pos).sum()):,}) / "
          f"val {len(sp['b_va']):,} (pos {int((sp['b_va'] < n_pos).sum()):,}) / "
          f"test {len(sp['b_te']):,} (pos {int((sp['b_te'] < n_pos).sum()):,})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default=None, help="特征文件名，默认 config training.feat_file_map['esm']")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = C.load()
    feat_name = args.feat or cfg.get("training.feat_file_map")["esm"]
    seed = args.seed if args.seed is not None else int(cfg.get("training.seed"))
    feat_dir = cfg.get_path("paths.features_dir")
    row_met, row_tar, n_pos = load_row_entities(feat_dir, feat_name)
    sp = load_or_build(feat_dir, feat_name, row_met, row_tar, n_pos,
                       seed=seed, force=args.force)

    _check_sizes(sp, n_pos)
    # 落盘自检：规模必须与 05c 原实现一致
    got = {
        "train": (len(sp["b_tr"]), int((sp["b_tr"] < n_pos).sum())),
        "val": (len(sp["b_va"]), int((sp["b_va"] < n_pos).sum())),
        "test": (len(sp["b_te"]), int((sp["b_te"] < n_pos).sum())),
    }
    for k, exp in EXPECTED_SIZES.items():
        if got[k] != exp:
            raise SystemExit(f"[strict] {k} 规模 {got[k]} != 预期 {exp}；划分逻辑已漂移")
    print(f"[strict] 落盘 -> {strict_split_path(feat_dir, feat_name)}")
    print("[strict] 规模自检通过 ✓")


if __name__ == "__main__":
    main()
