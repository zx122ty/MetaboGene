#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 4 — Feature encoding (lightweight features, single-modality-first)

分子特征:  ECFP4 / Morgan fingerprint (位数/radius 见 config), RDKit 从 SMILES 计算。
蛋白特征:  dipeptide + AA composition + length (维度由 config 的 aa_alphabet 推导)。
缺失模态:  用全零向量占位（绝不让「有无结构」泄漏成分类信号）。

两个任务:
  task_binary   : 互作二分类 (pos=1, neg=0) —— 全部 pos+neg 样本
  task_direction: 方向三分类 (activation=2/inhibition=1/neutral=0) —— 仅正样本

按 config (features.seed / split_*) 固定随机划分，保存 features.npz / labels.npz / split.npz。

交付:
  data/features/features.npz  (X_binary, X_direction, has_smiles, has_seq)
  data/features/labels.npz    (y_binary, y_direction, direction_mask)
  data/features/split.npz     (binary_train/val/test idx, dir_train/val/test idx)
  data/features/feature_stats.json
参数统一来自 config.yaml (paths.* / features.*)。
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split

import config as C

cfg = C.load()

OUT = cfg.get_path("paths.features_dir")
os.makedirs(OUT, exist_ok=True)


# ---------------------------------------------------------------------------
# MolEncoder 策略层：fp（原 ECFP4）/ unimol2 / hybrid
# 开闭原则：fp 与原逻辑完全一致；unimol2/hybrid 为新增，不覆盖基线 features.npz。
# 本模块（04）只负责【编码与拼接】，维度规整/投影在 05 训练时做（--mol_proj）。
# ---------------------------------------------------------------------------
def _mol_fp_for_row(smiles):
    """单条 ECFP4 bits list；无效(含非字符串/NaN)返回 None。"""
    if not smiles or not isinstance(smiles, str):
        return None
    mol = Chem.MolFromSmiles(smiles.strip())
    if mol is None:
        return None
    return list(AllChem.GetMorganFingerprintAsBitVect(
        mol, cfg.get("features.fp_radius"), nBits=cfg.get("features.fp_bits")))


def encode_smiles_fp(smiles_list):
    """策略 fp：原 ECFP4。返回 (X_mol float32, has_mol mask)。"""
    fp_bits = cfg.get("features.fp_bits")
    fps, valid = [], []
    for smi in smiles_list:
        fp = _mol_fp_for_row(smi)
        if fp is not None:
            valid.append(True)
            fps.append(np.array(fp, dtype=np.float32))
        else:
            valid.append(False)
            fps.append(np.zeros(fp_bits, dtype=np.float32))
    print(f"[04-fp] SMILES 有效指纹: {sum(valid)}/{len(smiles_list)} ({100*sum(valid)/max(len(smiles_list),1):.1f}%)")
    return np.vstack(fps), np.array(valid)


def load_unimol_cache(cache_dir):
    """加载 04c 生成的 Uni-Mol2 缓存 → (cls_np (N,1536), smiles->index dict)。

    - 失败键（failed.json）与未覆盖键：has_mol=False，由调用方占零。
    - atomic_reprs/atom_counts 主要在 05 的 Cross-Attention 阶段使用，此处仅校验存在性。
    """
    cls_f = os.path.join(cache_dir, "cls_reprs.npy")
    ids_f = os.path.join(cache_dir, "ids.json")
    fail_f = os.path.join(cache_dir, "failed.json")
    if not (os.path.exists(cls_f) and os.path.exists(ids_f)):
        raise FileNotFoundError(
            f"未找到 Uni-Mol2 缓存：{cache_dir}/cls_reprs.npy。请先运行 code/04c_unimol2_embed.py")
    cls = np.load(cls_f).astype(np.float32)         # (N_unique, 1536)
    with open(ids_f) as f:
        ids = json.load(f)                          # [{index, smiles}, ...]
    smi_to_idx = {rec["smiles"]: rec["index"] for rec in ids}
    failed = []
    if os.path.exists(fail_f):
        with open(fail_f) as f:
            failed = set(json.load(f))
    return cls, smi_to_idx, failed


def _l2_normalize(X):
    """逐行 L2 归一化（对全零行保持不变）。用于把 Unimol 高范数向量与已归一化的 ESM 对齐。"""
    eps = cfg.get("features.l2_norm_eps")
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return np.divide(X, n, out=np.zeros_like(X), where=n > eps)


def encode_smiles_unimol2(smiles_list, cls_emb, smi_to_idx, failed, normalize=False):
    """策略 unimol2：读取 Uni-Mol2 CLS 缓存回填；未命中/失败 → 全零占位 + 掩码。

    normalize=True 时对命中行做 L2 归一化（与 ESM 特征同尺度，避免 Unimol 高范数主导梯度）。
    """
    dim = cls_emb.shape[1]
    rows, valid = [], []
    n_hit = 0
    for smi in smiles_list:
        if not smi or not isinstance(smi, str) or smi in failed:
            valid.append(False)
            rows.append(np.zeros(dim, dtype=np.float32))
            continue
        idx = smi_to_idx.get(smi)
        if idx is None:
            valid.append(False)
            rows.append(np.zeros(dim, dtype=np.float32))
        else:
            v = cls_emb[idx]
            if v.sum() == 0:           # 全零 → 视为失败占位
                valid.append(False)
                rows.append(np.zeros(dim, dtype=np.float32))
            else:
                n_hit += 1
                valid.append(True)
                rows.append(v)
    X = np.vstack(rows).astype(np.float32)
    if normalize and X.shape[0]:
        X[valid] = _l2_normalize(X[valid])
    print(f"[unimol2] 缓存命中: {n_hit}/{len(smiles_list)} "
          f"({100*n_hit/max(len(smiles_list),1):.1f}%)；未命中/失败 {len(smiles_list)-n_hit}"
          f"；L2归一化={normalize}")
    return X, np.array(valid)


def encode_mol(smiles_list, mol_type, cache_dir, normalize_mol=True):
    """分子编码分发（策略模式）。返回 (X_mol (N, mol_dim), has_mol mask)。

    normalize_mol：对 unimol2/hybrid 的 UniCLS 部分做 L2 归一化（默认开启，与 ESM 同尺度）。
    """
    if mol_type == "fp":
        return encode_smiles_fp(smiles_list)
    # 保留原逻辑
    elif mol_type == "unimol2":
        cls, smi_to_idx, failed = load_unimol_cache(cache_dir)
        return encode_smiles_unimol2(smiles_list, cls, smi_to_idx, failed, normalize=normalize_mol)
    elif mol_type == "hybrid":
        X_fp, h_fp = encode_smiles_fp(smiles_list)
        cls, smi_to_idx, failed = load_unimol_cache(cache_dir)
        X_um, h_um = encode_smiles_unimol2(smiles_list, cls, smi_to_idx, failed, normalize=normalize_mol)
        # 拼接：分子块 = [ECFP4, UniCLS]；任一命中则 has_mol=True
        X_mol = np.hstack([X_fp, X_um]).astype(np.float32)
        has_mol = np.logical_or(h_fp, h_um)
        # NaN 兜底
        X_mol = np.nan_to_num(X_mol)
        return X_mol, has_mol


def protein_feat(seq):
    """dipeptide(len(AA)^2) + AA comp(len(AA)) + length(1); None if no seq"""
    aa = cfg.get("features.aa_alphabet")
    min_len = cfg.get("features.dde_min_seq_len")
    if not seq or not isinstance(seq, str) or len(seq) < min_len:
        return None
    seq = seq.upper()
    comp = [seq.count(a) for a in aa]
    dipe = [seq.count(a + b) for a in aa for b in aa]
    feats = dipe + comp + [len(seq)]
    return feats


def dde_dim():
    """蛋白轻量特征维度 = len(AA)^2 + len(AA) + 1 (原 421)。"""
    n = len(cfg.get("features.aa_alphabet"))
    return n * n + n + 1


def _load_esm_cache(cache_dir):
    """加载 ESM 唯一序列缓存 → (emb_np[N,2560], seq->index dict)。"""
    ids_path = os.path.join(cache_dir, "ids.json")
    emb_path = os.path.join(cache_dir, "embeddings.npy")
    with open(ids_path) as f:
        ids = json.load(f)
    seq_to_idx = {rec["seq"]: rec["index"] for rec in ids}
    emb = np.load(emb_path)
    return emb, seq_to_idx


def esm_protein_feat(seq, esm_emb, esm_idx, l2_norm=True):
    """用 ESM-2-3B 缓存向量取代 DDE：命中返回向量；未命中/无序列 → None（全零占位由调用方处理）。"""
    eps = cfg.get("features.l2_norm_eps")
    if not seq or not isinstance(seq, str):
        return None
    s = seq.upper()
    idx = esm_idx.get(s)
    if idx is None:
        return None
    v = esm_emb[idx].astype(np.float32)
    if l2_norm:
        norm = np.linalg.norm(v) + eps
        v = v / norm
    return v


def encode_seq(seq_list, protein_type="dde", esm_cache=None, l2_norm=True):
    """蛋白质编码分发：`dde`=轻量二肽特征；`esm`=ESM 缓存向量。
    返回 (feat_matrix float32, has_seq_mask)。缺失/未命中 → 全零占位（保留掩码，防结构泄漏）。"""
    if protein_type == "esm":
        esm_emb, esm_idx = esm_cache
        dim = esm_emb.shape[1]
        valid, rows = [], []
        for s in seq_list:
            v = esm_protein_feat(s, esm_emb, esm_idx, l2_norm)
            if v is None:
                valid.append(False)
                rows.append(np.zeros(dim, dtype=np.float32))
            else:
                valid.append(True)
                rows.append(v)
        print(f"[04-esm] 蛋白特征有效(Esm cache hit): {sum(valid)}/{len(seq_list)} "
              f"({100*sum(valid)/max(len(seq_list), 1):.1f}%)")
        return np.vstack(rows).astype(np.float32), np.array(valid)
    # dde 默认（原逻辑，保留）
    dim = dde_dim()
    feats, valid = [], []
    for s in seq_list:
        f = protein_feat(s)
        if f is not None:
            valid.append(True)
            feats.append(np.array(f, dtype=np.float32))
        else:
            valid.append(False)
            feats.append(np.zeros(dim, dtype=np.float32))
    print(f"[04] 蛋白特征有效: {sum(valid)}/{len(seq_list)} ({100*sum(valid)/max(len(seq_list), 1):.1f}%)")
    return np.vstack(feats).astype(np.float32), np.array(valid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--protein_encoder_type", choices=["dde", "esm"], default="dde",
                    help="dde=轻量二肽特征；esm=ESM-2-3B 缓存向量(需先跑 04b_esm_embed.py)")
    ap.add_argument("--esm_cache_dir", default=cfg.get_path("paths.esm_cache_dir"),
                    help="ESM 唯一序列缓存目录（04b 输出）")
    ap.add_argument("--mol_encoder", choices=["fp", "unimol2", "hybrid"], default="fp",
                    help="fp=ECFP；unimol2=Uni-Mol2 CLS(需先跑 04c_unimol2_embed.py)；"
                         "hybrid=ECFP + UniCLS 拼接")
    ap.add_argument("--mol_cache_dir", default=cfg.get_path("paths.unimol2_cache_dir"),
                    help="Uni-Mol2 缓存目录（04c 输出）")
    ap.add_argument("--out_tag", default=None,
                    help="输出名后缀；缺省时按 features.out_tag_map / default_out_tag 决定")
    ap.add_argument("--no_normalize_mol", action="store_false", dest="normalize_mol",
                    default=cfg.get("features.normalize_mol_default"),
                    help="关闭 UniCLS 逐行 L2 归一化（默认值见 config features.normalize_mol_default）")
    args = ap.parse_args()
    protein_type = args.protein_encoder_type
    mol_type = args.mol_encoder
    print(f"[04] mol_encoder={mol_type} protein_encoder={protein_type} normalize_mol={args.normalize_mol}")

    pos = pd.read_parquet(cfg.get_path("paths.positives"))
    neg = pd.read_parquet(cfg.get_path("paths.negatives"))

    # 互作二分类：合并 pos + neg，y=label
    pos_y = pos["label"].astype(int).values
    neg_y = neg["label"].astype(int).values
    y_binary = np.concatenate([pos_y, neg_y])

    # 方向三分类（仅正样本）
    y_direction = pos["direction_label"].astype(int).values
    dir_valid = y_direction != -1  # 占位（全部有效）

    all_smiles = pd.concat([pos["smiles"], neg["smiles"]]).tolist()
    all_seq = pd.concat([pos["protein_seq"], neg["protein_seq"]]).tolist()

    # 按 mol_encoder 类型分发分子表征（策略模式；fp 为原逻辑）
    X_mol, has_smiles = encode_mol(all_smiles, mol_type, args.mol_cache_dir, normalize_mol=args.normalize_mol)
    mol_dim = X_mol.shape[1]

    # 按 encoder 类型分发蛋白表征
    if protein_type == "esm":
        if not os.path.exists(os.path.join(args.esm_cache_dir, "embeddings.npy")):
            raise FileNotFoundError(f"未找到 ESM 缓存：{args.esm_cache_dir}/embeddings.npy。请先运行 code/04b_esm_embed.py")
        esm_cache = _load_esm_cache(args.esm_cache_dir)
        X_seq, has_seq = encode_seq(all_seq, protein_type="esm", esm_cache=esm_cache, l2_norm=True)
    else:
        X_seq, has_seq = encode_seq(all_seq, protein_type="dde")
    seq_dim = X_seq.shape[1]

    # 特征维度
    n_mol = X_mol.shape[1]
    n_pos = len(pos)
    n_all = len(y_binary)
    seed = cfg.get("features.seed")
    test_size = cfg.get("features.split_test_size")
    second_half = cfg.get("features.split_second_half")

    # ================= 【缺陷#4】缺失模态剔除 + 1:1 配比 =================
    # 零填充行构成标签泄漏通道：正负样本的结构覆盖率不对称（正样本缺 7.49%，
    # 负样本缺 25.41%），单用「是否缺模态」这一个 bit 做预测 ROC-AUC 已达 0.5896。
    # 原实现的 has_smiles/has_seq 掩码只存不用，零填充行照样参与训练。
    # 修法：训练前直接剔除不完整行，再按少数类下采样恢复 1:1，并重新划分。
    is_pos = np.zeros(n_all, dtype=bool)
    is_pos[:n_pos] = True
    complete = has_smiles & has_seq
    stats_drop = {
        "n_raw_pos": int(n_pos), "n_raw_neg": int(n_all - n_pos),
        "pos_missing_smiles_rate": round(float(1 - has_smiles[is_pos].mean()), 4),
        "pos_missing_seq_rate": round(float(1 - has_seq[is_pos].mean()), 4),
        "neg_missing_smiles_rate": round(float(1 - has_smiles[~is_pos].mean()), 4),
        "neg_missing_seq_rate": round(float(1 - has_seq[~is_pos].mean()), 4),
        "pos_incomplete_rate": round(float(1 - complete[is_pos].mean()), 4),
        "neg_incomplete_rate": round(float(1 - complete[~is_pos].mean()), 4),
    }
    if cfg.get("features.drop_incomplete_modalities", True):
        pos_keep = np.where(complete & is_pos)[0]
        neg_keep = np.where(complete & ~is_pos)[0]
        stats_drop["pos_dropped_incomplete"] = int(n_pos - len(pos_keep))
        stats_drop["neg_dropped_incomplete"] = int((n_all - n_pos) - len(neg_keep))
        if cfg.get("features.rebalance_pos_neg", True):
            n_keep = min(len(pos_keep), len(neg_keep))
            rng = np.random.RandomState(seed)
            if len(pos_keep) > n_keep:
                pos_keep = np.sort(rng.choice(pos_keep, n_keep, replace=False))
            if len(neg_keep) > n_keep:
                neg_keep = np.sort(rng.choice(neg_keep, n_keep, replace=False))
            stats_drop["pos_dropped_rebalance"] = int(
                complete[is_pos].sum() - len(pos_keep))
            stats_drop["neg_dropped_rebalance"] = int(
                complete[~is_pos].sum() - len(neg_keep))
        keep_idx = np.concatenate([pos_keep, neg_keep])   # 正样本在前，索引约定不变
        keep_idx.sort()
        # 方向任务：仅正样本，且同样只保留完整模态（并与被选中的正样本行取交）
        pos_selected = np.intersect1d(np.where(is_pos)[0][complete[is_pos]],
                                      pos_keep, assume_unique=True)
    else:
        keep_idx = np.arange(n_all)
        pos_selected = np.where(is_pos)[0]
        print("[04] 警告: features.drop_incomplete_modalities=false，零填充行将参与训练"
              "（存在标签泄漏风险，仅供复现旧结果）")
    stats_drop["n_final"] = int(len(keep_idx))
    stats_drop["n_final_pos"] = int(is_pos[keep_idx].sum())
    stats_drop["n_final_neg"] = int((~is_pos[keep_idx]).sum())
    stats_drop["final_ratio_neg_per_pos"] = round(
        stats_drop["n_final_neg"] / max(stats_drop["n_final_pos"], 1), 4)
    print(f"[04] 缺失模态剔除: 正样本 {stats_drop['n_raw_pos']:,} -> "
          f"{stats_drop['n_final_pos']:,}  负样本 {stats_drop['n_raw_neg']:,} -> "
          f"{stats_drop['n_final_neg']:,}  最终 {stats_drop['n_final']:,} 行 "
          f"(1:{stats_drop['final_ratio_neg_per_pos']:.3f})")
    print(f"[04] 剔除前缺失率: 正 SMILES {stats_drop['pos_missing_smiles_rate']:.2%} / "
          f"序列 {stats_drop['pos_missing_seq_rate']:.2%}；负 SMILES "
          f"{stats_drop['neg_missing_smiles_rate']:.2%} / 序列 {stats_drop['neg_missing_seq_rate']:.2%}")

    # 先建方向矩阵（仅正样本，较小）再建二分类矩阵，避免大数组同时驻留多份副本
    X_direction = np.ascontiguousarray(
        np.hstack([X_mol[pos_selected], X_seq[pos_selected]]), dtype=np.float32)
    y_direction = y_direction[pos_selected]
    X_binary = np.ascontiguousarray(
        np.hstack([X_mol[keep_idx], X_seq[keep_idx]]), dtype=np.float32)
    y_binary = y_binary[keep_idx]
    has_smiles = has_smiles[keep_idx]
    has_seq = has_seq[keep_idx]
    del X_mol, X_seq

    # 划分（固定 SEED，80/10/10）——用过滤后的数组重新划分
    n = len(y_binary)
    idx = np.arange(n)
    tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=y_binary)
    va, te = train_test_split(te, test_size=second_half, random_state=seed, stratify=y_binary[te])
    binary_idx = {"train": tr, "val": va, "test": te}

    dtr, dte = train_test_split(np.arange(len(y_direction)), test_size=test_size,
                                random_state=seed, stratify=y_direction)
    dva, dte = train_test_split(dte, test_size=second_half, random_state=seed, stratify=y_direction[dte])
    dir_idx = {"train": dtr, "val": dva, "test": dte}

    # 输出文件名分离，避免覆盖 Baseline features.npz
    # 【缺陷#3】输出名由 (mol_encoder, protein_encoder) 二元组共同决定。原实现只按 mol_type
    # 取名，于是 `--protein_encoder_type esm`（README 原文命令）会静默把 2469 维基线
    # features.npz 覆盖成 4608 维 ESM 特征，而 hybrid_esm/unimol_esm 这类名字根本产不出来。
    tag_map = cfg.get("features.out_tag_map", {})
    default_tag = cfg.get("features.default_out_tag")
    combo = f"{mol_type}+{protein_type}"
    if args.out_tag:
        tag = args.out_tag
    else:
        tag = tag_map.get(combo)
        if tag is None:
            raise KeyError(
                f"features.out_tag_map 缺少组合 {combo!r}（现有键: {sorted(tag_map)}）；"
                f"请补 config 或显式传 --out_tag")
    # 防覆盖守卫：features.npz 是 (fp, dde) 基线专名，其它组合落到此名一律中止
    if tag == default_tag and combo != "fp+dde":
        raise ValueError(
            f"组合 {combo} 解析出的输出名 {tag!r} 会覆盖 (fp+dde) 基线特征，已中止；"
            f"请检查 features.out_tag_map 或显式传 --out_tag")
    print(f"[04] 输出文件名: {tag}  (mol={mol_type}, protein={protein_type})")
    # 行实体表：缺陷#4 引入剔除/下采样后，X_binary 的行序不再是「全部正样本 | 全部负样本」，
    # 下游 05c/13/14 原先靠 positives.parquet + negatives.parquet 重建 (met,tar) 的假设失效。
    # 故把与矩阵行严格对齐的实体 ID 一并落盘。
    met_all = np.concatenate([pos["met"].values, neg["met"].values]).astype("int64")
    tar_all = np.concatenate([pos["tar"].values, neg["tar"].values]).astype("int64")
    np.savez_compressed(
        os.path.join(OUT, tag),
        X_binary=X_binary, X_direction=X_direction,
        has_smiles=has_smiles, has_seq=has_seq,
        row_met=met_all[keep_idx], row_tar=tar_all[keep_idx],       # 与 X_binary 行对齐
        dir_met=met_all[pos_selected], dir_tar=tar_all[pos_selected],  # 与 X_direction 行对齐
    )
    np.savez_compressed(
        cfg.get_path("paths.labels_npz"),
        y_binary=y_binary, y_direction=y_direction,
        direction_mask=np.ones(len(y_direction), dtype=bool),
    )
    np.savez_compressed(
        cfg.get_path("paths.split_npz"),
        b_tr=binary_idx["train"], b_va=binary_idx["val"], b_te=binary_idx["test"],
        d_tr=dir_idx["train"], d_va=dir_idx["val"], d_te=dir_idx["test"],
    )
    # 每个特征文件另存一套配套的 labels/split（缺陷#4 连带）：
    # 不同编码器的可特征化集合行数可能不同（fp/hybrid 399,528 vs unimol2 399,020），
    # 共用全局那一份会让下游拿错索引（实测 05_unimol_dde_mp512 越界崩）。
    # 命名：labels_<特征文件词干>.npz / split_<特征文件词干>.npz，由 cfg.sidecar_npz 查找。
    stem = os.path.splitext(tag)[0]
    np.savez_compressed(
        os.path.join(OUT, f"labels_{stem}.npz"),
        y_binary=y_binary, y_direction=y_direction,
        direction_mask=np.ones(len(y_direction), dtype=bool),
    )
    np.savez_compressed(
        os.path.join(OUT, f"split_{stem}.npz"),
        b_tr=binary_idx["train"], b_va=binary_idx["val"], b_te=binary_idx["test"],
        d_tr=dir_idx["train"], d_va=dir_idx["val"], d_te=dir_idx["test"],
    )

    stats = {
        "mol_encoder": mol_type,
        "mol_dim": int(mol_dim),
        "protein_encoder": protein_type,
        "protein_dim": int(seq_dim),
        "n_binary": int(n),
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "smiles_cov_binary": round(float(has_smiles.mean()), 4),
        "seq_cov_binary": round(float(has_seq.mean()), 4),
        "X_binary_dim": int(X_binary.shape[1]),
        "y_binary_dist": {int(k): int(v) for k, v in zip(*np.unique(y_binary, return_counts=True))},
        "y_direction_dist": {int(k): int(v) for k, v in zip(*np.unique(y_direction, return_counts=True))},
        "split_sizes": {k: int(len(v)) for k, v in binary_idx.items()},
        "output_tag": tag,
        "modality_filter": stats_drop,   # 缺陷#4：剔除/配比统计（含按标签缺失率）
    }
    with open(cfg.get_path("paths.feature_stats"), "w") as f:
        json.dump(stats, f, indent=2)
    # 同样按 tag 另存一份：全局 feature_stats.json 会被最后一次运行覆盖，
    # 而各编码器的可特征化集合行数不同（fp/hybrid 399,528 vs unimol2 399,020），
    # 下游若想引用「规范特征文件」的统计就会读到错的（如 S4 补充表）。
    with open(os.path.join(OUT, f"feature_stats_{stem}.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[04] 已保存 {tag} / labels.npz / split.npz / feature_stats.json")
    print(f"[04] stats: {json.dumps(stats, indent=2)}")

    # 验收
    min_samples = cfg.get("features.accept_min_samples")
    min_dim = cfg.get("features.accept_min_feature_dim")
    min_test = cfg.get("features.accept_min_test")
    print("\n========== 验收验证 ==========")
    ok = (n >= min_samples and X_binary.shape[1] > min_dim and len(binary_idx["test"]) > min_test)
    print(f"样本数 {n:,} (>={min_samples:,}), 特征维 {X_binary.shape[1]} (>{min_dim}), "
          f"测试集 {len(binary_idx['test']):,} (>{min_test:,})")
    print(f"[04] 验收{'通过 ✓' if ok else '未通过 ✗'}")


if __name__ == "__main__":
    main()
