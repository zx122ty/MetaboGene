#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 1 — 数据治理与金标准子集构建
读取 merged_standardized_tree.parquet，过滤噪声，构建金标准正样本集。

交付物:
  data/gold/gold_positive.parquet   # 金标准正样本 (is_direct_interaction==True)
  data/gold/gold_stats.json         # 关键统计
参数统一来自 config.yaml (paths.* / gold_dataset.*)。
"""
import json
import os

import pandas as pd

import config as C

cfg = C.load()


def load():
    path = cfg.get_path("paths.source_parquet")
    if not os.path.exists(path):
        # 兼容项目内相对路径
        path = os.path.abspath(os.path.join(
            cfg.base_dir, "..", "..", "deepseekAPI/result/02-process/merged_standardized_tree.parquet"))
    print(f"[01] 读取 parquet: {path}")
    df = pd.read_parquet(path, columns=cfg.get("gold_dataset.keep_columns"))
    print(f"[01] 原始行数: {len(df):,}")
    return df


def filter_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    # 1. 剔除否定句
    neg = df["is_negated"].astype(str).str.contains("True", na=False)
    df = df[~neg]
    print(f"[01] 剔除 is_negated=True: {before:,} -> {len(df):,} ({len(df)/before*100:.1f}%)")

    # 2. 剔除缺失双ID
    before = len(df)
    has_cid = df["metabolite_pubchem_cid"].notna()
    has_entrez = df["gene_entrez_id"].notna()
    df = df[has_cid & has_entrez]
    print(f"[01] 剔除双ID缺失: {before:,} -> {len(df):,} ({len(df)/before*100:.1f}%)")

    # 3. 剔除极低置信度 (阈值来自 config)
    min_conf = cfg.get("gold_dataset.min_confidence")
    before = len(df)
    conf = df["confidence_score"].fillna(0)
    df = df[conf >= min_conf]
    print(f"[01] 剔除 conf<{min_conf}: {before:,} -> {len(df):,} ({len(df)/before*100:.1f}%)")

    # 4. 方向标签: 剔除 biphasic / NaN
    before = len(df)
    df = df[~df["direction"].isna()]
    if cfg.get("gold_dataset.drop_biphasic"):
        df = df[df["direction"] != "biphasic"]
    print(f"[01] 剔除 biphasic/NaN direction: {before:,} -> {len(df):,} ({len(df)/before*100:.1f}%)")
    return df


def label_direction(s: pd.Series) -> pd.Series:
    """direction -> 数字标签，映射表取自 config.yaml training.direction_label_map（唯一真相）。

    默认映射: activation=2, inhibition=1, neutral=0；其它/未知名（如 biphasic 已在
    过滤阶段剔除）一律落到 neutral=0，与历史行为一致。
    """
    m = cfg.direction_label_map()
    neutral = m.get("neutral", 0)
    return s.map(m).fillna(neutral).astype(int)


def main():
    out_dir = cfg.get_path("paths.gold_dir")
    os.makedirs(out_dir, exist_ok=True)

    df = load()
    df = filter_pipeline(df)
    print(f"[01] 过滤后总行数: {len(df):,}")

    # 金标准正样本 = 直接互作
    direct = df["is_direct_interaction"].astype(str).str.contains("True", na=False)
    gold = df[direct].copy()
    print(f"[01] 金标准正样本 (is_direct==True): {len(gold):,}")

    # 方向标签
    gold["direction_label"] = label_direction(gold["direction"])

    # 保存
    out = cfg.get_path("paths.gold_positive")
    gold.to_parquet(out, index=False)
    print(f"[01] 已保存: {out}")

    # 统计
    stats = {
        "raw_rows": len(df),
        "gold_positive_rows": int(len(gold)),
        "unique_pubchem_cid": int(gold["metabolite_pubchem_cid"].nunique()),
        "unique_entrez": int(gold["gene_entrez_id"].nunique()),
        "unique_pairs": int(gold[["metabolite_pubchem_cid", "gene_entrez_id"]].drop_duplicates().shape[0]),
        "direction_counts": gold["direction"].value_counts().to_dict(),
        "species_top5": gold["species"].value_counts().head(5).to_dict(),
        "interaction_type_counts": gold["interaction_type"].value_counts().head(8).to_dict(),
        "conf_mean": float(gold["confidence_score"].mean()),
    }
    stat_path = cfg.get_path("paths.gold_stats")
    with open(stat_path, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False, default=str)
    print(f"[01] 已保存统计: {stat_path}")

    # ---- 验收验证 ----
    min_pos = cfg.get("gold_dataset.accept_min_positives")
    min_frac = cfg.get("gold_dataset.accept_min_act_inh_frac")
    print("\n========== 验收验证 ==========")
    pos = int(len(gold))
    dir_nonzero = int((gold["direction_label"] != 0).sum())
    frac_ab = dir_nonzero / max(pos, 1)
    print(f"金标准正样本: {pos:,}  (验收: >={min_pos:,})")
    print(f"激活/抑制(非neutral)方向占比: {frac_ab*100:.1f}%  (验收: >={min_frac*100:.0f}%)")
    print(f"唯一代谢物: {stats['unique_pubchem_cid']:,}  唯一靶点: {stats['unique_entrez']:,}")
    print(f"方向分布: {gold['direction'].value_counts().to_dict()}")

    ok = (pos >= min_pos) and (frac_ab >= min_frac)
    print(f"\n[01] 验收{'通过 ✓' if ok else '未通过 ✗'}")
    if not ok:
        print("[01] 提示: 未达标准，需检查过滤规则是否过严。")


if __name__ == "__main__":
    main()
