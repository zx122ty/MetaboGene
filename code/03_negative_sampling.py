#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 3 — 负样本采样（语义驱动 + 随机）

策略（反制审稿攻击点 4「负样本缺失」）：
  * 正样本 = gold（direct 互作，唯一对）。
  * 候选池【只保留从未在源表任何互作行出现过的 pair】（缺陷#8）：
    原实现允许「出现在源表、但 is_direct_interaction != True」的间接互作对被抽作负样本，
    这些对生物学上未必「无互作」，只是「非直接」，模型可能学到的是文献表述层面的
    「直接 vs 间接」措辞差异而非结合与否。现显式排除源表全部 53 万唯一 pair。
  * 候选池进一步限定在【可特征化宇宙】：met 有 PubChem SMILES、tar 的 UniProt 在本地
    FASTA 中命中。只评估能取到特征的 pair，也让下游 Step4 的缺失模态剔除几乎只作用于正样本。
  * 难负样本（semantic hard）：在「同 disease_level1 但从未构成正样本」的 (pubchem_cid, entrez)
    对中采样 —— 这些「看起来相关但从未被报道」，模型无法靠统计特征作弊。
  * 易负样本（easy random）：全局随机 (pubchem_cid, entrez) 对，排除已知正样本。
  * 混合比见 config.yaml (negative_sampling.hard_ratio)；采样倍数见 oversample
    （Step4 会剔除缺失模态行并按少数类下采样，故此处先多采 oversample 倍）。
  * 分布对齐检查（缺陷#7）：原用 KS 检验比较「各物种占比向量」，用法错误且注释方向写反。
    现改为对两个分类分布做【卡方检验 + 总变差距离(TVD)】，并如实报告差异。

交付物：
  data/neg/positives.parquet   # 正样本（含 direction_label, smiles, protein_seq）
  data/neg/negatives.parquet   # 负样本（label=0）
  data/neg/neg_stats.json      # 正负比、难/易、物种卡方/TVD、审计列
参数统一来自 config.yaml (paths.* / negative_sampling.*)。
"""
import gzip
import json
import os
import random

import numpy as np
import pandas as pd
from scipy import stats

import config as C

cfg = C.load()


def load_gold():
    gold = pd.read_parquet(cfg.get_path("paths.gold_with_uniprot"))
    gold = gold.drop_duplicates(["metabolite_pubchem_cid", "gene_entrez_id"])
    gold["met"] = gold["metabolite_pubchem_cid"].astype("int64")
    gold["tar"] = gold["gene_entrez_id"].astype("int64")
    # 从 gold_with_struct 取 smiles / protein_seq
    struct = pd.read_parquet(cfg.get_path("paths.gold_with_struct"))[
        ["metabolite_pubchem_cid", "gene_entrez_id", "smiles", "protein_seq"]]
    struct["met"] = struct["metabolite_pubchem_cid"].astype("int64")
    struct["tar"] = struct["gene_entrez_id"].astype("int64")
    gold = gold.merge(struct[["met", "tar", "smiles", "protein_seq"]],
                      on=["met", "tar"], how="left")
    # 从源表取 metabolite_chemical_class / species（去重；只取 gold 缺失的，避免冲突）
    src = pd.read_parquet(cfg.get_path("paths.source_parquet"),
                          columns=["metabolite_pubchem_cid", "gene_entrez_id",
                                   "species", "metabolite_chemical_class"])
    src = src.dropna(subset=["metabolite_pubchem_cid", "gene_entrez_id"])
    src["met"] = src["metabolite_pubchem_cid"].astype("int64")
    src["tar"] = src["gene_entrez_id"].astype("int64")
    src = src.drop_duplicates(["met", "tar"])
    gold = gold.drop(columns=["species"], errors="ignore")
    gold = gold.merge(src[["met", "tar", "species", "metabolite_chemical_class"]],
                      on=["met", "tar"], how="left")
    return gold


def load_reported_pairs():
    """源表中出现过的【全部】互作 pair（不论直接/间接），供候选池排除（缺陷#8）。

    返回 (all_pairs, indirect_only_pairs)：
      all_pairs     —— 源表任意互作行的唯一 pair，负样本一律不得落入；
      indirect_only —— 只以非直接互作出现（从未被标为 direct）的 pair，
                       用于审计「原方案会拿它们当负样本」的规模。
    """
    df = pd.read_parquet(cfg.get_path("paths.source_parquet"),
                         columns=["metabolite_pubchem_cid", "gene_entrez_id",
                                  "is_direct_interaction"])
    df = df.dropna(subset=["metabolite_pubchem_cid", "gene_entrez_id"])
    m = df["metabolite_pubchem_cid"].astype("int64")
    t = df["gene_entrez_id"].astype("int64")
    direct = df["is_direct_interaction"].astype(str).str.contains("True", na=False)
    all_pairs = set(zip(m, t))
    indirect_only = all_pairs - set(zip(m[direct], t[direct]))
    return all_pairs, indirect_only


def load_fasta_accessions():
    """本地 UniProt FASTA 的 accession 集合（只读 header，不存序列，省内存）。"""
    accs = set()
    with gzip.open(cfg.get_path("paths.uniprot_fasta"), "rt") as f:
        for line in f:
            if line.startswith(">"):
                parts = line[1:].split("|")
                if len(parts) > 1:
                    accs.add(parts[1])
    return accs


def featurizable_entities():
    """可特征化宇宙：met 有 PubChem SMILES；tar 的 UniProt 在本地 FASTA 中命中。

    只评估能取到特征的 pair，既符合科学口径，也让 Step4 的缺失模态剔除
    几乎只作用于正样本（避免负样本被大规模剔掉后配比失衡）。
    """
    with open(cfg.get_path("paths.pubchem_smiles_cache")) as f:
        smiles = json.load(f)
    ok_mets = {int(c) for c, s in smiles.items() if s}

    uid = pd.read_parquet(cfg.get_path("paths.source_parquet"),
                          columns=["gene_entrez_id", "uniprot_id"])
    uid = uid.dropna(subset=["gene_entrez_id", "uniprot_id"]).drop_duplicates("gene_entrez_id")
    fasta_accs = load_fasta_accessions()
    uid = uid[uid["uniprot_id"].isin(fasta_accs)]
    ok_tars = set(uid["gene_entrez_id"].astype("int64"))
    print(f"[03] 可特征化实体: met {len(ok_mets):,} / tar {len(ok_tars):,} "
          f"(FASTA accession {len(fasta_accs):,})")
    return ok_mets, ok_tars


def load_candidates(ok_mets=None, ok_tars=None):
    cols = ["metabolite_pubchem_cid", "gene_entrez_id", "disease_level1",
            "metabolite_chemical_class", "species"]
    df = pd.read_parquet(cfg.get_path("paths.source_parquet"), columns=cols)
    df = df.dropna(subset=["metabolite_pubchem_cid", "gene_entrez_id"])
    df["met"] = df["metabolite_pubchem_cid"].astype("int64")
    df["tar"] = df["gene_entrez_id"].astype("int64")
    # 限定可特征化宇宙（仅影响采样池；元数据回填仍用全表，见 fill_map）
    if ok_mets is not None:
        df = df[df["met"].isin(ok_mets)]
    if ok_tars is not None:
        df = df[df["tar"].isin(ok_tars)]
    return df


def sample_negatives(cand, seen, n_needed):
    """难(同 disease 未报道) + 易(全局随机)，返回 DataFrame。

    `seen` = 必须回避的 pair 集合（= 源表全部互作 pair ∪ 已知正样本，缺陷#8）。
    """
    seed = cfg.get("negative_sampling.seed")
    hard_ratio = cfg.get("negative_sampling.hard_ratio")
    rng = random.Random(seed)
    n_hard = int(n_needed * hard_ratio)
    n_easy = n_needed - n_hard

    # 同 disease 分组: disease -> (met list, tar list)
    by_dis = {}
    for _, row in cand.dropna(subset=["disease_level1"]).iterrows():
        by_dis.setdefault(row["disease_level1"], [[], []])
        by_dis[row["disease_level1"]][0].append(row["met"])
        by_dis[row["disease_level1"]][1].append(row["tar"])
    groups = [g for g in by_dis.values() if len(g[0]) >= 2 and len(g[1]) >= 2]
    rng.shuffle(groups)

    seen = set(seen)
    hard = []
    gi = 0
    attempts = 0
    while len(hard) < n_hard and attempts < n_hard * 60:
        attempts += 1
        if not groups:
            break
        g = groups[gi % len(groups)]; gi += 1
        m = rng.choice(g[0]); t = rng.choice(g[1])
        if (m, t) in seen:
            continue
        seen.add((m, t))
        hard.append((m, t, "hard"))

    all_m = list(cand["met"].unique()); all_t = list(cand["tar"].unique())
    easy = []
    attempts = 0
    while len(easy) < n_easy and attempts < n_easy * 60:
        attempts += 1
        m = rng.choice(all_m); t = rng.choice(all_t)
        if (m, t) in seen:
            continue
        seen.add((m, t))
        easy.append((m, t, "easy"))

    neg = pd.DataFrame(hard + easy, columns=["met", "tar", "neg_type"])
    return neg


def dist_diff(pos_counts, neg_counts):
    """两个分类分布的差异度量（缺陷#7）。

    原实现用 stats.ks_2samp(各物种占比向量) —— 两个输入不是样本而是两组离散占比，
    检验的是「排序后占比的经验分布」，与业务问题无关，且注释方向写反。
    现改为：卡方检验（对原始计数）+ 总变差距离 TVD = 0.5 * Σ|p_i - n_i|。
    TVD ∈ [0,1]，0 = 分布完全一致，1 = 完全不相交；卡方 p 越小 = 差异越显著。
    """
    cats = sorted(set(pos_counts.index) | set(neg_counts.index))
    p = np.array([pos_counts.get(c, 0) for c in cats], dtype=float)
    n = np.array([neg_counts.get(c, 0) for c in cats], dtype=float)
    chi2 = pval = float("nan")
    dof = 0
    if p.sum() > 0 and n.sum() > 0:
        chi2, pval, dof, _ = stats.chi2_contingency(np.vstack([p, n]))
    ps = p / p.sum() if p.sum() else p
    ns = n / n.sum() if n.sum() else n
    tvd = 0.5 * float(np.abs(ps - ns).sum())
    return float(chi2), float(pval), int(dof), tvd


def target_species_map():
    """tar -> 该靶点在源表中最常出现的物种（负样本物种回填用）。

    原实现用 pair 级回填，而采样出的 pair 通常不是源表的现成行，于是负样本 86.8%
    的 species 落到 NaN —— 「物种分布对齐」检验实际是在比较正样本分布 vs 一个几乎
    全 NA 的分布，毫无意义（这也是 species_ks_p=0.0 的真实来源之一）。
    改为按靶点回填后覆盖率接近 100%，检验才有解释力。
    """
    src = pd.read_parquet(cfg.get_path("paths.source_parquet"),
                          columns=["gene_entrez_id", "species"]).dropna()
    src["tar"] = src["gene_entrez_id"].astype("int64")
    g = src.groupby(["tar", "species"]).size().reset_index(name="n")
    g = g.sort_values("n", ascending=False).drop_duplicates("tar")
    return dict(zip(g["tar"], g["species"]))


def main():
    out_dir = cfg.get_path("paths.neg_dir")
    os.makedirs(out_dir, exist_ok=True)

    gold = load_gold()
    known = set(zip(gold["met"], gold["tar"]))

    # ---- 缺陷#8：候选池排除源表全部互作 pair ----
    exclude_reported = cfg.get("negative_sampling.exclude_reported_pairs", True)
    reported, indirect_only = load_reported_pairs()
    seen = (set(reported) | known) if exclude_reported else set(known)
    print(f"[03] 源表互作 pair: {len(reported):,}（其中仅以非直接互作出现: {len(indirect_only):,}）"
          f"  负样本排除集: {len(seen):,}  排除开关={exclude_reported}")

    # ---- 限定可特征化宇宙 ----
    if cfg.get("negative_sampling.restrict_to_featurizable", True):
        ok_mets, ok_tars = featurizable_entities()
        cand = load_candidates(ok_mets, ok_tars)
    else:
        cand = load_candidates()
    print(f"[03] 正样本唯一对: {len(gold):,}  候选表: {len(cand):,} 行")

    # ---- 采样（oversample 补偿 Step4 的缺失模态剔除）----
    ratio_cfg = cfg.get("negative_sampling.pos_neg_ratio")
    oversample = cfg.get("negative_sampling.oversample", 1.0)
    n_needed = int(len(gold) * ratio_cfg * oversample)
    print(f"[03] 采样目标: {n_needed:,} = 正样本 {len(gold):,} × 比例 {ratio_cfg} × oversample {oversample}")
    neg = sample_negatives(cand, seen, n_needed)
    print(f"[03] 负样本: hard {int(neg['neg_type'].eq('hard').sum()):,} + easy "
          f"{int(neg['neg_type'].eq('easy').sum()):,} = {len(neg):,}")

    # ---- 正样本输出（含方向标签、结构/序列）----
    pos = gold[["met", "tar", "uniprot_id", "smiles", "protein_seq",
                "direction_label", "species", "metabolite_chemical_class"]].copy()
    pos.columns = ["met", "tar", "uniprot_id", "smiles", "protein_seq",
                   "direction_label", "species", "chem_class"]
    pos["label"] = 1
    pos["neg_type"] = "positive"

    # ---- 负样本输出：回填 species / chem_class（用于分布对齐检验）----
    # pair 级回填覆盖率很低（采样出的 pair 通常不是源表现成行），故再用 tar 级兜底
    fill_map = cand[["met", "tar", "species", "metabolite_chemical_class"]].drop_duplicates(["met", "tar"])
    neg = neg.merge(fill_map, on=["met", "tar"], how="left")
    tar_sp = target_species_map()
    neg["species"] = neg["species"].fillna(neg["tar"].map(tar_sp))
    neg["label"] = 0
    neg["direction_label"] = np.nan
    neg["uniprot_id"] = np.nan
    neg["smiles"] = np.nan
    neg["protein_seq"] = np.nan
    neg = neg.rename(columns={"species": "species", "metabolite_chemical_class": "chem_class"})

    out_pos = cfg.get_path("paths.positives")
    out_neg = cfg.get_path("paths.negatives")
    pos.to_parquet(out_pos, index=False)
    neg.to_parquet(out_neg, index=False)

    # ---- 分布对齐检验（缺陷#7）：卡方 + 总变差距离，取代误用的 KS ----
    pos_sp = pos["species"].fillna("NA").value_counts()
    neg_sp = neg["species"].fillna("NA").value_counts()
    chi2, chi2_p, dof, tvd = dist_diff(pos_sp, neg_sp)
    print(f"[03] 物种分布对比: chi2={chi2:,.1f} (dof={dof}, p={chi2_p:.3g})  TVD={tvd:.4f}"
          f"  (正样本缺失率 {pos['species'].isna().mean():.2%}, 负样本 {neg['species'].isna().mean():.2%})")
    print("[03] 注意: TVD 越大 / p 越小 = 正负物种分布差异【越显著】；"
          "差异大意味着需在消融里验证模型未把物种当捷径。")

    # ---- 缺陷#8 审计：负样本是否曾被源表报道过 ----
    neg_pairs = set(zip(neg["met"], neg["tar"]))
    n_prev_reported = len(neg_pairs & reported)

    stats_out = {
        "n_pos": int(len(pos)),
        "n_neg": int(len(neg)),
        "ratio": round(len(neg) / len(pos), 3),
        "n_hard": int(neg["neg_type"].eq("hard").sum()),
        "n_easy": int(neg["neg_type"].eq("easy").sum()),
        "oversample": oversample,
        "exclude_reported_pairs": bool(exclude_reported),
        "n_reported_pairs_excluded": int(len(reported)),
        "n_indirect_only_pairs_excluded": int(len(indirect_only)),
        "neg_previously_reported_frac": round(n_prev_reported / max(len(neg_pairs), 1), 6),
        "species_chi2": round(chi2, 2),
        "species_chi2_p": float(chi2_p),
        "species_chi2_dof": int(dof),
        "species_tvd": round(tvd, 4),
        "pos_species_missing_rate": round(float(pos["species"].isna().mean()), 4),
        "neg_species_missing_rate": round(float(neg["species"].isna().mean()), 4),
        "pos_direction_dist": pos["direction_label"].value_counts().to_dict(),
        "note": ("物种分布用卡方+TVD 比较（原 KS 用法错误且注释方向写反）；"
                 "TVD 越大 = 正负分布差异越显著。负样本已排除源表全部互作 pair（缺陷#8）。"),
    }
    out_stats = cfg.get_path("paths.neg_stats")
    with open(out_stats, "w") as f:
        json.dump(stats_out, f, indent=2, default=str)
    print(f"[03] 已保存: {out_pos} / {out_neg} / {out_stats}")

    # ---- 验收 ----
    ratio_min = cfg.get("negative_sampling.accept_ratio_min")
    ratio_max = cfg.get("negative_sampling.accept_ratio_max")
    print("\n========== 验收验证 ==========")
    ratio = stats_out["ratio"]
    print(f"正样本: {len(pos):,}  负样本: {len(neg):,}  正负比: {ratio} "
          f"(目标 {ratio_cfg * oversample:.2f} = 比例 {ratio_cfg} × oversample {oversample})")
    overlap = len(neg_pairs & set(zip(pos["met"], pos["tar"])))
    print(f"负样本与正样本重叠对: {overlap} (应为0)")
    print(f"负样本中曾被源表报道过的对: {n_prev_reported} (应为0，缺陷#8)")
    ok = (ratio_min <= ratio <= ratio_max) and overlap == 0 and (
        n_prev_reported == 0 or not exclude_reported)
    print(f"[03] 验收{'通过 ✓' if ok else '未通过 ✗'}")
    if not ok:
        print("[03] 需要调整采样参数。")


if __name__ == "__main__":
    main()
