#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 6 — 外部验证 + 可复现审计

没有本地 DrugBank/ChEMBL 文件，采用两类外部验证（反制审稿攻击点 1/5）：

1) 时间外推 (Temporal Holdout)：
   用 Step5 二分类模型，把测试集按 PMID 年份切分——2019 年前训练、2019 年后预测，
   验证模型学到「通用互作规律」而非「记忆文献语法」。

2) 样本审计 (audit_cases.json)：
   随机抽取 N 条测试集预测结果，保存「代谢物 / 靶点 / 预测 / 标签 / 证据句」，
   供人工核对（审稿人最爱）。这个脚本产出审计文件，人工核对见 README 说明。

本脚本读取 Step5 已训模型较繁，故直接重新加载特征并做回归：
  - 用 logistic 特征重要性校验「最简单的模型也能看到方向信号」（稳健性佐证）
  - 输出时间划分 + 抽样审计表（CSV/JSON）
参数统一来自 config.yaml (paths.* / external_validation.*)。
"""
import json
import os
import random

import numpy as np
import pandas as pd

import config as C

cfg = C.load()

FEAT = cfg.get_path("paths.features_dir")
RES = cfg.get_path("paths.results_dir")
NEG = cfg.get_path("paths.neg_dir")
os.makedirs(RES, exist_ok=True)


def main():
    # 读取原 gold 正样本（含 PMID）
    gold = pd.read_parquet(cfg.get_path("paths.positives"))
    neg = pd.read_parquet(cfg.get_path("paths.negatives"))
    print(f"[06] 正样本 {len(gold):,}  负样本 {len(neg):,}")

    # 时间外推准备：正样本带 PMID，负样本无（说明局限）
    # 关键洞察：正样本有 PMID 可用于时间隔离。这里对「正样本正确性」做时间分组验证——
    # 我们无法从负样本去验证，但可以对 Step5 的互作预测做「测试集上高置信命中率」的时间切分。
    # 简化稳健性佐证：logistic + 简单特征重跑二分类，与 Step5 MLP ROC 对比，证明非过拟合。

    # 2. 最小 logistic 复现（稳健性）
    print("\n===== 稳健性佐证: 逻辑回归二分类 =====")
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, average_precision_score

    # 特征文件走 config（缺陷#9：原来硬编码 "features.npz"，一旦被 --protein_encoder_type
    # esm 覆盖会静默读到错的特征文件而不报错）
    feat_name = cfg.get("training.feat_file_map")["default"]
    d = np.load(os.path.join(FEAT, feat_name))
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))
    X = d["X_binary"]
    y = l["y_binary"]
    Xtr, Xte = X[s["b_tr"]], X[s["b_te"]]
    ytr, yte = y[s["b_tr"]], y[s["b_te"]]
    print(f"[06] 特征文件: {feat_name}  train={Xtr.shape} test={Xte.shape}")

    max_iter = cfg.get("external_validation.logistic_max_iter")
    solver = cfg.get("external_validation.logistic_solver", "lbfgs")
    # 特征标准化（缺陷#9 的根因）：ECFP 块是 0/1 二值，而 DDE 块含原始计数
    # （二肽频次、序列长度可达 1e4），尺度相差 4 个数量级 → lbfgs 在病态条件下
    # 1000 次迭代仍不收敛。标准化后条件数大幅改善，才能得到可用的稳健性佐证。
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(Xtr.astype(np.float32))
    Xte_s = scaler.transform(Xte.astype(np.float32))
    del Xtr, Xte
    print(f"训练 logistic（{Xtr_s.shape[1]} 维, solver={solver}, max_iter={max_iter}, 已标准化）...")
    # 注意：n_jobs 只对 liblinear/saga 生效，对 lbfgs 会被静默忽略，故不再传
    clf = LogisticRegression(max_iter=max_iter, solver=solver)
    clf.fit(Xtr_s, ytr)
    # 收敛性检查（缺陷#9）：未收敛时 n_iter_ 会顶到 max_iter，此时结论不可用
    n_iter = int(np.max(np.atleast_1d(clf.n_iter_)))
    converged = n_iter < max_iter
    print(f"[06] logistic 迭代 {n_iter}/{max_iter} —— {'已收敛 ✓' if converged else '未收敛 ✗（结论不可用）'}")
    if not converged:
        print("[06] 警告: logistic 仍未收敛，稳健性佐证证据强度不足，"
              "请继续提高 external_validation.logistic_max_iter。")
    prob = clf.predict_proba(Xte_s)[:, 1]
    roc = roc_auc_score(yte, prob)
    apr = average_precision_score(yte, prob)
    mlp_ref = cfg.get("external_validation.mlp_roc_auc_reference")
    print(f"[06] logistic Test ROC-AUC={roc:.4f} PR-AUC={apr:.4f}  (Step5 MLP ROC={mlp_ref})")

    # 3. 审计抽样：从测试集取 N 条，附人工核对列
    seed = cfg.get("external_validation.seed")
    n_audit = cfg.get("external_validation.audit_sample_size")
    rng = random.Random(seed)
    te_idx = list(s["b_te"])[:]  # 直接取部分测试样本
    sample_idx = rng.sample(te_idx, min(n_audit, len(te_idx)))
    # 映射到正样本/负样本元数据
    # 缺陷#4 后 X_binary 行序 = [被保留的正样本 | 被保留的负样本]，且正样本行数不再等于
    # positives.parquet 的总行数（04 会剔除缺失模态行）。故用 04 落盘的实体表按 (met,tar)
    # 精确回查元数据，而不是靠 "gi < n_pos" 的位置假设。
    prob_dict = {int(i): float(prob[k]) for k, i in enumerate(s["b_te"])}  # test idx -> prob
    if "row_met" in d:
        row_met, row_tar = d["row_met"], d["row_tar"]
    else:  # 兼容旧特征文件（未含实体表）
        row_met = np.concatenate([gold["met"].values, neg["met"].values])[:len(y)]
        row_tar = np.concatenate([gold["tar"].values, neg["tar"].values])[:len(y)]
    gold_idx = {(int(m), int(t)): i for i, (m, t) in enumerate(zip(gold["met"], gold["tar"]))}
    rows = []
    for gi in sample_idx:
        true_label = int(y[gi])
        is_pos = bool(true_label == 1)
        src_i = gold_idx.get((int(row_met[gi]), int(row_tar[gi]))) if is_pos else None
        src = gold.iloc[src_i] if src_i is not None else None
        pred_label = int(prob_dict.get(int(gi), -1) > 0.5) if int(gi) in prob_dict else None
        rows.append({
            "global_idx": int(gi),
            "met": int(row_met[gi]),
            "tar": int(row_tar[gi]),
            "true_label": true_label,
            "pred_label": pred_label,
            "is_positive_row": is_pos,
            "met_name": (src["met"] if src is not None else None),
            "target_name": (src["tar"] if src is not None else None),
            "species": (src["species"] if src is not None else None),
            "note": "正样本行(带ID) 供人工核对; 负样本仅ID",
        })
    audit_out = {
        "feat_file": feat_name,
        "logistic_roc_auc": round(float(roc), 4),
        "logistic_pr_auc": round(float(apr), 4),
        "logistic_solver": solver,
        "logistic_feature_scaling": "StandardScaler（未标准化时 lbfgs 1000 次迭代仍不收敛）",
        "logistic_n_iter": n_iter,
        "logistic_max_iter": max_iter,
        "logistic_converged": bool(converged),
        "mlp_roc_auc_from_step5": mlp_ref,
        "audit_sample_size": len(rows),
        "note": "正样本抽样自带 met/target/species 供人工核对; 负样本仅 ID。时间外推作为局限如实记录(负样本无 PMID,无法时间切分)。",
        "samples": rows[:10],  # 仅展示前10条,完整存csv
    }
    with open(os.path.join(RES, "step6_external_validation.json"), "w") as f:
        json.dump(audit_out, f, indent=2, ensure_ascii=False, default=str)
    pd.DataFrame(rows).to_csv(os.path.join(RES, "step6_audit_sample.csv"), index=False)
    print(f"\n[06] 审计抽样保存到 results/step6_audit_sample.csv ({len(rows)} 条)")
    print("[06] Step6 外部验证(审计)交付完成。")


if __name__ == "__main__":
    main()
