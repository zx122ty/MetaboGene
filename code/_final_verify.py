#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""最终验收：逐项核对 prompt/26-10-15/05-检查结果.md 的 Verification Checklist。

只读，不改任何产物。运行完打印通过/未通过汇总。
"""
import glob
import json
import os
import re
import subprocess
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

cfg = C.load()
ROOT = cfg.base_dir
FEAT = cfg.get_path("paths.features_dir")
RES = cfg.get_path("paths.results_dir")
CODE = os.path.join(ROOT, "code")
PASS, FAIL = [], []


def chk(ok, item, detail=""):
    (PASS if ok else FAIL).append(item)
    print(f"  {'✓' if ok else '✗'} {item}" + (f"  — {detail}" if detail else ""))


def grep(pattern, paths):
    try:
        out = subprocess.run(["grep", "-rn", pattern] + paths,
                             capture_output=True, text=True).stdout
    except Exception:
        return ""
    return out


def main():
    print("=" * 74)
    print("#1 方向标签映射：全流程唯一真相")
    print("=" * 74)
    hits = grep("label2idx", [CODE])
    hard = [l for l in hits.splitlines()
            if '"activation": 0' in l or '"activation":0' in l]
    chk(not hard, "无硬编码 {'activation':0,...} 残留", f"{len(hard)} 处")
    users = [l.split(":")[0].split("/")[-1] for l in grep("direction_label_map()", [CODE]).splitlines()]
    chk(len(set(users)) >= 4, "多处改为 cfg.direction_label_map()", f"{sorted(set(users))}")

    print()
    print("=" * 74)
    print("#2 混淆矩阵刻度")
    print("=" * 74)
    names = cfg.direction_label_names()
    chk(names == ["neutral", "inhibition", "activation"], "config 刻度顺序正确", f"{names}")
    src = open(os.path.join(CODE, "05_train_evaluate.py")).read()
    # 只检查【实际调用】而非文档字符串里对历史 bug 的描述
    chk('set_xticklabels(["act"' not in src and "set_xticklabels(['act'" not in src,
        "05 不再硬编码 act/inh/neu 刻度")
    chk("ax.set_xticklabels(label_names)" in src, "_save_cm 按 label_names 生成刻度")
    cm_pngs = glob.glob(os.path.join(RES, "*_direction_cm.png"))
    fresh = [p for p in cm_pngs if os.path.getmtime(p) > os.path.getmtime(
        os.path.join(ROOT, "prompt", "26-10-15", "05-检查结果.md"))]
    chk(len(fresh) >= 27, "混淆矩阵图已重出", f"{len(fresh)}/{len(cm_pngs)} 张晚于检查报告")

    print()
    print("=" * 74)
    print("#3 特征文件命名由 (mol, protein) 决定")
    print("=" * 74)
    expect = {"features.npz": 2469, "features_esm.npz": 4608,
              "features_unimol.npz": 1957, "features_unimol_esm.npz": 4096,
              "features_hybrid.npz": 4005, "features_hybrid_esm.npz": 6144}
    for fn, dim in expect.items():
        p = os.path.join(FEAT, fn)
        ok = os.path.exists(p) and np.load(p)["X_binary"].shape[1] == dim
        chk(ok, f"{fn} = {dim} 维")
    d_u = np.load(os.path.join(FEAT, "features_unimol.npz"))["X_binary"].shape[1]
    d_ue = np.load(os.path.join(FEAT, "features_unimol_esm.npz"))["X_binary"].shape[1]
    chk(d_u != d_ue, "unimol(dde) 与 unimol_esm 维度不同", f"{d_u} vs {d_ue}")

    print()
    print("=" * 74)
    print("#4 缺失模态泄漏通道")
    print("=" * 74)
    d = np.load(os.path.join(FEAT, "features_esm.npz"))
    miss = ~(d["has_smiles"] & d["has_seq"])
    chk(int(miss.sum()) == 0, "保留行 100% 模态完整", f"缺失 {int(miss.sum())} 行")
    fs = json.load(open(cfg.get_path("paths.feature_stats")))
    mf = fs.get("modality_filter", {})
    chk("pos_dropped_incomplete" in mf and "neg_dropped_incomplete" in mf,
        "输出含剔除统计与按标签缺失率",
        f"pos 剔 {mf.get('pos_dropped_incomplete')}, neg 剔 {mf.get('neg_dropped_incomplete')}")
    # 等覆盖率：正负都不再缺模态 → 该 bit 无区分度
    chk(mf.get("final_ratio_neg_per_pos") == 1.0, "过滤后正负严格 1:1",
        f"ratio={mf.get('final_ratio_neg_per_pos')}")

    print()
    print("=" * 74)
    print("#5 指标更名 topk_recall → precision_at_topk")
    print("=" * 74)
    src5 = open(os.path.join(CODE, "05_train_evaluate.py")).read()
    chk("topk_recall" not in re.sub(r"#.*", "", src5), "05 源码无 topk_recall")
    m = json.load(open(os.path.join(RES, "step5_esm_proj1024_metrics.json")))
    b = m["binary"]
    chk("precision_at_topk" in b and "recall_at_topk" in b and "enrichment_factor" in b,
        "新指标已落盘",
        f"P@K={b['precision_at_topk']:.4f} R@K={b['recall_at_topk']:.4f} EF={b['enrichment_factor']:.2f}")
    chk(b["recall_at_topk"] < 0.05, "真 Recall@1% 远小于 1（物理上限 ~2%）",
        f"{b['recall_at_topk']:.4f}")

    print()
    print("=" * 74)
    print("#6 05b --variant 生效")
    print("=" * 74)
    srcb = open(os.path.join(CODE, "05b_direction_variants.py")).read()
    chk("f'_{suffix}'" in srcb or "{suffix}" in srcb, "out_path 使用 suffix")

    print()
    print("=" * 74)
    print("#7 负样本分布检验方法")
    print("=" * 74)
    ns = json.load(open(cfg.get_path("paths.neg_stats")))
    chk("species_ks_p" not in ns, "已移除误用的 KS")
    chk("species_tvd" in ns and "species_chi2_p" in ns, "改用卡方 + TVD",
        f"TVD={ns.get('species_tvd')} chi2_p={ns.get('species_chi2_p')}")

    print()
    print("=" * 74)
    print("#8 负样本不再与正样本同源")
    print("=" * 74)
    chk(ns.get("neg_previously_reported_frac") == 0,
        "负样本中曾被源表报道过的比例为 0",
        f"排除 {ns.get('n_reported_pairs_excluded'):,} 个 pair")

    print()
    print("=" * 74)
    print("#9 06 logistic 收敛且路径走 config")
    print("=" * 74)
    s6 = json.load(open(os.path.join(RES, "step6_external_validation.json")))
    chk(s6.get("logistic_converged") is True, "logistic 已收敛",
        f"n_iter={s6.get('logistic_n_iter')}/{s6.get('logistic_max_iter')}")
    src6 = open(os.path.join(CODE, "06_external_validation.py")).read()
    chk('os.path.join(FEAT, "features.npz")' not in src6, "不再硬编码 features.npz")

    print()
    print("=" * 74)
    print("#10 仅分子消融不再建常量投影层")
    print("=" * 74)
    src10 = open(os.path.join(CODE, "05_train_evaluate.py")).read()
    chk("self.protein_dim > 0" in src10, "DualTaskMLP 有 protein_dim>0 守卫")
    chk("proj_esm=0, mol_proj=args.mol_proj" in src10, "消融路径显式传 proj_esm=0")

    print()
    print("=" * 74)
    print("外部方向迁移（#1 的最终数字）")
    print("=" * 74)
    d10 = json.load(open(os.path.join(RES, "step10_external_sources_metrics.json")))
    db = d10.get("drugbank_direction", {})
    rx = d10.get("reactome_direction", {})
    ctd = json.load(open(os.path.join(RES, "step12_ctd_direction_metrics.json")))
    print(f"  DrugBank  3-class={db.get('direction_acc')}  act-vs-inh={db.get('act_vs_inh_acc')}  n={db.get('n_pairs')}")
    print(f"  Reactome  3-class={rx.get('direction_acc')}  n={rx.get('n_pairs')}  (样本过少)")
    print(f"  CTD       act-vs-inh={ctd.get('act_vs_inh_acc')}  n={ctd.get('n_pairs')}  (仍低于随机)")

    print()
    print("=" * 74)
    print(f"通过 {len(PASS)} 项，未通过 {len(FAIL)} 项")
    if FAIL:
        for f in FAIL:
            print(f"  ✗ {f}")
    print("=" * 74)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
