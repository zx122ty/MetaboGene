#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""features 阶段验收：逐项核对专家检查清单里「数据层」的检查项。

只读，不改任何产物。
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C  # noqa: E402

cfg = C.load()
FEAT = cfg.get_path("paths.features_dir")
FAIL = []


def check(ok, label, detail=""):
    print(f"  {'✓' if ok else '✗'} {label}{('  ' + detail) if detail else ''}")
    if not ok:
        FAIL.append(label)


def main():
    print("=" * 72)
    print("1) 六个特征文件：维度 / 完整性掩码 / 实体表")
    print("=" * 72)
    expect = {
        "features.npz": 2469, "features_esm.npz": 4608,
        "features_unimol.npz": 1957, "features_unimol_esm.npz": 4096,
        "features_hybrid.npz": 4005, "features_hybrid_esm.npz": 6144,
    }
    ref_rows = None
    for fn, dim in expect.items():
        p = os.path.join(FEAT, fn)
        if not os.path.exists(p):
            check(False, fn, "文件不存在")
            continue
        d = np.load(p)
        Xb, Xd = d["X_binary"], d["X_direction"]
        n = Xb.shape[0]
        complete = bool((d["has_smiles"] & d["has_seq"]).all())
        has_ent = "row_met" in d and "dir_met" in d
        check(Xb.shape[1] == dim, f"{fn}", f"dim={Xb.shape[1]} (期望 {dim}), n={n:,}")
        # 缺陷#4 的核心断言：保留行必须 100% 模态完整
        check(complete, f"{fn} 掩码全 True", f"完整率={(d['has_smiles'] & d['has_seq']).mean():.4f}")
        check(has_ent, f"{fn} 含实体表 row_met/dir_met")
        if ref_rows is None:
            ref_rows = n

    print()
    print("=" * 72)
    print("2) labels.npz / split.npz")
    print("=" * 72)
    l = np.load(cfg.get_path("paths.labels_npz"))
    s = np.load(cfg.get_path("paths.split_npz"))
    yb, yd = l["y_binary"], l["y_direction"]
    cnt = dict(zip(*[x.tolist() for x in np.unique(yb, return_counts=True)]))
    ratio = cnt.get(0, 0) / max(cnt.get(1, 1), 1)
    check(abs(ratio - 1.0) < 1e-6, "互作正负 1:1", f"{cnt}")
    dmap = cfg.direction_label_map()
    dname = {v: k for k, v in dmap.items()}
    ddist = {dname.get(int(k), k): int(v) for k, v in zip(*np.unique(yd, return_counts=True))}
    check(len(yd) == cnt.get(1, 0), "方向样本数 == 正样本数", f"{len(yd):,} vs {cnt.get(1,0):,}")
    print(f"    方向分布: {ddist}  (0=neutral 1=inhibition 2=activation)")
    check(set(ddist) <= {"neutral", "inhibition", "activation"}, "方向标签取值合法")
    sizes = {k: int(len(s[k])) for k in ("b_tr", "b_va", "b_te")}
    check(abs(sizes["b_tr"] / sum(sizes.values()) - 0.8) < 0.01, "互作 80/10/10 划分",
          f"{sizes}")
    dsizes = {k: int(len(s[k])) for k in ("d_tr", "d_va", "d_te")}
    check(abs(dsizes["d_tr"] / sum(dsizes.values()) - 0.8) < 0.01, "方向 80/10/10 划分",
          f"{dsizes}")

    print()
    print("=" * 72)
    print("3) 缺陷#4 核心：缺失模态泄漏通道是否已消除")
    print("=" * 72)
    d = np.load(os.path.join(FEAT, "features_esm.npz"))
    miss = ~(d["has_smiles"] & d["has_seq"])
    n_miss = int(miss.sum())
    check(n_miss == 0, "任一模态缺失行数", f"{n_miss} (期望 0)")
    if n_miss == 0:
        print("    → 缺失 bit 为常数，「缺模态」不再携带任何标签信息")
        print("    → 原泄漏 AUC 0.5896 的通道已结构性关闭（旧值见 prompt/26-10-15/05-检查结果.md）")

    print()
    print("=" * 72)
    print("4) 实体表与 X_binary 行序一致性（下游 05c/13/14 依赖）")
    print("=" * 72)
    row_met, row_tar = d["row_met"], d["row_tar"]
    dir_met, dir_tar = d["dir_met"], d["dir_tar"]
    n_pos = len(dir_met)
    check(len(row_met) == len(yb), "row_met 行数 == y_binary 长度")
    check(n_pos == cnt.get(1, 0), "dir_met 行数 == 正样本数", f"{n_pos:,}")
    check(bool((row_met[:n_pos] == dir_met).all() and (row_tar[:n_pos] == dir_tar).all()),
          "前 n_pos 行即方向样本（行序约定成立）")
    neg_pairs = set(zip(row_met[n_pos:].tolist(), row_tar[n_pos:].tolist()))
    check(len(neg_pairs) == len(row_met) - n_pos, "负样本 pair 唯一",
          f"{len(neg_pairs):,}/{len(row_met)-n_pos:,}")

    print()
    print("=" * 72)
    print("5) 负样本审计（缺陷#8）")
    print("=" * 72)
    ns = json.load(open(cfg.get_path("paths.neg_stats")))
    check(ns.get("neg_previously_reported_frac") == 0,
          "负样本中曾被源表报道过的比例", f"{ns.get('neg_previously_reported_frac')}")
    check("species_tvd" in ns, "物种分布改用 TVD",
          f"TVD={ns.get('species_tvd')} chi2_p={ns.get('species_chi2_p')}")
    check("species_ks_p" not in ns, "已移除误用的 KS 统计量")
    check(ns.get("n_reported_pairs_excluded", 0) > 0, "已排除源表互作 pair",
          f"{ns.get('n_reported_pairs_excluded'):,}")

    print()
    print("=" * 72)
    if FAIL:
        print(f"✗ {len(FAIL)} 项未通过:")
        for f in FAIL:
            print(f"    - {f}")
        return 1
    print("✓ 全部检查通过")
    return 0


if __name__ == "__main__":
    sys.exit(main())
