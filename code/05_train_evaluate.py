#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 5 — 模型训练与评估（MLP 双任务）

任务 1 (task_binary): 互作二分类, y ∈ {0,1}，BCE。指标: ROC-AUC, PR-AUC, F1, Top-K Recall。
任务 2 (task_direction): 方向三分类, y ∈ {0,1,2}, CE。指标: 准确率, 混淆矩阵, macro-F1。
消融:  基线 = 仅 ECFP(smiles) 特征;  完整 = ECFP + 蛋白序列特征。

数据来自 Step 4:  data/features/*.npz / labels.npz / split.npz
--feat default → features.npz（dde 421 维蛋白，即 Baseline）
--feat esm     → features_esm.npz（ESM-3B 2560 维蛋白）
--variant <名> 让结果/指标/图文件名带上后缀，避免覆盖 Baseline。

交付:
  results/<prefix>_metrics.json
  results/<prefix>_binary_roc.png, <prefix>_binary_pr.png, <prefix>_direction_cm.png
参数统一来自 config.yaml (paths.* / training.*)。
"""
import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

import config as C
from fusion import FusionModule

cfg = C.load()

FEAT_DIR = cfg.get_path("paths.features_dir")
RESULT_DIR = cfg.get_path("paths.results_dir")
os.makedirs(RESULT_DIR, exist_ok=True)

SEED = cfg.get("training.seed")
BS = cfg.get("training.batch_size")
EPOCHS = cfg.get("training.epochs")
LR = cfg.get("training.lr")
PATIENCE = cfg.get("training.patience")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FP_DIM = cfg.get("features.fp_bits")  # ECFP 指纹位数


def set_seed(seed=SEED):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


class DualTaskMLP(nn.Module):
    """双任务 MLP。in_dim 为分子+蛋白拼接后的总维度。

    proj_esm (可选, int>0)：把蛋白部分投影到 proj_esm 维后与分子 concat（蛋白投影消融）。
    mol_proj (可选, int>0)：把分子块投影到 mol_proj 维后与蛋白 concat（分子投影，与 ESM proj 对齐）。
    fp_dim   ：分子块维度（ECFP / unimol2 CLS / hybrid 拼接）。默认取 in_dim - protein_dim。
    forward 按 sum(有效蛋白投影+有效分子投影) 的卷积顺序拼接，语义与维度均兼容原有 proj_esm。
    fusion   ：模态融合方式 concat/bilinear/gated/xattn（26-10-13 消融）。
               默认 concat = 原路径（逐位保持旧行为）；非 concat 时由 FusionModule
               接管 [mol|prot] → trunk 输入（融合自带投影，此时 proj_esm/mol_proj
               应传 0 以免双重投影）。forward 签名 (x)->(b,d) 不变。
    """
    def __init__(self, in_dim, hidden=None, proj_esm=0, protein_dim=None, fp_dim=None,
                 mol_proj=0, fusion=None, work_dim=None, out_dim=None,
                 xattn_tokens=None, xattn_heads=None):
        super().__init__()
        if hidden is None:
            hidden = cfg.get("training.hidden_dim")
        dropout = cfg.get("training.dropout")
        if fp_dim is None:
            fp_dim = FP_DIM
        self.fp_dim = fp_dim
        self.protein_dim = in_dim - fp_dim if protein_dim is None else protein_dim
        self.mol_proj = mol_proj
        self.fusion_kind = fusion or cfg.get("training.fusion", "concat")

        # 分子块投影层（可选）
        if mol_proj > 0:
            self.mol_proj_layer = nn.Linear(fp_dim, mol_proj)
            eff_mol = mol_proj
        else:
            eff_mol = fp_dim
        # 蛋白块投影层（可选，原有 proj_esm）
        # 缺陷#10：protein_dim==0（仅分子消融）时投影无意义——原实现会构造
        # nn.Linear(0, proj_esm)，其输出是与输入无关的常量偏置并拼进 trunk，
        # 白白增加参数且使消融模型结构与「真·仅分子」不一致。此处直接置零。
        if proj_esm > 0 and self.protein_dim > 0:
            self.esm_proj = nn.Linear(self.protein_dim, proj_esm)
            eff_prot = proj_esm
        else:
            if proj_esm > 0:
                print(f"[DualTaskMLP] protein_dim=0，忽略 proj_esm={proj_esm}（无蛋白块可投影）")
            proj_esm = 0
            eff_prot = self.protein_dim
        self.proj_esm = proj_esm

        if self.fusion_kind == "concat":
            # 原路径：投影后 concat，trunk 输入 = eff_mol + eff_prot（逐位保持旧行为）
            eff_in = eff_mol + eff_prot
        else:
            # 新融合（自带投影，公式见 fusion.FusionModule docstring）；
            # protein_dim==0（仅分子消融）时 FusionModule 内部退化为只用 mol
            self.fusion_module = FusionModule(
                self.fusion_kind, eff_mol, eff_prot,
                work_dim if work_dim is not None else cfg.get("training.work_dim", 256),
                out_dim if out_dim is not None else cfg.get("training.out_dim", 512),
                xattn_tokens if xattn_tokens is not None else cfg.get("training.xattn_tokens", 4),
                xattn_heads if xattn_heads is not None else cfg.get("training.xattn_heads", 4),
            )
            eff_in = self.fusion_module.out_dim

        self.fp = nn.Sequential(
            nn.Linear(eff_in, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
        )
        self.binary_head = nn.Linear(hidden, 1)
        self.dir_head = nn.Linear(hidden, 3)

    def _split(self, x):
        mol = x[:, :self.fp_dim]
        prot = x[:, self.fp_dim:self.fp_dim + self.protein_dim]
        if self.mol_proj > 0:
            mol = self.mol_proj_layer(mol)
        if self.proj_esm > 0:
            prot = self.esm_proj(prot)
        return mol, prot

    def forward(self, x):
        if self.fusion_kind == "concat":
            # 原路径（逐位保持）
            if self.proj_esm > 0 or self.mol_proj > 0:
                mol, prot = self._split(x)
                x = torch.cat([mol, prot], dim=1)
            h = self.fp(x)
        else:
            mol, prot = self._split(x)
            h = self.fp(self.fusion_module(mol, prot))
        return self.binary_head(h).squeeze(-1), self.dir_head(h)


def batch_gen(X, yb, ydc, idx, bs, shuffle=True):
    order = idx.copy()
    if shuffle:
        np.random.shuffle(order)
    for i in range(0, len(order), bs):
        b = order[i:i + bs]
        # .to(DEVICE) 对已在 GPU 上的 X（如 X_binary）是 no-op，对 CPU 上的 X_dir 才是真搬运
        yield X[b].to(DEVICE), yb[b], ydc[b]


def make_loader(X, yb, yd, idx, bs, shuffle=True):
    # 惰性生成器（26-10-13 OOM 修复）：原实现 list(batch_gen(...)) 会把整个 epoch 的
    # batch 预拷贝到 GPU（~6.4GB），加上常驻特征矩阵后超出 16GB 显存。改为逐批产生、
    # 逐批释放；batch 内容/顺序/随机性与原实现完全一致，训练数学路径零改动。
    return batch_gen(X, yb, yd, idx, bs, shuffle)


def eval_v1(model, device, X, yb, yd, idx, task='binary'):
    model.eval()
    with torch.no_grad():
        xb = X[idx].to(device)          # X_dir 在 CPU 时在此搬运；X_binary 已在 GPU 则 no-op
        if task == 'binary':
            out_b, _ = model(xb)
            y = yb[idx]
            return ((out_b.sigmoid() > 0.5).float() == y).float().mean().item()
        else:
            _, out_d = model(xb)
            y = yd[idx]
            return (out_d.argmax(1) == y).float().mean().item()


def train_binary(model, opt, criterion, device, X, yb, yd, tr, va):
    best_va = -1
    best_state = None
    bad = 0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb_b, _ in make_loader(X, yb, yd, tr, BS):
            opt.zero_grad()
            b_out, _ = model(xb)
            loss = criterion(b_out, yb_b.float())
            loss.backward()
            opt.step()
        va_acc = eval_v1(model, device, X, yb, yd, va, task='binary')
        if va_acc > best_va:
            best_va = va_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= PATIENCE:
            break
    model.load_state_dict(best_state)
    return best_va


def train_joint(model, opt, device, X_all, yb_all, ydir_full,
                b_tr, b_va, w_binary=None, w_dir=None):
    """multi-task 联合训练：共享表征，单模型同时优化互作 BCE 与方向 CE。
    方向监督仅正样本有效（ydir_full 其余为 -1）。
    每 batch 用互作样本 b_tr，同批内对带方向标签者叠加方向 CE；早停联合验证。"""
    if w_binary is None:
        w_binary = cfg.get("training.joint_w_binary")
    if w_dir is None:
        w_dir = cfg.get("training.joint_w_direction")
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()
    d_va_all = torch.tensor([i for i in b_va.tolist() if int(ydir_full[i]) != -1], device=device)
    best_score, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        order = b_tr.copy()
        np.random.shuffle(order)
        for i in range(0, len(order), BS):
            b = order[i:i + BS]
            opt.zero_grad()
            b_out, d_out = model(X_all[b])
            loss = w_binary * nn.functional.binary_cross_entropy_with_logits(b_out, yb_all[b].float())
            mask_d = ydir_full[b] != -1
            if mask_d.any():
                loss = loss + w_dir * ce(d_out[mask_d], ydir_full[b][mask_d])
            loss.backward()
            opt.step()
        bva = eval_v1(model, device, X_all, yb_all, yb_all, b_va, task='binary')
        dva = eval_v1(model, device, X_all, yb_all, yb_all, d_va_all, task='dir')
        score = 0.5 * (bva + dva)
        if score > best_score:
            best_score = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return best_score


def main():
    feat_map = cfg.get("training.feat_file_map")
    prefix_map = cfg.get("training.prefix_map")
    mol_dim_map = cfg.get("training.mol_dim_map")

    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", choices=list(feat_map.keys()), default="default",
                    help="特征变体；文件名映射见 config training.feat_file_map")
    ap.add_argument("--variant", default=None,
                    help="输出文件后缀（如 esm3b）；缺省时按 config training.prefix_map 决定")
    ap.add_argument("--proj_esm", type=int, default=0,
                    help=">0 时将蛋白特征投影到此维度再与分子 concat（蛋白 Projection 降维消融）")
    ap.add_argument("--mol_proj", type=int, default=0,
                    help=">0 时将分子块投影到此维度再与蛋白 concat（分子 Projection，与 ESM proj 对齐）")
    ap.add_argument("--fusion",
                    choices=["concat", "bilinear", "gated", "xattn",
                             "bilinear_raw", "gated_raw", "xattn_raw"],
                    default=cfg.get("training.fusion", "concat"),
                    help="模态融合方式（默认 concat=投影后拼接原路径；*_raw=v2 无瓶颈版，"
                         "mol 保持 raw、prot 用 proj_esm，与基线对齐；v1 融合自带投影配 --proj_esm 0）")
    ap.add_argument("--work_dim", type=int, default=cfg.get("training.work_dim", 256),
                    help="融合前各模态投影工作维（bilinear/gated 交互瓶颈；xattn token 总维）")
    ap.add_argument("--out_dim", type=int, default=cfg.get("training.out_dim", 512),
                    help="融合输出维（=trunk 输入维；concat 模式忽略）")
    ap.add_argument("--xattn_tokens", type=int, default=cfg.get("training.xattn_tokens", 4),
                    help="xattn pseudo-token 数（work_dim 必须整除此值）")
    ap.add_argument("--xattn_heads", type=int, default=cfg.get("training.xattn_heads", 4),
                    help="xattn 每方向注意力头数")
    ap.add_argument("--joint", action="store_true",
                    help="共享表征 multi-task 联合训练：单模型同时优化(互作 BCE + 方向 CE)，仅正样本有方向监督")
    ap.add_argument("--seed", type=int, default=None,
                    help="覆盖 training.seed（默认 config 值）；用于融合消融的多 seed 稳健性验证")
    args = ap.parse_args()
    seed_used = args.seed if args.seed is not None else SEED
    set_seed(seed_used)
    print(f"[05] device={DEVICE} seed={seed_used} proj_esm={args.proj_esm} mol_proj={args.mol_proj} "
          f"joint={args.joint} fusion={args.fusion} work_dim={args.work_dim} out_dim={args.out_dim} "
          f"xattn_tokens={args.xattn_tokens} xattn_heads={args.xattn_heads}")

    feat_name = feat_map[args.feat]
    prefix = args.variant or prefix_map[args.feat]
    base = FEAT_DIR
    d = np.load(os.path.join(base, feat_name))
    # 用与特征文件配套的 labels/split（见 cfg.sidecar_npz：不同编码器行数可能不同）
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))
    print(f"[05] 特征文件: {feat_name}  X_binary_dim={d['X_binary'].shape[1]}")

    # 显存策略：X_binary 常驻 GPU（任务1 + 消融都要用）；X_direction 留在 CPU，
    # 训练/评估时按批 .to(DEVICE)。原因：hybrid_esm 是 6144 维 × 399,528 行 = 9.8GB，
    # 加上 X_direction 的 4.9GB 就是 13.7GiB，而卡只有 15.48GiB，再叠加 eval 时
    # 一次性的大张量拷贝必然 OOM（05_hybrid_esm_mp512 实测爆过）。
    X_binary = torch.tensor(d["X_binary"]).float().to(DEVICE)
    X_dir = torch.tensor(d["X_direction"]).float()          # CPU，按批搬运
    y_binary = torch.tensor(l["y_binary"]).to(DEVICE)
    y_dir = torch.tensor(l["y_direction"]).to(DEVICE)

    # 依据 --feat 推断蛋白/分子块维度（供 DualTaskMLP 找分子-蛋白边界）
    auth_mol_dim = mol_dim_map[args.feat]
    protein_dim = X_binary.shape[1] - auth_mol_dim
    b_tr, b_va, b_te = s["b_tr"], s["b_va"], s["b_te"]
    d_tr, d_va, d_te = s["d_tr"], s["d_va"], s["d_te"]
    print(f"[05] 互作: train {len(b_tr):,} val {len(b_va):,} test {len(b_te):,}")
    print(f"[05] 方向: train {len(d_tr):,} val {len(d_va):,} test {len(d_te):,}")
    print(f"[05] 分子块 dim={auth_mol_dim} 蛋白块 dim={protein_dim}")

    # 方向类别名（按索引升序）：唯一真相来自 config，禁止手写 ["act","inh","neu"]
    label_names = cfg.direction_label_names()
    results = {}

    # 统一构造器：按分子/蛋白 dim 注入 fp_dim，并透传 mol_proj/proj_esm/fusion*
    def build_model(in_dim, hidden=None):
        return DualTaskMLP(in_dim=in_dim, hidden=hidden,
                           fp_dim=auth_mol_dim, protein_dim=protein_dim,
                           proj_esm=args.proj_esm, mol_proj=args.mol_proj,
                           fusion=args.fusion, work_dim=args.work_dim,
                           out_dim=args.out_dim, xattn_tokens=args.xattn_tokens,
                           xattn_heads=args.xattn_heads).to(DEVICE)

    # ============ [可选] multi-task 联合训练（共享表征）============
    if args.joint:
        print("\n===== Joint multi-task：共享表征 =====\n 模型同时优化 互作BCE + 方向CE（方向仅正样本）")
        in_dim = X_binary.shape[1]
        model_j = build_model(in_dim)
        optj = torch.optim.Adam(model_j.parameters(), lr=LR)
        # 方向全量掩码：互作全体 n_binary，方向标签仅正样本位置（前 len(pos)）
        n_all = X_binary.shape[0]
        ydir_full = torch.full((n_all,), -1, device=DEVICE, dtype=torch.long)
        ydir_full[:y_dir.numel()] = y_dir  # 方向样本 = 正样本（X_dir 与 X_binary 前段对齐）
        _ = train_joint(model_j, optj, DEVICE, X_binary, y_binary, ydir_full, b_tr, b_va)
        # 互作测试
        model_j.eval()
        with torch.no_grad():
            out_bj, _ = model_j(X_binary[b_te])
        prob_j = out_bj.sigmoid().cpu().numpy()
        yt_j = y_binary[b_te].cpu().numpy()
        roc_j = roc_auc_score(yt_j, prob_j)
        apr_j = average_precision_score(yt_j, prob_j)
        results['binary_joint'] = {"roc_auc": round(float(roc_j), 4), "pr_auc": round(float(apr_j), 4)}
        print(f"[05-joint] 互作 Test: ROC-AUC={roc_j:.4f} PR-AUC={apr_j:.4f}")
        # 方向性能（d_te 是方向样本的真实索引）
        with torch.no_grad():
            _, d_log_j = model_j(X_dir[d_te].to(DEVICE))
        ydt_j = y_dir[d_te].cpu().numpy()
        d_pred_j = d_log_j.argmax(1).cpu().numpy()
        d_acc_j = accuracy_score(ydt_j, d_pred_j)
        results['direction_joint'] = {"accuracy": round(float(d_acc_j), 4)}
        print(f"[Joint] 方向 Test: Acc={d_acc_j:.4f}")

    # ============ 任务1：互作二分类 ============
    print("\n===== 任务1 互作二分类 =====")
    in_dim = X_binary.shape[1]
    model_b = build_model(in_dim)
    opt = torch.optim.Adam(model_b.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    _ = train_binary(model_b, opt, crit, DEVICE, X_binary, y_binary, y_binary, b_tr, b_va)

    model_b.eval()
    with torch.no_grad():
        # 用完整 forward（当 proj_esm>0 时内含 Projection），binary_head 在第一个输出
        out_b, _ = model_b(X_binary[b_te])
        probs = out_b.sigmoid().cpu().numpy()
    yt = y_binary[b_te].cpu().numpy()
    roc = roc_auc_score(yt, probs)
    apr = average_precision_score(yt, probs)
    pred = (probs > 0.5).astype(int)
    f1 = f1_score(yt, pred)
    prec = precision_score(yt, pred)
    rec = recall_score(yt, pred)
    # ---- Top-K 筛选指标（K = topk_frac * 测试集）----
    # 缺陷#5：yt[topk_idx].mean() 是「前 K 个预测中正例的占比」= Precision@K，不是 Recall。
    # 真正的 Recall@K = 命中数 / 总正例数，物理上限 K/n_pos（本数据约 2%），不可能接近 1。
    k = max(1, int(cfg.get("training.topk_frac") * len(yt)))
    topk_idx = np.argsort(-probs)[:k]
    n_pos_te = int(yt.sum())
    hits = int(yt[topk_idx].sum())
    precision_at_topk = float(yt[topk_idx].mean())
    recall_at_topk = (hits / n_pos_te) if n_pos_te else 0.0
    base_rate = (n_pos_te / len(yt)) if len(yt) else 0.0
    enrichment_factor = (precision_at_topk / base_rate) if base_rate else 0.0
    results['binary'] = {"roc_auc": roc, "pr_auc": apr, "f1": f1,
                         "precision": prec, "recall": rec,
                         "precision_at_topk": precision_at_topk,
                         "recall_at_topk": recall_at_topk,
                         "enrichment_factor": enrichment_factor,
                         "topk_k": k,
                         "accuracy": accuracy_score(yt, pred)}
    print(f"[05] 互作 Test: ROC-AUC={roc:.4f} PR-AUC={apr:.4f} F1={f1:.4f} "
          f"Prec={prec:.4f} Rec={rec:.4f} Acc={results['binary']['accuracy']:.4f}")
    print(f"[05] Top-{k} (K=1% 测试集): Precision@K={precision_at_topk:.4f} "
          f"Recall@K={recall_at_topk:.4f} (上限 {k/max(n_pos_te,1):.4f}) EF={enrichment_factor:.2f}")

    # ============ 任务2 方向三分类 ============
    print("\n===== 任务2 方向三分类 =====")
    model_dir = build_model(X_dir.shape[1])
    opt2 = torch.optim.Adam(model_dir.parameters(), lr=LR)
    crit2 = nn.CrossEntropyLoss()
    best_dir_va = -1
    best_dstate = None
    bad2 = 0
    for ep in range(min(EPOCHS, cfg.get("training.direction_max_epochs"))):
        model_dir.train()
        for xb, _, yd_b in make_loader(X_dir, y_dir, y_dir, d_tr, BS):
            opt2.zero_grad()
            _, d_out = model_dir(xb)
            loss = crit2(d_out, yd_b)
            loss.backward()
            opt2.step()
        va_d = eval_v1(model_dir, DEVICE, X_dir, y_dir, y_dir, d_va, task='dir')
        if va_d > best_dir_va:
            best_dir_va = va_d
            best_dstate = {k: v.clone() for k, v in model_dir.state_dict().items()}
            bad2 = 0
        else:
            bad2 += 1
        if bad2 >= PATIENCE:
            break
    model_dir.load_state_dict(best_dstate)
    model_dir.eval()
    with torch.no_grad():
        _, d_logits = model_dir(X_dir[d_te].to(DEVICE))
    ydt = y_dir[d_te].cpu().numpy()
    d_pred = d_logits.argmax(1).cpu().numpy()
    d_acc = accuracy_score(ydt, d_pred)
    d_cm = confusion_matrix(ydt, d_pred)
    d_f1 = f1_score(ydt, d_pred, average='macro')
    results['direction'] = {"accuracy": d_acc, "macro_f1": d_f1,
                            "confusion_matrix_label_order": label_names,
                            "cm": d_cm.tolist()}
    print(f"[05] 方向 Test: Acc={d_acc:.4f} macro-F1={d_f1:.4f}")
    print(f"[05] 方向混淆矩阵 (rows=真实, 顺序 {label_names}):")
    print(d_cm)

    # ============ 消融: 仅分子特征（去掉蛋白特征）============
    print("\n===== 消融: 仅分子特征 (去掉蛋白特征) =====")
    # 仅取分子块前 auth_mol_dim 列；input 维随蛋白表征变化而保留
    X_binary_fp = X_binary[:, :auth_mol_dim]
    # proj_esm 强制 0：protein_dim=0 时蛋白投影无意义（缺陷#10，消融保持干净）
    model_fp = DualTaskMLP(auth_mol_dim, fp_dim=auth_mol_dim, protein_dim=0,
                           proj_esm=0, mol_proj=args.mol_proj,
                           fusion=args.fusion, work_dim=args.work_dim,
                           out_dim=args.out_dim, xattn_tokens=args.xattn_tokens,
                           xattn_heads=args.xattn_heads).to(DEVICE)
    optf = torch.optim.Adam(model_fp.parameters(), lr=LR)
    critf = nn.BCEWithLogitsLoss()
    _ = train_binary(model_fp, optf, critf, DEVICE, X_binary_fp, y_binary, y_binary, b_tr, b_va)
    model_fp.eval()
    with torch.no_grad():
        # 走完整 forward：mol_proj>0 时内部先投影分子块，再交 fp；避免绕开投影直接喂 fp 造成维不匹配
        out_fp, _ = model_fp(X_binary_fp[b_te])
        prob_fp = out_fp.sigmoid().cpu().numpy()
    roc_fp = roc_auc_score(yt, prob_fp)
    apr_fp = average_precision_score(yt, prob_fp)
    results['ablation'] = {"binary_mol_only": {"roc_auc": roc_fp, "pr_auc": apr_fp, "protein_dim": int(protein_dim)}}
    print(f"[05] 仅分子特征: ROC-AUC={roc_fp:.4f} PR-AUC={apr_fp:.4f}  "
          f"vs Full ROC-AUC={roc:.4f} (Δ={roc-roc_fp:+.4f})  蛋白维={protein_dim}")

    # 保存指标
    metrics_path = os.path.join(RESULT_DIR, f"{prefix}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[05] 指标已保存: {metrics_path}")

    # 保存图（修正 y_true 为二值标签，避免连续概率问题）
    yt_bin = np.where(y_binary[b_te].cpu().numpy() >= 0.5, 1, 0)
    _save_curves(probs, yt_bin, RESULT_DIR, prefix)
    _save_cm(d_cm, RESULT_DIR, prefix, label_names=label_names)

    # ============ 验收 ============
    accept_roc = cfg.get("training.accept_binary_roc_auc")
    accept_dir_acc = cfg.get("training.accept_direction_acc")
    print("\n========== 验收验证 ==========")
    ok_auc = roc >= accept_roc
    ok_dir = d_acc >= accept_dir_acc
    print(f"互作 ROC-AUC={roc:.4f} (验收>={accept_roc}): {'通过 ✓' if ok_auc else '未达 ✗'}")
    print(f"方向准确率={d_acc:.4f} (验收>={accept_dir_acc}): {'通过 ✓' if ok_dir else '未达 ✗'}")
    ok = ok_auc and ok_dir
    print(f"\n[05] 整体验收{'通过 ✓' if ok else '未通过 ✗'}")


def _save_curves(probs, yt, rd, prefix="step5"):
    from sklearn.metrics import roc_curve, precision_recall_curve
    fpr, tpr, _ = roc_curve(yt, probs)
    prec, rec, _ = precision_recall_curve(yt, probs)
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title("ROC"); plt.savefig(os.path.join(rd, f"{prefix}_roc.png")); plt.close()
    plt.figure(); plt.plot(rec, prec); plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("PR"); plt.savefig(os.path.join(rd, f"{prefix}_pr.png")); plt.close()
    # 曲线数组一并落盘：否则论文里的 ROC/PR 图只能从 PNG 重画（无法提取数据），
    # 也无法在不重训的前提下调整样式。与 direction.cm 落盘同理。
    np.savez_compressed(os.path.join(rd, f"{prefix}_curves.npz"),
                        fpr=fpr, tpr=tpr, precision=prec, recall=rec,
                        y_true=yt, probs=probs)


def _save_cm(cm, rd, prefix="step5", label_names=None):
    """方向混淆矩阵图。刻度按 label_names 顺序（索引 0,1,2 = neutral,inhibition,activation）。

    历史 bug：此处硬编码 ["act","inh","neu"]，与 confusion_matrix 的行列顺序
    （模型类别索引 0/1/2 = neutral/inhibition/activation）相反，导致整张图行列标注错位。
    """
    if label_names is None:
        label_names = cfg.direction_label_names()
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    ticks = range(len(label_names))
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(label_names); ax.set_yticklabels(label_names)
    ax.set_xlabel("predicted"); ax.set_ylabel("true")
    ax.set_title("Direction confusion matrix"); plt.colorbar(im)
    plt.savefig(os.path.join(rd, f"{prefix}_direction_cm.png")); plt.close()


if __name__ == "__main__":
    main()
