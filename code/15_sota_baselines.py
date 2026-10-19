#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 15 — SOTA DTI baselines: DeepDTA / GraphDTA / MolTrans.

设计文档: prompt/26-10-16/03-SOTA基线补充.md（v2）

关键契约
--------
* 输入是原始 SMILES + 蛋白序列；**不使用**本文的 ECFP4 / ESM 特征（F3）。
* 数据、划分、早停协议、指标口径与本文模型一致：
    - row-level: split_features_esm.npz
    - strict   : split_strict_features_esm.npz（code/strict_split.py，唯一真相）
    - cold-*   : import 14_uncertainty_stats.cold_split_indices，
                 rng = RandomState(42 + seed)（与主模型 14 号 run 同口径）
* 双协议（§3.4）:
    - own   : 各自原文超参（DeepDTA 100ep/lr1e-3；GraphDTA 1000ep/lr5e-4，
              本实现因算力截断为 300ep；MolTrans 100ep/lr1e-3/batch128）
    - paper : Adam lr2e-3 / batch512 / max15ep / patience6 / 验证 accuracy
              （MolTrans batch 128；本实现 16GB 下 64）
* 截断契约（§4.0）：主结果 max_len_smi=128 / max_len_prot=1500；
  原文值（DeepDTA 85/1000、MolTrans 50/545）作敏感性。
* 每个 run 落盘逐 epoch 曲线（F12）与测试集预测（供 F11 DeLong）。

产物
----
results/sota_runs/<model>_<split>_s<seed>_<protocol>.json
results/sota_curves/<model>_<split>_s<seed>_<protocol>.npz
results/sota_preds/<model>_<split>_s<seed>_<protocol>.npz
results/sota_ckpt/<model>_<split>_s<seed>_<protocol>.pt
results/sota_external_metrics.json           (--external all)
results/sota_baselines_summary.json + .md    (--summarize)
"""
import argparse
import importlib.util
import json
import os
import pickle
import random
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config as C
import strict_split

cfg = C.load()
FEAT_DIR = cfg.get_path("paths.features_dir")
RES_DIR = cfg.get_path("paths.results_dir")
CACHE_DIR = os.path.join(FEAT_DIR, "sota_cache")
RUNS_DIR = os.path.join(RES_DIR, "sota_runs")
CURVES_DIR = os.path.join(RES_DIR, "sota_curves")
PRED_DIR = os.path.join(RES_DIR, "sota_preds")
CKPT_DIR = os.path.join(RES_DIR, "sota_ckpt")
LOG_DIR = os.path.join(cfg.base_dir, "logs")
for p in (CACHE_DIR, RUNS_DIR, CURVES_DIR, PRED_DIR, CKPT_DIR, LOG_DIR):
    os.makedirs(p, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED0 = int(cfg.get("training.seed"))
TOPK_FRAC = float(cfg.get("training.topk_frac", 0.01))

# 主结果截断（§4.0）
DEFAULT_SMI = 128
DEFAULT_PROT = 1500

# 各基线原文（own）协议
OWN = {
    "deepdta":  dict(lr=1e-3, batch=512, max_epochs=100, patience=10),
    "graphdta": dict(lr=5e-4, batch=512, max_epochs=300, patience=30),  # 原文1000ep，算力截断
    "moltrans": dict(lr=1e-3, batch=64,  max_epochs=100, patience=10),  # 原文128，16GB降批
}
# 本文协议（paper）：与 05 严格同条件
_PAPER_BASE = dict(lr=2e-3, batch=512, max_epochs=15, patience=6)
PAPER = {"deepdta": dict(_PAPER_BASE), "graphdta": dict(_PAPER_BASE),
         "moltrans": dict(_PAPER_BASE, batch=64)}


def log(m):
    print(f"[15] {m}", flush=True)


# ============================================================ 工具
def set_seed(seed, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # 部分 scatter/index_add 在 CUDA 上没有确定性实现；warn_only 保证可运行，
        # A3 以 |Δ|<1e-4 容差验收并在报告中说明。
        torch.use_deterministic_algorithms(True, warn_only=True)


def n_params(model):
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def compute_metrics(y, p, topk_frac=TOPK_FRAC):
    pred = (p > 0.5).astype(int)
    k = max(1, int(topk_frac * len(y)))
    top = np.argsort(-p)[:k]
    return {
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "f1": round(float(f1_score(y, pred)), 4),
        "precision": round(float(precision_score(y, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y, pred, zero_division=0)), 4),
        "accuracy": round(float(accuracy_score(y, pred)), 4),
        "precision_at_topk": round(float(y[top].mean()), 4),
        "topk_k": int(k),
        "n_test": int(len(y)),
        "n_pos_test": int(y.sum()),
    }


# ============================================================ 数据缓存
def _load_ref():
    """positives/negatives → 唯一 (met,tar) → smiles/protein_seq。"""
    cols = ["met", "tar", "smiles", "protein_seq"]
    pos = pd.read_parquet(cfg.get_path("paths.positives"), columns=cols)
    neg = pd.read_parquet(cfg.get_path("paths.negatives"), columns=cols)
    ref = pd.concat([pos, neg], ignore_index=True).drop_duplicates(["met", "tar"])
    return ref


def _build_vocab(seqs, min_count=1):
    from collections import Counter
    c = Counter()
    for s in seqs:
        c.update(s)
    items = [(k, v) for k, v in c.items() if v >= min_count]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return {k: i + 2 for i, (k, _) in enumerate(items)}  # 0=pad, 1=unk


def _build_kmer_vocab(seqs, max_len, min_count=1):
    from collections import Counter
    c = Counter()
    for s in seqs:
        ss = str(s)[:max_len]
        c.update(ss[i:i + 3] for i in range(max(0, len(ss) - 2)))
    items = [(k, v) for k, v in c.items() if v >= min_count]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return {k: i + 2 for i, (k, _) in enumerate(items)}


def _tokenize_chars(s, vocab, max_len):
    ids = [vocab.get(ch, 1) for ch in str(s)[:max_len]]
    if len(ids) < 32:  # 三次 kernel=12/8/4 卷积的最小长度保护
        # 用 <unk>(1) 而非 pad(0) 填充：极短分子（单原子离子）也需要非空感受野；
        # 若填 0，mask 会把这些位置排除，三次卷积后为空 → -1e4 污染。
        ids = ids + [1] * (32 - len(ids))
    return ids


def _tokenize_kmers(seq, vocab, max_len_prot):
    n = max(0, max_len_prot - 2)
    seq = str(seq)[:n + 2]
    ids = [vocab.get(seq[i:i + 3], 1) for i in range(max(0, len(seq) - 2))]
    if len(ids) < 32:
        ids = ids + [1] * (32 - len(ids))
    return ids


def _build_graph(smiles):
    """RDKit 分子图 → (atom_feats [n,F], edge_index [2,2E])；零键分子保留单节点。"""
    from rdkit import Chem
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        mol = Chem.MolFromSmiles("C")
    if mol is None:
        mol = Chem.RWMol()
        mol.AddAtom(Chem.Atom(6))
        mol = mol.GetMol()
    elements = ["C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "Si", "B", "Se",
                "Na", "K", "Ca", "Mg", "Fe", "Zn", "Cu", "Mn", "Co", "Ni", "H"]
    eidx = {e: i for i, e in enumerate(elements)}
    feats = []
    for a in mol.GetAtoms():
        sym = a.GetSymbol()
        f = [0.0] * (len(elements) + 1 + 6 + 5 + 1 + 6 + 5 + 1)
        off = 0
        if sym in eidx:
            f[eidx[sym]] = 1.0
        else:
            f[len(elements)] = 1.0
        off = len(elements) + 1
        deg = min(int(a.GetDegree()), 5)
        f[off + deg] = 1.0
        off += 6
        nh = min(int(a.GetTotalNumHs()), 4)
        f[off + nh] = 1.0
        off += 5
        f[off] = 1.0 if a.GetIsAromatic() else 0.0
        off += 1
        hyb = str(a.GetHybridization())
        hyb_map = {"SP": 0, "SP2": 1, "SP3": 2, "SP3D": 3, "SP3D2": 4}
        f[off + hyb_map.get(hyb, 5)] = 1.0
        off += 6
        chg = max(-2, min(2, int(a.GetFormalCharge())))
        f[off + chg + 2] = 1.0
        off += 5
        f[off] = 1.0 if a.IsInRing() else 0.0
        feats.append(f)
    atom_feats = np.asarray(feats, dtype=np.float32)
    edges = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        edges.extend([[i, j], [j, i]])
    edge_index = np.asarray(edges, dtype=np.int64).T if edges else np.zeros((2, 0), dtype=np.int64)
    return atom_feats, edge_index


def build_cache(force=False):
    """构建内部数据 token / 图 / ESM 缓存（幂等）。"""
    meta_path = os.path.join(CACHE_DIR, "meta.json")
    need = ["tokens.npz", "vocab.json", "graphs.pkl", "esm_per_tar.npy"]
    if not force and all(os.path.exists(os.path.join(CACHE_DIR, f)) for f in need):
        log(f"cache 已存在: {CACHE_DIR}")
        return

    t0 = time.time()
    log("构建 SOTA 缓存（首次运行，约 2-4 分钟）...")
    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(FEAT_DIR, feat_name), allow_pickle=True)
    row_met = d["row_met"].astype(np.int64)
    row_tar = d["row_tar"].astype(np.int64)
    n_pos = int(len(d["dir_met"]))

    ref = _load_ref()
    log(f"  ref: {len(ref):,} unique pairs")
    key = pd.DataFrame({"met": row_met, "tar": row_tar})
    seq_df = key.merge(ref, on=["met", "tar"], how="left")
    assert seq_df["smiles"].notna().all(), "SMILES join 出现 NaN（join 逻辑错误）"
    assert seq_df["protein_seq"].notna().all(), "序列 join 出现 NaN（join 逻辑错误）"

    met_ids = np.unique(row_met)
    tar_ids = np.unique(row_tar)
    row_met_idx = np.searchsorted(met_ids, row_met).astype(np.int32)
    row_tar_idx = np.searchsorted(tar_ids, row_tar).astype(np.int32)

    met2smi = ref.drop_duplicates("met").set_index("met")["smiles"].to_dict()
    tar2seq = ref.drop_duplicates("tar").set_index("tar")["protein_seq"].to_dict()
    met_smiles = [met2smi[int(m)] for m in met_ids]
    tar_seqs = [tar2seq[int(t)] for t in tar_ids]

    smi_vocab = _build_vocab(met_smiles)
    prot_vocab = _build_vocab([s[:DEFAULT_PROT] for s in tar_seqs])
    kmer_vocab = _build_kmer_vocab(tar_seqs, DEFAULT_PROT, min_count=1)

    max_smi_len = max(len(s) for s in met_smiles)
    log(f"  vocab: smi={len(smi_vocab)}+2, prot={len(prot_vocab)}+2, "
        f"3mer={len(kmer_vocab)}+2; smiles max={max_smi_len}")

    smi_tokens = np.zeros((len(met_ids), max_smi_len), dtype=np.int16)
    prot_tokens = np.zeros((len(tar_ids), DEFAULT_PROT), dtype=np.int16)
    kmer_tokens = np.zeros((len(tar_ids), DEFAULT_PROT), dtype=np.int16)
    for i, s in enumerate(met_smiles):
        ids = [smi_vocab.get(ch, 1) for ch in s]
        if len(ids) < 32:  # 与 _tokenize_chars 一致：三次卷积最小长度保护
            ids = ids + [1] * (32 - len(ids))
        smi_tokens[i, :len(ids)] = np.asarray(ids, dtype=np.int16)
    for i, s in enumerate(tar_seqs):
        ids = [prot_vocab.get(ch, 1) for ch in s[:DEFAULT_PROT]]
        prot_tokens[i, :len(ids)] = np.asarray(ids, dtype=np.int16)
        kk = _tokenize_kmers(s, kmer_vocab, DEFAULT_PROT)
        kmer_tokens[i, :len(kk)] = np.asarray(kk[:DEFAULT_PROT], dtype=np.int16)
    np.savez_compressed(os.path.join(CACHE_DIR, "tokens.npz"),
                        met_ids=met_ids, tar_ids=tar_ids,
                        row_met_idx=row_met_idx, row_tar_idx=row_tar_idx,
                        smi_tokens=smi_tokens, prot_tokens=prot_tokens,
                        kmer_tokens=kmer_tokens, n_pos=n_pos)
    with open(os.path.join(CACHE_DIR, "vocab.json"), "w") as f:
        json.dump({"smi": smi_vocab, "prot": prot_vocab, "kmer": kmer_vocab}, f)

    # ---- RDKit graphs ----
    graphs = {}
    for i, s in enumerate(met_smiles):
        graphs[i] = _build_graph(s)
    with open(os.path.join(CACHE_DIR, "graphs.pkl"), "wb") as f:
        pickle.dump(graphs, f, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"  graphs: {len(graphs):,} molecules")

    # ---- 冻结 ESM-2 3B（F6）----
    esm_dir = cfg.get_path("paths.esm_cache_dir")
    emb = np.load(os.path.join(esm_dir, "embeddings.npy"), mmap_mode="r")
    ids = json.load(open(os.path.join(esm_dir, "ids.json")))
    seq2idx = {r["seq"]: int(r["index"]) for r in ids}
    esm = np.zeros((len(tar_ids), emb.shape[1]), dtype=np.float32)
    miss = 0
    for i, s in enumerate(tar_seqs):
        j = seq2idx.get(s)
        if j is None:
            miss += 1
        else:
            esm[i] = emb[j]
    np.save(os.path.join(CACHE_DIR, "esm_per_tar.npy"), esm)
    log(f"  ESM per target: {esm.shape}; missing={miss}")

    with open(meta_path, "w") as f:
        json.dump({"n_met": len(met_ids), "n_tar": len(tar_ids),
                   "n_pos": n_pos, "max_smi_len": max_smi_len,
                   "smiles_max_len_used": DEFAULT_SMI,
                   "protein_max_len_used": DEFAULT_PROT,
                   "vocab_smi": len(smi_vocab) + 2,
                   "vocab_prot": len(prot_vocab) + 2,
                   "vocab_kmer": len(kmer_vocab) + 2,
                   "esm_missing": miss,
                   "built_seconds": round(time.time() - t0, 1)}, f, indent=2)
    log(f"缓存完成: {time.time() - t0:.1f}s")


class TokenData:
    """内部行索引 → token / 图 / ESM 张量（按 batch 取，避免整矩阵上显存）。"""

    def __init__(self, cache_dir=CACHE_DIR):
        z = np.load(os.path.join(cache_dir, "tokens.npz"))
        self.met_ids = z["met_ids"]
        self.tar_ids = z["tar_ids"]
        self.row_met = z["row_met_idx"]
        self.row_tar = z["row_tar_idx"]
        self.smi = z["smi_tokens"]
        self.prot = z["prot_tokens"]
        self.kmer = z["kmer_tokens"]
        self.n_pos = int(z["n_pos"])
        self.vocab = json.load(open(os.path.join(cache_dir, "vocab.json")))
        self.esm = np.load(os.path.join(cache_dir, "esm_per_tar.npy"), mmap_mode="r")
        with open(os.path.join(cache_dir, "graphs.pkl"), "rb") as f:
            self.graphs = pickle.load(f)

    def batch(self, rows, model_name, max_smi, max_prot, protein_mode="native"):
        mi = self.row_met[rows]
        ti = self.row_tar[rows]
        smi = torch.from_numpy(self.smi[mi, :max_smi].astype(np.int64))
        if model_name == "moltrans":
            prot = torch.from_numpy(self.kmer[ti, :max(32, max_prot - 2)].astype(np.int64))
        else:
            prot = torch.from_numpy(self.prot[ti, :max_prot].astype(np.int64))
        if protein_mode == "esm_frozen":
            esm = torch.from_numpy(np.asarray(self.esm[ti], dtype=np.float32))
            return smi, esm
        return smi, prot

    def graphs_from(self, rows):
        mi = self.row_met[rows]
        return [self.graphs[int(i)] for i in mi]


# ============================================================ 模型
class ConvBranch(nn.Module):
    """DeepDTA / GraphDTA 共用的 1D CNN 蛋白分支。"""

    def __init__(self, vocab, emb=128, kernels=(4, 8, 12), channels=(32, 64, 96)):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        convs = []
        in_c = emb
        for k, out_c in zip(kernels, channels):
            convs.append(nn.Conv1d(in_c, out_c, k))
            in_c = out_c
        self.convs = nn.ModuleList(convs)
        self.out_dim = in_c

    def forward(self, tokens):
        mask = (tokens != 0).unsqueeze(1)
        x = self.emb(tokens).transpose(1, 2)
        for conv in self.convs:
            x = F.relu(conv(x))
            k = conv.kernel_size[0]
            mask = mask[:, :, k - 1:]
            x = x * mask
        x = x.masked_fill(~mask, -1e4)
        return x.amax(dim=-1)


class DeepDTA(nn.Module):
    """DeepDTA (Öztürk 2018) 二分类适配版。

    偏离：回归头 → 1 logit + BCE；FC 采用原文 1024→1024→512；
    max_len 按本数据选 128/1500（§4.0）。
    """

    def __init__(self, n_smi, n_prot, protein_mode="native", prot_dim=2560,
                 emb=128, fc_hidden=1024, dropout=0.1):
        super().__init__()
        self.protein_mode = protein_mode
        self.smi_branch = ConvBranch(n_smi, emb=emb)
        if protein_mode == "native":
            self.prot_branch = ConvBranch(n_prot, emb=emb)
            prot_out = self.prot_branch.out_dim
        else:
            self.prot_proj = nn.Linear(prot_dim, 1024)
            prot_out = 1024
        self.fc = nn.Sequential(
            nn.Linear(self.smi_branch.out_dim + prot_out, fc_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_hidden, fc_hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(fc_hidden, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1))

    def forward(self, smi, prot):
        m = self.smi_branch(smi)
        p = self.prot_branch(prot) if self.protein_mode == "native" else \
            F.relu(self.prot_proj(prot))
        return self.fc(torch.cat([m, p], dim=1)).squeeze(-1)


class GINLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden))
        self.bn = nn.BatchNorm1d(hidden)
        self.eps = nn.Parameter(torch.zeros(1))

    def forward(self, x, edge_index):
        if edge_index.shape[1] > 0:
            row, col = edge_index
            agg = torch.zeros_like(x).index_add_(0, row, x[col])
        else:
            agg = torch.zeros_like(x)
        return self.bn(self.mlp((1.0 + self.eps) * x + agg))


class GraphDTA(nn.Module):
    """GraphDTA (Nguyen 2021) GIN 变体，纯 PyTorch 实现（无 PyG 依赖）。"""

    def __init__(self, n_prot, protein_mode="native", prot_dim=2560,
                 atom_dim=48, hidden=128, n_layers=4, dropout=0.1):
        super().__init__()
        self.protein_mode = protein_mode
        self.atom_proj = nn.Linear(atom_dim, hidden)
        self.gins = nn.ModuleList([GINLayer(hidden) for _ in range(n_layers)])
        self.mol_out = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU())
        if protein_mode == "native":
            self.prot_branch = ConvBranch(n_prot)
            prot_out = self.prot_branch.out_dim
        else:
            self.prot_proj = nn.Linear(prot_dim, 1024)
            prot_out = 1024
        self.fc = nn.Sequential(
            nn.Linear(hidden + prot_out, 1024), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(1024, 512), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(512, 1))

    def encode_graph(self, x, edge_index, graph_batch, n_graphs):
        h = self.atom_proj(x)
        for gin in self.gins:
            h = gin(h, edge_index)
        sum_p = h.new_zeros(n_graphs, h.shape[1]).index_add_(
            0, graph_batch, h)
        cnt = h.new_zeros(n_graphs).index_add_(
            0, graph_batch, h.new_ones(h.shape[0]))
        mean_p = sum_p / cnt.unsqueeze(1).clamp_min(1)
        return self.mol_out(torch.cat([sum_p, mean_p], dim=1))

    def forward(self, graphs, prot):
        atom_feats, edge_index, graph_batch, n_graphs = graphs
        m = self.encode_graph(atom_feats, edge_index, graph_batch, n_graphs)
        p = self.prot_branch(prot) if self.protein_mode == "native" else \
            F.relu(self.prot_proj(prot))
        return self.fc(torch.cat([m, p], dim=1)).squeeze(-1)


class MolTrans(nn.Module):
    """MolTrans (Huang 2021) 二分类适配版。

    保持 2 层 / 12 头 / emb 384 与“外积 → 2D CNN”交互结构。
    偏离（报告逐条披露）：
      * SMILES 用字符级 token（原文 ESPF/BPE 子词，词表 23,532）；
      * 蛋白 3-mer（原文 3-mer，词表 16,693；本数据重建词表）；
      * 外积前将 384 维投影到 inter_dim（原文直接 384 通道外积，
        在 max_len 128×1500 下显存不可行）；
      * 交互 CNN 用 stride-2 两层 + 全局 max pool。
    """

    def __init__(self, n_smi, n_kmer, d=384, heads=12, layers=2,
                 max_smi=128, max_prot=1500, inter_dim=32, dropout=0.1,
                 window=512):
        super().__init__()
        self.window = window
        self.smi_emb = nn.Embedding(n_smi, d, padding_idx=0)
        self.prot_emb = nn.Embedding(n_kmer, d, padding_idx=0)
        self.smi_pos = nn.Embedding(max_smi, d)
        self.prot_pos = nn.Embedding(max_prot, d)
        layer1 = nn.TransformerEncoderLayer(d, heads, 384, dropout=dropout,
                                            batch_first=True)
        layer2 = nn.TransformerEncoderLayer(d, heads, 384, dropout=dropout,
                                            batch_first=True)
        self.smi_enc = nn.TransformerEncoder(layer1, layers, enable_nested_tensor=False)
        self.prot_enc = nn.TransformerEncoder(layer2, layers, enable_nested_tensor=False)
        self.smi_proj = nn.Linear(d, inter_dim)
        self.prot_proj = nn.Linear(d, inter_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(inter_dim, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU())
        # 原 MolTrans 的交互 CNN 后是 flatten + FC；全图 flatten 在 128×1498 下不可行，
        # 用 4×4 adaptive average pool 保留空间格局（比 global max 的梯度更密）。
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * 4 * 4, 512), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(512, 1))

    def _encode(self, enc, x, mask):
        """窗口化 Transformer：长序列按 window 分段编码后拼接（不截断）。

        偏离说明：原文 MolTrans 对 max_target_seq=545 一次性编码；本数据
        max_len 1500 时 1500² attention 在 16 GB 上不可行。窗口 512 保留
        2 层/12 头/384 维与全序列覆盖，只限制单窗口内的长程依赖。
        """
        L = x.shape[1]
        if L <= self.window:
            return enc(x, src_key_padding_mask=mask)
        outs = []
        for s in range(0, L, self.window):
            xc = x[:, s:s + self.window]
            mc = mask[:, s:s + self.window]
            if bool(mc.all()):
                # 整个 batch 在该窗口全是 padding：保留长度、输出零，不能直接跳过，
                # 否则返回序列变短，与外层 mask 长度不一致。
                outs.append(x.new_zeros((x.shape[0], xc.shape[1], x.shape[2])))
                continue
            if xc.shape[1] < self.window:
                pad = self.window - xc.shape[1]
                xc = F.pad(xc, (0, 0, 0, pad))
                mc = torch.cat([mc, mc.new_ones(mc.shape[0], pad, dtype=torch.bool)], dim=1)
            # ★ 关键：某些行在该窗口内可能全是 padding。若 src_key_padding_mask
            # 整行全 True，softmax 对全 -inf 归一 → NaN。这里临时放开第一个位置，
            # 并在输出中把该行的这个窗口清零。
            all_pad = mc.all(dim=1)
            if bool(all_pad.any()):
                mc = mc.clone()
                mc[all_pad, 0] = False
            oc = enc(xc, src_key_padding_mask=mc)
            if bool(all_pad.any()):
                oc = oc * (~all_pad).float()[:, None, None]
            outs.append(oc)
        if not outs:
            return x.new_zeros(x.shape)
        return torch.cat(outs, dim=1)[:, :L]

    def forward(self, smi, prot):
        smi_mask = smi == 0
        prot_mask = prot == 0
        ds = self.smi_emb(smi) + self.smi_pos(
            torch.arange(smi.shape[1], device=smi.device).unsqueeze(0))
        ts = self.prot_emb(prot) + self.prot_pos(
            torch.arange(prot.shape[1], device=prot.device).unsqueeze(0))
        ds = self._encode(self.smi_enc, ds, smi_mask)
        ts = self._encode(self.prot_enc, ts, prot_mask)
        # ★ 交互图只保留有效 token：padding 位置在 Transformer 后仍有非零输出，
        # 若不清零，外积图的池化会被 padding 噪声主导（实测 loss 不降）。
        dp = self.smi_proj(ds) * (~smi_mask).unsqueeze(-1).to(ds.dtype)
        tp = self.prot_proj(ts) * (~prot_mask).unsqueeze(-1).to(ts.dtype)
        x = torch.einsum("bik,bjk->bijk", dp, tp).permute(0, 3, 1, 2)
        x = self.pool(self.cnn(x))
        return self.fc(x).squeeze(-1)


# ============================================================ 图 batch
def collate_graphs(graph_list, device):
    xs, eis, batches = [], [], []
    off = 0
    for i, (af, ei) in enumerate(graph_list):
        xs.append(af)
        eis.append(ei + off)
        batches.append(np.full(len(af), i, dtype=np.int64))
        off += len(af)
    x = torch.from_numpy(np.concatenate(xs, axis=0)).float().to(device)
    ei = torch.from_numpy(np.concatenate(eis, axis=1) if eis else np.zeros((2, 0), np.int64)).long().to(device)
    gb = torch.from_numpy(np.concatenate(batches, axis=0)).long().to(device)
    return x, ei, gb, len(graph_list)


# ============================================================ 划分
def get_split(name, data, seed):
    feat_name = cfg.get("training.feat_file_map")["esm"]
    s = np.load(cfg.sidecar_npz("split", feat_name))
    if name == "row":
        return s["b_tr"], s["b_va"], s["b_te"], None
    if name == "strict":
        sp = strict_split.load_or_build(FEAT_DIR, feat_name,
                                        row_met=data.met_ids[data.row_met],
                                        row_tar=data.tar_ids[data.row_tar],
                                        n_pos=data.n_pos, seed=SEED0)
        return sp["b_tr"], sp["b_va"], sp["b_te"], None
    # cold-*: 复用 14 号实现；每 seed 新划分（论文口径）
    spec = importlib.util.spec_from_file_location(
        "u14", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "14_uncertainty_stats.py"))
    m14 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m14)
    rng = np.random.RandomState(SEED0 + int(seed))
    met = data.met_ids[data.row_met]
    tar = data.tar_ids[data.row_tar]
    tr, va, te, drop = m14.cold_split_indices(met, tar, name, rng)
    return tr, va, te, drop


# ============================================================ 训练 / 预测
def make_model(model_name, data, args):
    vs = data.vocab
    if model_name == "deepdta":
        return DeepDTA(len(vs["smi"]) + 2, len(vs["prot"]) + 2,
                       protein_mode=args.protein).to(DEVICE)
    if model_name == "graphdta":
        return GraphDTA(len(vs["prot"]) + 2, protein_mode=args.protein).to(DEVICE)
    if model_name == "moltrans":
        return MolTrans(len(vs["smi"]) + 2, len(vs["kmer"]) + 2,
                        max_smi=args.max_len_smi,
                        max_prot=max(32, args.max_len_prot - 2)).to(DEVICE)
    raise ValueError(model_name)


def predict(model, model_name, data, rows, args, bs):
    model.eval()
    # 显存上限：MolTrans 外积交互最重，DeepDTA 次之
    cap = {"moltrans": 64, "deepdta": 1024, "graphdta": 512}[model_name]
    if model_name == "moltrans":
        cap = min(cap, int(getattr(args, "batch_size", cap)))
    bs = min(int(bs), cap)
    outs = []
    with torch.no_grad():
        for i in range(0, len(rows), bs):
            r = rows[i:i + bs]
            if model_name == "graphdta":
                g = collate_graphs(data.graphs_from(r), DEVICE)
                _, prot = data.batch(r, model_name, args.max_len_smi,
                                     args.max_len_prot, args.protein)
                prot = prot.to(DEVICE)
                logits = model(g, prot)
            else:
                smi, prot = data.batch(r, model_name, args.max_len_smi,
                                       args.max_len_prot, args.protein)
                smi = (tuple(x.to(DEVICE) for x in smi)
                       if isinstance(smi, tuple) else smi.to(DEVICE))
                prot = (tuple(x.to(DEVICE) for x in prot)
                        if isinstance(prot, tuple) else prot.to(DEVICE))
                logits = model(smi, prot)
            outs.append(torch.sigmoid(logits).float().cpu().numpy())
    return np.concatenate(outs)


def train_one(model, model_name, data, tr, va, y, args):
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss()
    use_amp = bool(getattr(args, "amp", "fp16") != "none" and DEVICE == "cuda")
    amp_dtype = torch.bfloat16 if getattr(args, "amp", "fp16") == "bf16" else torch.float16
    # GradScaler 只对 fp16 必要；bf16/fp32 直接 backward
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and args.amp == "fp16"))
    rng = np.random.RandomState(args.seed)
    best_mon, best_state, best_ep, bad = -1.0, None, 0, 0
    init_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    nan_batches = 0
    curves = {"train_loss": [], "val_auc": [], "val_acc": []}
    bs = args.batch_size
    t0 = time.time()
    for ep in range(args.max_epochs):
        model.train()
        perm = rng.permutation(len(tr))
        losses = []
        for i in range(0, len(perm), bs):
            rows = tr[perm[i:i + bs]]
            if model_name == "graphdta":
                g = collate_graphs(data.graphs_from(rows), DEVICE)
                _, prot = data.batch(rows, model_name, args.max_len_smi,
                                     args.max_len_prot, args.protein)
                prot = prot.to(DEVICE)
                smi = None
            else:
                smi, prot = data.batch(rows, model_name, args.max_len_smi,
                                       args.max_len_prot, args.protein)
                smi = (tuple(x.to(DEVICE) for x in smi)
                       if isinstance(smi, tuple) else smi.to(DEVICE))
                prot = (tuple(x.to(DEVICE) for x in prot)
                        if isinstance(prot, tuple) else prot.to(DEVICE))
            yb = torch.from_numpy(y[rows]).float().to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                logits = model(g, prot) if model_name == "graphdta" else model(smi, prot)
                loss = crit(logits, yb)
            if not torch.isfinite(loss):
                nan_batches += 1
                log(f"    WARN non-finite loss (batch {i//bs}, total {nan_batches}); skip")
                opt.zero_grad(set_to_none=True)
                if nan_batches >= 5:
                    log("    ERROR 连续/累计 5 个非有限 loss；回滚到 best/initial state 并停止本 run")
                    model.load_state_dict(best_state if best_state is not None else init_state)
                    break
                continue
            nan_batches = 0
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if torch.isfinite(gn):
                    scaler.step(opt)
                    scaler.update()
                else:
                    log("    WARN non-finite grad; skip step")
                    opt.zero_grad(set_to_none=True)
                    scaler.update()  # 必须 update，否则下一次 unscale_ 会报 already called
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if torch.isfinite(gn):
                    opt.step()
                else:
                    log("    WARN non-finite grad; skip step")
                    opt.zero_grad(set_to_none=True)
            losses.append(float(loss.detach()))
        if not losses:
            log("    ERROR 本 epoch 无有效 batch；停止")
            model.load_state_dict(best_state if best_state is not None else init_state)
            break
        p_va = predict(model, model_name, data, va, args, bs=max(256, bs))
        if not np.isfinite(p_va).all():
            log("    ERROR 验证预测含 NaN/Inf；回滚到 best/initial state 并停止本 run")
            model.load_state_dict(best_state if best_state is not None else init_state)
            break
        va_auc = float(roc_auc_score(y[va], p_va))
        va_acc = float(((p_va > 0.5).astype(int) == y[va]).mean())
        curves["train_loss"].append(float(np.mean(losses)))
        curves["val_auc"].append(va_auc)
        curves["val_acc"].append(va_acc)
        mon = va_auc if args.protocol == "own" else va_acc
        if mon > best_mon:
            best_mon, bad, best_ep = mon, 0, ep + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        log(f"    ep {ep+1:03d}/{args.max_epochs} loss={curves['train_loss'][-1]:.4f} "
            f"val_auc={va_auc:.4f} val_acc={va_acc:.4f} best_ep={best_ep} bad={bad}")
        if bad >= args.patience:
            log(f"    early stop at epoch {ep+1} (patience {args.patience})")
            break
    train_seconds = time.time() - t0
    model.load_state_dict(best_state if best_state is not None else init_state)
    return model, {
        "train_seconds": round(train_seconds, 1),
        "epochs_run": len(curves["train_loss"]),
        "best_epoch": int(best_ep),
        "best_monitor": round(float(best_mon), 4),
    }, curves


# ============================================================ 单 run
def run_tag(args):
    """主 run 用默认 tag；消融/敏感性加后缀，禁止覆盖主结果。"""
    tag = f"{args.model}_{args.split}_s{args.seed}_{args.protocol}"
    if getattr(args, "protein", "native") != "native":
        tag += "_esm"
    if (int(args.max_len_smi) != DEFAULT_SMI) or (int(args.max_len_prot) != DEFAULT_PROT):
        tag += f"_L{int(args.max_len_smi)}x{int(args.max_len_prot)}"
    return tag


def run(args):
    t_all = time.time()
    set_seed(args.seed)
    build_cache()
    data = TokenData()
    y = np.load(cfg.sidecar_npz("labels", cfg.get("training.feat_file_map")["esm"]))["y_binary"]
    y = y.astype(np.int64)
    tr, va, te, drop = get_split(args.split, data, args.seed)
    log(f"split={args.split} seed={args.seed}: train {len(tr):,} / val {len(va):,} / "
        f"test {len(te):,}" + (f" / drop {drop:.1%}" if drop is not None else ""))

    model = make_model(args.model, data, args)
    log(f"model={args.model} params={n_params(model):,} "
        f"protein={args.protein} max_len={args.max_len_smi}/{args.max_len_prot}")
    model, info, curves = train_one(model, args.model, data, tr, va, y, args)

    p_te = predict(model, args.model, data, te, args, bs=max(256, args.batch_size))
    y_te = y[te]
    metrics = compute_metrics(y_te, p_te)
    tag = run_tag(args)
    out = {
        "model": args.model, "split": args.split, "seed": int(args.seed),
        "protocol": args.protocol, "protein": args.protein,
        "max_len_smi": int(args.max_len_smi), "max_len_prot": int(args.max_len_prot),
        "n_params": n_params(model), "batch_size": int(args.batch_size),
        "lr": float(args.lr), "patience": int(args.patience),
        "max_epochs": int(args.max_epochs),
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "drop_rate": float(drop) if drop is not None else 0.0,
        **info, **metrics,
        "wall_seconds": round(time.time() - t_all, 1),
    }
    with open(os.path.join(RUNS_DIR, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(os.path.join(PRED_DIR, f"{tag}.npz"),
                        y=y_te, p=p_te, rows=te)
    np.savez_compressed(os.path.join(CURVES_DIR, f"{tag}.npz"),
                        train_loss=np.asarray(curves["train_loss"]),
                        val_auc=np.asarray(curves["val_auc"]),
                        val_acc=np.asarray(curves["val_acc"]),
                        best_epoch=np.asarray([info["best_epoch"]]),
                        best_monitor=np.asarray([info["best_monitor"]]),
                        model=np.asarray([args.model]), split=np.asarray([args.split]),
                        seed=np.asarray([args.seed]), protocol=np.asarray([args.protocol]))
    torch.save({"state_dict": model.state_dict(), "meta": out},
               os.path.join(CKPT_DIR, f"{tag}.pt"))
    log(f"RESULT {tag}: ROC-AUC {metrics['roc_auc']:.4f} PR-AUC {metrics['pr_auc']:.4f} "
        f"F1 {metrics['f1']:.4f} P@1% {metrics['precision_at_topk']:.4f} "
        f"({info['train_seconds']:.0f}s, {info['epochs_run']} ep, best {info['best_epoch']})")
    log(f"saved {os.path.join(RUNS_DIR, f'{tag}.json')}")
    return out


# ============================================================ 外部五源
def _load_fasta():
    """本地 reviewed FASTA → {acc: seq}（带 pickle 缓存）。"""
    cache = os.path.join(CACHE_DIR, "fasta.pkl")
    if os.path.exists(cache):
        with open(cache, "rb") as f:
            return pickle.load(f)
    import gzip
    acc2seq, acc, buf = {}, None, []
    with gzip.open(cfg.get_path("paths.uniprot_fasta"), "rt") as f:
        for line in f:
            if line.startswith(">"):
                if acc:
                    acc2seq[acc] = "".join(buf)
                parts = line[1:].split("|")
                acc = parts[1] if len(parts) >= 2 else None
                buf = []
            elif acc:
                buf.append(line.strip())
    if acc:
        acc2seq[acc] = "".join(buf)
    with open(cache, "wb") as f:
        pickle.dump(acc2seq, f, protocol=pickle.HIGHEST_PROTOCOL)
    return acc2seq


def _pubchem_smiles():
    p = os.path.join(cfg.get_path("paths.external_dir"), "pubchem_smiles_ext.json")
    return json.load(open(p)) if os.path.exists(p) else {}


def _external_frames():
    ext = cfg.get_path("paths.external_dir")
    udir = os.path.join(ext, "unified_ext")
    frames = {}
    chem = os.path.join(ext, "chembl_ext_dataset.parquet")
    if os.path.exists(chem):
        frames["chembl"] = pd.read_parquet(chem)
    for src in ("hmdb_assoc", "pdb", "drugbank"):
        p = os.path.join(udir, f"{src}_pairs.parquet")
        if os.path.exists(p):
            frames[src] = pd.read_parquet(p)
    return frames


def score_external(args):
    """F9：只推理，不重训。使用 row 划分 seed 42 的 own 协议 checkpoint。"""
    tag = f"{args.model}_row_s{SEED0}_own"
    ck_path = os.path.join(CKPT_DIR, f"{tag}.pt")
    if not os.path.exists(ck_path):
        raise SystemExit(f"缺少 checkpoint: {ck_path}；请先训练 row seed 42 own")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    args.max_len_smi = meta["max_len_smi"]
    args.max_len_prot = meta["max_len_prot"]
    args.protein = meta["protein"]
    if args.protein != "native":
        raise SystemExit("外部五源打分使用 native checkpoint（F9）；esm_frozen 仅用于 F6 消融")
    build_cache()
    data = TokenData()
    model = make_model(args.model, data, args)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    fa = _load_fasta()
    smi_cache = _pubchem_smiles()
    frames = _external_frames()
    out = {"model": args.model, "seed": int(SEED0), "protocol": "own",
           "checkpoint": ck_path, "sources": {}}

    def smiles_of(df, is_chembl):
        if is_chembl and "canonical_smiles" in df.columns:
            vals = df["canonical_smiles"].tolist()
        else:
            vals = [None] * len(df)
        cids = df["metabolite_pubchem_cid"].astype("int64").astype(str).tolist()
        return [v if isinstance(v, str) else smi_cache.get(c) for v, c in zip(vals, cids)]

    def _score_subset(d, smis, seqs, ii, valid):
        """对给定行子集 ii 做 token/图编码与打分。"""
        smi_ids = np.zeros((len(ii), args.max_len_smi), dtype=np.int64)
        prot_ids = np.zeros((len(ii), max(32, args.max_len_prot - 2)
                             if args.model == "moltrans" else args.max_len_prot),
                            dtype=np.int64)
        for j, k in enumerate(ii):
            s = smis[k]
            if args.model == "moltrans":
                ids = _tokenize_kmers(seqs[k], data.vocab["kmer"], args.max_len_prot)
                prot_ids[j, :len(ids)] = ids[:prot_ids.shape[1]]
            else:
                ids = _tokenize_chars(seqs[k], data.vocab["prot"], args.max_len_prot)
                prot_ids[j, :len(ids)] = ids[:prot_ids.shape[1]]
            sid = _tokenize_chars(s, data.vocab["smi"], args.max_len_smi)
            smi_ids[j, :len(sid)] = sid[:args.max_len_smi]
        yy = d["is_active"].values[ii].astype(int)
        ps = []
        bs = 64 if args.model == "moltrans" else (512 if args.model == "graphdta" else 2048)
        with torch.no_grad():
            for i in range(0, len(ii), bs):
                smi_t = torch.from_numpy(smi_ids[i:i + bs]).to(DEVICE)
                prot_t = torch.from_numpy(prot_ids[i:i + bs]).to(DEVICE)
                if args.model == "graphdta":
                    # 外部按 SMILES 重建图（内部图缓存只覆盖内部实体）
                    gs = [_build_graph(smis[k]) for k in ii[i:i + bs]]
                    g = collate_graphs(gs, DEVICE)
                    logits = model(g, prot_t)
                else:
                    logits = model(smi_t, prot_t)
                ps.append(torch.sigmoid(logits).float().cpu().numpy())
        p = np.concatenate(ps)
        r = compute_metrics(yy, p)
        r.update({"n_scored": int(len(ii)), "coverage": round(float(valid.mean()), 4),
                  "n_rows": int(len(d)), "smiles_cov": round(float(np.mean(
                      [s is not None for s in smis])), 4),
                  "seq_cov": round(float(np.mean([q is not None for q in seqs])), 4)})
        return r

    esm_seqset = None
    if os.path.exists(os.path.join(cfg.get_path("paths.esm_cache_dir"), "ids.json")):
        _ids = json.load(open(os.path.join(cfg.get_path("paths.esm_cache_dir"), "ids.json")))
        esm_seqset = {r["seq"] for r in _ids}

    def score_frame(name, df, chemistry=False):
        d = df.reset_index(drop=True)
        smis = smiles_of(d, chemistry)
        seqs = [fa.get(str(u)) for u in d["target_uniprot"].tolist()]
        valid = np.array([s is not None and q is not None for s, q in zip(smis, seqs)])
        ii = np.where(valid)[0]
        if len(ii) == 0:
            return {"n_scored": 0}
        r = _score_subset(d, smis, seqs, ii, valid)
        # 与主模型 complete-only 完全同口径：再限制到 ESM 缓存覆盖的序列
        if esm_seqset is not None:
            esm_ok = np.array([q in esm_seqset for q in seqs])
            ii2 = np.where(valid & esm_ok)[0]
            if len(ii2) > 20 and len(np.unique(d["is_active"].values[ii2])) > 1:
                r2 = _score_subset(d, smis, seqs, ii2, valid & esm_ok)
                r2["coverage_of_all"] = round(float(len(ii2) / len(d)), 4)
                r["esm_subset"] = r2
        return r

    if "chembl" in frames:
        out["sources"]["chembl"] = score_frame("chembl", frames["chembl"], chemistry=True)
        log(f"  chembl: {out['sources']['chembl']}")
    if "hmdb_assoc" in frames:
        hm = frames["hmdb_assoc"]
        a = hm[hm["is_active"] == 1]
        n0 = hm[hm["is_active"] == 0]
        if len(a) > len(n0):
            a = a.sample(len(n0), random_state=SEED0)
        hm = pd.concat([a, n0], ignore_index=True)
        out["sources"]["hmdb_assoc"] = score_frame("hmdb_assoc", hm)
        log(f"  hmdb_assoc: {out['sources']['hmdb_assoc']}")
    if "pdb" in frames:
        out["sources"]["pdb"] = score_frame("pdb", frames["pdb"])
        log(f"  pdb: {out['sources']['pdb']}")
    if "drugbank" in frames:
        db = frames["drugbank"]
        out["sources"]["drugbank"] = score_frame("drugbank", db)
        sub = db[db["in_hmdb"] == True] if "in_hmdb" in db.columns else db.iloc[0:0]
        # 与 step10 完全同口径：正例下采样到与负例 1:1
        sa = sub[sub["is_active"] == 1]
        sn = sub[sub["is_active"] == 0]
        if len(sa) > len(sn):
            sa = sa.sample(len(sn), random_state=SEED0)
        sub = pd.concat([sa, sn], ignore_index=True)
        out["sources"]["drugbank_in_hmdb"] = score_frame("drugbank_in_hmdb", sub)
        log(f"  drugbank: {out['sources']['drugbank']}")

    # 主模型参考（complete-only 口径，见 step7/step10 JSON）
    ref_path = os.path.join(RES_DIR, "sota_main_model_external_reference.json")
    if os.path.exists(ref_path):
        out["main_model_reference"] = json.load(open(ref_path))

    path = os.path.join(RES_DIR, "sota_external_metrics.json")
    allres = json.load(open(path)) if os.path.exists(path) else {}
    allres[args.model] = out
    with open(path, "w") as f:
        json.dump(allres, f, indent=2)
    log(f"saved {path}")


# ============================================================ 汇总
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2


def delong_paired(y, p1, p2):
    from scipy.stats import norm
    y = np.asarray(y).astype(int)
    pos, neg = y == 1, y == 0
    m, n = int(pos.sum()), int(neg.sum())
    if m == 0 or n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    tz1, tz2 = _midrank(p1), _midrank(p2)
    tx1, tx2 = _midrank(p1[pos]), _midrank(p2[pos])
    ty1, ty2 = _midrank(p1[neg]), _midrank(p2[neg])
    auc1 = (tz1[pos].sum() - m * (m + 1) / 2) / (m * n)
    auc2 = (tz2[pos].sum() - m * (m + 1) / 2) / (m * n)
    # DeLong placement values: (tz - tx) = n * V10（正样本），(tz - ty) = m * V01（负样本）
    d1p, d2p = tz1[pos] - tx1, tz2[pos] - tx2
    d1n, d2n = tz1[neg] - ty1, tz2[neg] - ty2
    s10_1 = float(np.var(d1p, ddof=1)) / (n ** 2)
    s01_1 = float(np.var(d1n, ddof=1)) / (m ** 2)
    s10_2 = float(np.var(d2p, ddof=1)) / (n ** 2)
    s01_2 = float(np.var(d2n, ddof=1)) / (m ** 2)
    cov10 = float(np.cov(d1p, d2p, ddof=1)[0, 1]) / (n ** 2)
    cov01 = float(np.cov(d1n, d2n, ddof=1)[0, 1]) / (m ** 2)
    var = (s10_1 / m + s01_1 / n + s10_2 / m + s01_2 / n
           - 2 * (cov10 / m + cov01 / n))
    se = np.sqrt(max(var, 1e-18))
    z = (auc1 - auc2) / se if se > 0 else 0.0
    p = float(2 * (1 - norm.cdf(abs(z))))
    return float(auc1), float(auc2), float(z), p


def summarize(args):
    import glob
    import statistics
    runs = []
    for p in sorted(glob.glob(os.path.join(RUNS_DIR, "*.json"))):
        if os.path.basename(p).endswith("_smoke.json"):
            continue
        r = json.load(open(p))
        if r.get("model") not in ("deepdta", "graphdta", "moltrans"):
            continue
        runs.append(r)
    if not runs:
        raise SystemExit("没有 sota_runs/*.json，无法汇总")
    summary = {"runs": runs, "aggregate": {}}
    for model in ("deepdta", "graphdta", "moltrans"):
        for split in ("row", "strict", "cold_drug", "cold_target", "dual_cold"):
            for prot in ("own", "paper"):
                sel = [r for r in runs if r["model"] == model and r["split"] == split
                       and r["protocol"] == prot and r["protein"] == "native"
                       and r.get("max_len_smi", 128) == 128
                       and r.get("max_len_prot", 1500) == 1500]
                if not sel:
                    continue
                key = f"{model}|{split}|{prot}"
                agg = {}
                for m in ("roc_auc", "pr_auc", "f1", "precision_at_topk",
                          "train_seconds", "epochs_run", "best_epoch", "n_params"):
                    vals = [r[m] for r in sel]
                    agg[m] = {"mean": round(float(statistics.mean(vals)), 4),
                              "std": round(float(statistics.stdev(vals)) if len(vals) > 1 else 0.0, 4),
                              "values": [round(float(v), 4) for v in vals]}
                agg["seeds"] = [r["seed"] for r in sel]
                summary["aggregate"][key] = agg
    path = os.path.join(RES_DIR, "sota_baselines_summary.json")
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    log(f"saved {path} ({len(runs)} runs)")

    # ---- D3/D4/D5: 每模型一份 metrics JSON（primary = row s42 own native） ----
    for model in ("deepdta", "graphdta", "moltrans"):
        mruns = [r for r in runs if r["model"] == model]
        if not mruns:
            continue
        primary = next((r for r in mruns if r["split"] == "row" and r["seed"] == SEED0
                        and r["protocol"] == "own" and r["protein"] == "native"
                        and r.get("max_len_smi", 128) == 128
                        and r.get("max_len_prot", 1500) == 1500), None)
        payload = {"model": model, "primary": primary, "runs": mruns,
                   "aggregate": {k: v for k, v in summary["aggregate"].items()
                                 if k.startswith(model + "|")}}
        with open(os.path.join(RES_DIR, f"sota_{model}_metrics.json"), "w") as f:
            json.dump(payload, f, indent=2)

    # ---- D11: cold 汇总 ----
    cold = {}
    for model in ("deepdta", "graphdta", "moltrans"):
        cold[model] = {}
        for split in ("cold_drug", "cold_target", "dual_cold"):
            sel = [r for r in runs if r["model"] == model and r["split"] == split
                   and r["protocol"] == "own" and r["protein"] == "native"]
            if sel:
                cold[model][split] = {
                    "seeds": [r["seed"] for r in sel],
                    "roc_auc_mean": round(float(statistics.mean([r["roc_auc"] for r in sel])), 4),
                    "roc_auc_std": round(float(statistics.stdev([r["roc_auc"] for r in sel]))
                                         if len(sel) > 1 else 0.0, 4),
                    "pr_auc_mean": round(float(statistics.mean([r["pr_auc"] for r in sel])), 4),
                    "f1_mean": round(float(statistics.mean([r["f1"] for r in sel])), 4),
                    "per_run": sel,
                }
    cold["main_model_reference"] = {"cold_drug": 0.8540, "cold_target": 0.8886,
                                    "dual_cold": 0.7999,
                                    "note": "主模型 5 seed 均值，存于 data/external/unified_ext/unc/runs/"}
    with open(os.path.join(RES_DIR, "sota_cold_metrics.json"), "w") as f:
        json.dump(cold, f, indent=2)
    log(f"saved {os.path.join(RES_DIR, 'sota_cold_metrics.json')}")

    # ---- markdown ----
    lines = ["# SOTA 基线补充 — 结果汇总", "",
             "> 数据/划分/早停/指标口径与本文模型一致；基线只使用原始 SMILES + 蛋白序列。",
             "> 主结果协议 = `own`（各自原文超参 + 早停）；`paper` 为本文协议受控对照。",
             "> 实体地板线：met+tar ID 逻辑回归（不看化学/序列）。", ""]
    lines += ["", "## 1. 主表（row-level, own, 均值 ± 样本标准差）", "",
              "| Model | Params | ROC-AUC | PR-AUC | F1 | P@1% | Epochs | 训练时长 | Seeds |",
              "|---|---|---|---|---|---|---|---|---|"]
    floor = json.load(open(os.path.join(RES_DIR, "sota_entity_floor_metrics.json")))
    fl = floor["splits"]["row"]["met+tar"]
    lines.append(f"| 实体地板线 (met+tar ID) | {fl['n_features']:,} | "
                 f"{fl['roc_auc']:.4f} | {fl['pr_auc']:.4f} | {fl['f1']:.4f} | "
                 f"{fl['precision_at_topk']:.4f} | — | — | — |")
    for model in ("deepdta", "graphdta", "moltrans"):
        key = f"{model}|row|own"
        if key not in summary["aggregate"]:
            continue
        a = summary["aggregate"][key]
        lines.append(f"| {model} | {int(a['n_params']['mean']):,} | "
                     f"{a['roc_auc']['mean']:.4f} ± {a['roc_auc']['std']:.4f} | "
                     f"{a['pr_auc']['mean']:.4f} ± {a['pr_auc']['std']:.4f} | "
                     f"{a['f1']['mean']:.4f} ± {a['f1']['std']:.4f} | "
                     f"{a['precision_at_topk']['mean']:.4f} ± {a['precision_at_topk']['std']:.4f} | "
                     f"{a['epochs_run']['mean']:.1f} | {a['train_seconds']['mean']:.0f}s | "
                     f"{len(a['seeds'])} |")
        if model == "moltrans":
            official = [r for r in runs if r["model"] == "moltrans"
                        and r["split"] == "row" and r["protocol"] == "own"
                        and r.get("variant") == "original"]
            if official:
                a2 = statistics.mean([r["roc_auc"] for r in official])
                p2 = statistics.mean([r["pr_auc"] for r in official])
                f2 = statistics.mean([r["f1"] for r in official])
                lines.append(
                    f"| **MolTrans official ESPF** | "
                    f"{int(statistics.mean([r['n_params'] for r in official])):,} | "
                    f"{a2:.4f} | {p2:.4f} | {f2:.4f} | "
                    f"{statistics.mean([r['precision_at_topk'] for r in official]):.4f} | "
                    f"{statistics.mean([r['epochs_run'] for r in official]):.1f} | "
                    f"{statistics.mean([r['train_seconds'] for r in official]):.0f}s | "
                    f"{len(official)} |")
    lines.append(f"| **本文模型 (c2, ECFP4‖ESM-2 proj1024)** | — | 0.9277 | 0.9373 | 0.8586 | 0.9975 | — | — | 4 |")
    lines += ["", "> 本文模型主表行为稿件 headline（seed 42）；四 seed 均值为 "
              "0.9280 ± 0.0006 ROC-AUC（PR 0.9374 ± 0.0007，F1 0.8579 ± 0.0009）。", ""]
    lines += ["", "> **MolTrans 说明**：adapted 行（字符级 token、投影外积、长序列窗口）"
              "仍低于实体地板线，保留作负面消融；official ESPF + 原版单通道交互图"
              "已达到 0.9183，是正式 MolTrans 基线。两者不得混用。", ""]
    lines += ["", "## 2. strict pair-disjoint（own）", "",
              "| Model | ROC-AUC | PR-AUC | F1 | P@1% | Seeds |", "|---|---|---|---|---|---|"]
    fst = floor["splits"]["strict"]["met+tar"]
    lines.append(f"| 实体地板线 (met+tar ID) | {fst['roc_auc']:.4f} | {fst['pr_auc']:.4f} | "
                 f"{fst['f1']:.4f} | {fst['precision_at_topk']:.4f} | — |")
    lines.append("| **本文模型 (c2)** | 0.9191 | 0.9292 | 0.8476 | 1.0000 | 1 |")
    for model in ("deepdta", "graphdta", "moltrans"):
        key = f"{model}|strict|own"
        if key not in summary["aggregate"]:
            if model == "moltrans":
                official = [r for r in runs if r["model"] == "moltrans"
                            and r["split"] == "strict" and r["protocol"] == "own"
                            and r.get("variant") == "original"]
                if official:
                    a2 = statistics.mean([r["roc_auc"] for r in official])
                    p2 = statistics.mean([r["pr_auc"] for r in official])
                    f2 = statistics.mean([r["f1"] for r in official])
                    lines.append(
                        f"| **MolTrans official ESPF** | {a2:.4f} | {p2:.4f} | "
                        f"{f2:.4f} | "
                        f"{statistics.mean([r['precision_at_topk'] for r in official]):.4f} | "
                        f"{len(official)} |")
            continue
        a = summary["aggregate"][key]
        lines.append(f"| {model} | {a['roc_auc']['mean']:.4f} ± {a['roc_auc']['std']:.4f} | "
                     f"{a['pr_auc']['mean']:.4f} ± {a['pr_auc']['std']:.4f} | "
                     f"{a['f1']['mean']:.4f} | {a['precision_at_topk']['mean']:.4f} | "
                     f"{len(a['seeds'])} |")
    lines += ["", "## 3. 本文协议受控对照（paper, row-level, seed 42）", "",
              "| Model | ROC-AUC | PR-AUC | F1 | P@1% | Epochs |", "|---|---|---|---|---|---|"]
    for model in ("deepdta", "graphdta", "moltrans"):
        key = f"{model}|row|paper"
        if key not in summary["aggregate"]:
            continue
        a = summary["aggregate"][key]
        lines.append(f"| {model} | {a['roc_auc']['mean']:.4f} | {a['pr_auc']['mean']:.4f} | "
                     f"{a['f1']['mean']:.4f} | {a['precision_at_topk']['mean']:.4f} | "
                     f"{a['epochs_run']['mean']:.1f} |")
    # 收敛证据（F12）
    lines += ["", "## 3b. 收敛证据（best epoch / 曲线路径）", "",
              "| Model | Split | Seed | Protocol | Best ep | Epochs run | Best val | Curve |",
              "|---|---|---|---|---|---|---|---|"]
    for r in sorted(runs, key=lambda x: (x["model"], x["split"], x["seed"], x["protocol"])):
        if r.get("protein", "native") != "native":
            continue
        if r.get("variant") == "original":
            tag = f"moltrans_original_{r['split']}_s{r['seed']}_{r['protocol']}"
        else:
            tag = f"{r['model']}_{r['split']}_s{r['seed']}_{r['protocol']}"
        prot = r["protocol"]
        if r.get("max_len_smi", 128) != 128 or r.get("max_len_prot", 1500) != 1500:
            tag += f"_L{r['max_len_smi']}x{r['max_len_prot']}"
            prot += f" ({r['max_len_smi']}/{r['max_len_prot']})"
        curve = f"sota_curves/{tag}.npz"
        lines.append(f"| {r['model']} | {r['split']} | {r['seed']} | {prot} | "
                     f"{r['best_epoch']} | {r['epochs_run']} | {r['best_monitor']:.4f} | "
                     f"`{curve}` |")
    # A1 自检
    lines += ["", "## 3c. 验收自检（A1/A2）", ""]
    floor_row = fl["roc_auc"]
    floor_str = fst["roc_auc"]
    for model in ("deepdta", "graphdta", "moltrans"):
        key = f"{model}|row|own"
        if key not in summary["aggregate"]:
            continue
        m = summary["aggregate"][key]["roc_auc"]["mean"]
        if m < floor_row:
            lines.append(f"- **A1 失败**：{model} row-level {m:.4f} < 实体地板线 {floor_row:.4f}；"
                         "必须排查实现并如实报告，不得调参掩盖。")
        elif m > 0.99:
            lines.append(f"- **A1 可疑**：{model} row-level {m:.4f} > 0.99，检查数据泄漏。")
        else:
            lines.append(f"- A1 通过：{model} row-level {m:.4f} ∈ "
                         f"[{floor_row:.4f}, 0.95]。")
        ks = f"{model}|strict|own"
        if ks in summary["aggregate"]:
            ms = summary["aggregate"][ks]["roc_auc"]["mean"]
            if ms < 0.70:
                lines.append(f"- **A2 可疑**：{model} strict {ms:.4f} < 0.70。")
            else:
                lines.append(f"- A2 通过：{model} strict {ms:.4f}（未崩到 0.5）。")
    lines += ["", "## 4. 冷启动（own, seed 42）", "",
              "| Model | cold-drug | cold-target | dual-cold |", "|---|---|---|---|"]
    lines.append("| **本文模型 (c2, 5 seed)** | 0.8540 | 0.8886 | 0.7999 |")
    for model in ("deepdta", "graphdta", "moltrans"):
        row = [f"| {model} |"]
        for split in ("cold_drug", "cold_target", "dual_cold"):
            key = f"{model}|{split}|own"
            row.append(f" {summary['aggregate'][key]['roc_auc']['mean']:.4f} |"
                       if key in summary["aggregate"] else " — |")
        lines.append("".join(row))
    lines += ["", "## 5. 外部五源（own, row s42 checkpoint, 只推理）", ""]
    ep = os.path.join(RES_DIR, "sota_external_metrics.json")
    if os.path.exists(ep):
        ext = json.load(open(ep))
        lines += ["### 5a. 全部可编码行（baseline coverage 99.9–100%）", "",
                  "| Model | ChEMBL | HMDB | DrugBank∩HMDB | DrugBank full | PDB |",
                  "|---|---|---|---|---|---|",
                  "| 本文模型（主结果，含零填充行） | 0.8813 | 0.8514 | 0.8545 | 0.8143 | 0.7491 |"]
        for model in ("deepdta", "graphdta", "moltrans"):
            if model not in ext:
                continue
            s = ext[model]["sources"]
            def g(k):
                return f"{s[k]['roc_auc']:.4f}" if k in s and "roc_auc" in s[k] else "—"
            lines.append(f"| {model} | {g('chembl')} | {g('hmdb_assoc')} | "
                         f"{g('drugbank_in_hmdb')} | {g('drugbank')} | {g('pdb')} |")
        lines += ["", "### 5b. 与主模型 complete-only 严格同口径（仅 ESM 缓存覆盖的序列）", "",
                  "| Model | ChEMBL | HMDB | DrugBank∩HMDB | DrugBank full | PDB |",
                  "|---|---|---|---|---|---|",
                  "| 本文模型（complete-only） | 0.8813 | 0.8875 | 0.8598 | 0.8189 | 0.7451 |"]
        for model in ("deepdta", "graphdta", "moltrans"):
            if model not in ext:
                continue
            s = ext[model]["sources"]
            def ge(k):
                return (f"{s[k]['esm_subset']['roc_auc']:.4f}"
                        if k in s and "esm_subset" in s[k] else "—")
            lines.append(f"| {model} | {ge('chembl')} | {ge('hmdb_assoc')} | "
                         f"{ge('drugbank_in_hmdb')} | {ge('drugbank')} | {ge('pdb')} |")
        lines += ["", "> 5a 中基线对 99.9–100% 的行可编码（直接用本地 FASTA），"
                  "而主模型对部分外部序列只能零填充；因此 5a 对基线更有利。"
                  "5b 将基线限制到与主模型 `*_complete_only` 完全相同的行，"
                  "是公平的主比较。"]
    lines += ["", "## 6. 与本文模型的配对显著性（DeLong, row-level seed 42）", "",
              "| Model | 本文模型 AUC | 基线 AUC | Δ | z | p |", "|---|---|---|---|---|---|"]
    main_pred = os.path.join(cfg.get_path("paths.external_dir"), "unified_ext", "unc",
                             "runs", "bce_rand_s42_r1.npz")
    if os.path.exists(main_pred):
        mz = np.load(main_pred)
        for model in ("deepdta", "graphdta", "moltrans"):
            pp = os.path.join(PRED_DIR, f"{model}_row_s42_own.npz")
            if not os.path.exists(pp):
                continue
            z = np.load(pp)
            if len(z["y"]) != len(mz["y"]):
                continue
            auc1, auc2, zz, pv = delong_paired(mz["y"], mz["p"], z["p"])
            lines.append(f"| {model} | {auc1:.4f} | {auc2:.4f} | {auc1-auc2:+.4f} | "
                         f"{zz:+.2f} | {pv:.2e} |")
    lines += ["", "## 7. 与原文的偏离（逐条）", "",
              "- 所有基线：回归头 → 1 logit + BCEWithLogits（二分类适配）。",
              "- 所有基线：方向任务不适用（`n/a`）。",
              "- DeepDTA：整数 token + Embedding(128) + Conv1D 32/64/96；"
              "FC 1024→1024→512；主结果 max_len 128/1500（原文 85/1000）。",
              "- GraphDTA：GIN 4 层（纯 PyTorch），原子特征为元素/度/隐式氢/芳香性/杂化/"
              "形式电荷/成环 one-hot；蛋白分支同 DeepDTA；主结果 max_len 128/1500。"
              "零键分子保留单节点图。",
              "- MolTrans：2 层 / 12 头 / emb 384；SMILES 字符级（原文 ESPF 子词）；"
              "蛋白 3-mer；外积前投影到 32 维（显存约束）；交互 CNN stride-2 + 4×4 avg pool。"
              "主结果 max_len 128/1500（原文 50/545）。",
              "- MolTrans batch：own 64（原文 128；16 GB 下 128 会 OOM）。",
              "- GraphDTA max_epochs：300（原文 1000；算力截断，早停 patience 30）。",
              "- 所有 run 启用 AMP；确定性开关 cudnn.deterministic=True，"
              "`torch.use_deterministic_algorithms(warn_only=True)`。"]
    with open(os.path.join(RES_DIR, "sota_baselines.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"saved {os.path.join(RES_DIR, 'sota_baselines.md')}")


# ============================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["deepdta", "graphdta", "moltrans"])
    ap.add_argument("--split", choices=["row", "strict", "cold_drug", "cold_target", "dual_cold"],
                    default="row")
    ap.add_argument("--seed", type=int, default=SEED0)
    ap.add_argument("--protocol", choices=["own", "paper"], default="own")
    ap.add_argument("--max_len_smi", type=int, default=DEFAULT_SMI)
    ap.add_argument("--max_len_prot", type=int, default=DEFAULT_PROT)
    ap.add_argument("--protein", choices=["native", "esm_frozen"], default="native")
    ap.add_argument("--external", choices=["all"], default=None)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--build_cache", action="store_true")
    ap.add_argument("--force_cache", action="store_true")
    # 覆盖项（默认按协议）
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--amp", choices=["fp16", "bf16", "none"], default=None)
    args = ap.parse_args()

    if args.build_cache:
        build_cache(force=args.force_cache)
        return
    if args.summarize:
        summarize(args)
        return
    if args.external:
        if not args.model:
            raise SystemExit("--external 需要 --model")
        score_external(args)
        return
    if not args.model:
        raise SystemExit("需要 --model 或 --summarize/--external/--build_cache")

    base = OWN[args.model] if args.protocol == "own" else PAPER[args.model]
    if args.amp is None:
        # MolTrans 的 384 维外积/attention 在 fp16 下数值不稳（实测首个 epoch 后 NaN）；
        # bf16 与 fp32 同指数范围，实测 30 步稳定且比 fp32 快约 2 倍。
        args.amp = "bf16" if args.model == "moltrans" else "fp16"
    args.lr = args.lr if args.lr is not None else base["lr"]
    args.batch_size = args.batch_size if args.batch_size is not None else base["batch"]
    args.max_epochs = args.max_epochs if args.max_epochs is not None else base["max_epochs"]
    args.patience = args.patience if args.patience is not None else base["patience"]
    run(args)


if __name__ == "__main__":
    main()
