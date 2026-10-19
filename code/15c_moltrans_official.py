#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Faithful MolTrans reproduction using the official ESPF artifacts.

This entry point is intentionally separate from ``15_sota_baselines.py``:

* ``original`` uses the official ESPF BPE vocabularies, the official
  element-wise-product interaction map, and the paper's max lengths 50/545.
* The existing ``adapted`` implementation in ``15_sota_baselines.py`` remains
  unchanged for side-by-side comparison.

The shared data split, early-stopping loop and metrics come from step 15.
"""
import argparse
import codecs
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
sys.path.insert(0, BASE_DIR)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s15 = _load_module("sota_step15", os.path.join(BASE_DIR, "15_sota_baselines.py"))

ESPF_DIR = os.path.join(ROOT, "third_party", "MolTrans_official", "ESPF")
OFFICIAL_MODELS = os.path.join(ROOT, "third_party", "MolTrans_official", "models.py")
ESPF_CACHE = os.path.join(s15.CACHE_DIR, "espf_official.npz")
MAX_DRUG = 50
MAX_PROT = 545


def _read_espf():
    from subword_nmt.apply_bpe import BPE

    drug_bpe = BPE(
        codecs.open(os.path.join(ESPF_DIR, "drug_codes_chembl.txt")),
        merges=-1,
        separator="",
    )
    prot_bpe = BPE(
        codecs.open(os.path.join(ESPF_DIR, "protein_codes_uniprot.txt")),
        merges=-1,
        separator="",
    )
    drug_map = pd.read_csv(os.path.join(ESPF_DIR, "subword_units_map_chembl.csv"))
    prot_map = pd.read_csv(os.path.join(ESPF_DIR, "subword_units_map_uniprot.csv"))
    drug_words = drug_map["index"].astype(str).values
    prot_words = prot_map["index"].astype(str).values
    return (
        drug_bpe,
        prot_bpe,
        {w: i for i, w in enumerate(drug_words)},
        {w: i for i, w in enumerate(prot_words)},
        len(drug_words),
        len(prot_words),
    )


def _encode_espf(seq, bpe, word_to_idx, max_len):
    """Match stream.py: unknown-token failures map the whole sequence to [0]."""
    pieces = bpe.process_line(str(seq)).split()
    try:
        ids = np.asarray([word_to_idx[p] for p in pieces], dtype=np.int64)
    except KeyError:
        ids = np.asarray([0], dtype=np.int64)
    ids = ids[:max_len]
    mask = np.ones(len(ids), dtype=np.int8)
    if len(ids) < max_len:
        pad = max_len - len(ids)
        ids = np.pad(ids, (0, pad), constant_values=0)
        mask = np.pad(mask, (0, pad), constant_values=0)
    return ids.astype(np.int16), mask


def build_espf_cache(force=False):
    if os.path.exists(ESPF_CACHE) and not force:
        return
    s15.build_cache()
    tok = np.load(os.path.join(s15.CACHE_DIR, "tokens.npz"))
    met_ids = tok["met_ids"].astype(np.int64)
    tar_ids = tok["tar_ids"].astype(np.int64)

    ref = s15._load_ref()
    met2smi = ref.drop_duplicates("met").set_index("met")["smiles"].to_dict()
    tar2seq = ref.drop_duplicates("tar").set_index("tar")["protein_seq"].to_dict()
    met_smiles = [met2smi[int(m)] for m in met_ids]
    tar_seqs = [tar2seq[int(t)] for t in tar_ids]

    drug_bpe, prot_bpe, drug_map, prot_map, drug_vocab, prot_vocab = _read_espf()
    smi = np.zeros((len(met_smiles), MAX_DRUG), dtype=np.int16)
    smi_mask = np.zeros((len(met_smiles), MAX_DRUG), dtype=np.int8)
    prot = np.zeros((len(tar_seqs), MAX_PROT), dtype=np.int16)
    prot_mask = np.zeros((len(tar_seqs), MAX_PROT), dtype=np.int8)

    t0 = time.time()
    for i, seq in enumerate(met_smiles):
        smi[i], smi_mask[i] = _encode_espf(seq, drug_bpe, drug_map, MAX_DRUG)
    for i, seq in enumerate(tar_seqs):
        prot[i], prot_mask[i] = _encode_espf(seq, prot_bpe, prot_map, MAX_PROT)

    np.savez_compressed(
        ESPF_CACHE,
        smi=smi,
        smi_mask=smi_mask,
        prot=prot,
        prot_mask=prot_mask,
        drug_vocab=np.asarray([drug_vocab], dtype=np.int64),
        prot_vocab=np.asarray([prot_vocab], dtype=np.int64),
    )
    s15.log(
        f"ESPF cache saved: {ESPF_CACHE}; drug_vocab={drug_vocab}, "
        f"prot_vocab={prot_vocab}, {time.time() - t0:.1f}s"
    )


class ESPFData(s15.TokenData):
    def __init__(self):
        super().__init__()
        z = np.load(ESPF_CACHE)
        self.espf_smi = z["smi"]
        self.espf_smi_mask = z["smi_mask"]
        self.espf_prot = z["prot"]
        self.espf_prot_mask = z["prot_mask"]
        self.drug_vocab = int(z["drug_vocab"][0])
        self.prot_vocab = int(z["prot_vocab"][0])

    def batch(self, rows, model_name, max_smi, max_prot, protein_mode="native"):
        mi = self.row_met[rows]
        ti = self.row_tar[rows]
        smi = torch.from_numpy(self.espf_smi[mi].astype(np.int64))
        prot = torch.from_numpy(self.espf_prot[ti].astype(np.int64))
        smi_mask = torch.from_numpy(self.espf_smi_mask[mi].astype(np.int64))
        prot_mask = torch.from_numpy(self.espf_prot_mask[ti].astype(np.int64))
        return (smi, smi_mask), (prot, prot_mask)


class OfficialMolTrans(nn.Module):
    def __init__(self, official_cls, drug_vocab, prot_vocab, batch_size):
        super().__init__()
        config = {
            "batch_size": int(batch_size),
            "input_dim_drug": int(drug_vocab),
            "input_dim_target": int(prot_vocab),
            "max_drug_seq": MAX_DRUG,
            "max_protein_seq": MAX_PROT,
            "emb_size": 384,
            "dropout_rate": 0.1,
            "scale_down_ratio": 0.25,
            "growth_rate": 20,
            "transition_rate": 0.5,
            "num_dense_blocks": 4,
            "kernal_dense_size": 3,
            "intermediate_size": 1536,
            "num_attention_heads": 12,
            "attention_probs_dropout_prob": 0.1,
            "hidden_dropout_prob": 0.1,
            "flat_dim": (MAX_DRUG - 2) * (MAX_PROT - 2) * 3,
        }
        self.net = official_cls(**config)

    def forward(self, smi, prot):
        (smi_ids, smi_mask), (prot_ids, prot_mask) = smi, prot
        return self.net(smi_ids, prot_ids, smi_mask, prot_mask).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="row",
                    choices=["row", "strict", "cold_drug", "cold_target", "dual_cold"])
    ap.add_argument("--seed", type=int, default=s15.SEED0)
    ap.add_argument("--protocol", default="own", choices=["own", "paper"])
    ap.add_argument("--max_epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--force_cache", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="Keep a short timing/forward run separate from formal results")
    args = ap.parse_args()
    args.max_len_smi = MAX_DRUG
    args.max_len_prot = MAX_PROT
    args.protein = "native"

    s15.set_seed(args.seed)
    build_espf_cache(force=args.force_cache)
    data = ESPFData()
    official = _load_module("moltrans_official_models", OFFICIAL_MODELS)
    # The official module calls manual_seed(1) at import time. Re-apply the
    # requested seed so all six seeds differ only by initialization/data order.
    s15.set_seed(args.seed)

    y = np.load(s15.cfg.sidecar_npz(
        "labels", s15.cfg.get("training.feat_file_map")["esm"]
    ))["y_binary"].astype(np.int64)
    tr, va, te, drop = s15.get_split(args.split, data, args.seed)
    model = OfficialMolTrans(
        official.BIN_Interaction_Flat,
        data.drug_vocab,
        data.prot_vocab,
        args.batch_size,
    ).to(s15.DEVICE)
    s15.log(
        f"MolTrans-original split={args.split} seed={args.seed}: "
        f"train {len(tr):,} / val {len(va):,} / test {len(te):,}; "
        f"params={s15.n_params(model):,}; max_len={MAX_DRUG}/{MAX_PROT}"
    )

    t0 = time.time()
    model, info, curves = s15.train_one(model, "moltrans", data, tr, va, y, args)
    p_te = s15.predict(model, "moltrans", data, te, args, bs=args.batch_size)
    metrics = s15.compute_metrics(y[te], p_te)
    tag = f"moltrans_original_{args.split}_s{args.seed}_{args.protocol}"
    if args.smoke:
        tag += "_smoke"
    out = {
        "model": "moltrans",
        "variant": "original",
        "split": args.split,
        "seed": int(args.seed),
        "protocol": args.protocol,
        "protein": "native",
        "max_len_smi": MAX_DRUG,
        "max_len_prot": MAX_PROT,
        "n_params": s15.n_params(model),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "patience": int(args.patience),
        "max_epochs": int(args.max_epochs),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "drop_rate": float(drop) if drop is not None else 0.0,
        **info,
        **metrics,
        "wall_seconds": round(time.time() - t0, 1),
    }
    with open(os.path.join(s15.RUNS_DIR, f"{tag}.json"), "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(
        os.path.join(s15.PRED_DIR, f"{tag}.npz"),
        y=y[te],
        p=p_te,
        rows=te,
    )
    np.savez_compressed(
        os.path.join(s15.CURVES_DIR, f"{tag}.npz"),
        train_loss=np.asarray(curves["train_loss"]),
        val_auc=np.asarray(curves["val_auc"]),
        val_acc=np.asarray(curves["val_acc"]),
        best_epoch=np.asarray([info["best_epoch"]]),
        best_monitor=np.asarray([info["best_monitor"]]),
        model=np.asarray(["moltrans"]),
        variant=np.asarray(["original"]),
        split=np.asarray([args.split]),
        seed=np.asarray([args.seed]),
        protocol=np.asarray([args.protocol]),
    )
    torch.save(
        {"state_dict": model.state_dict(), "meta": out},
        os.path.join(s15.CKPT_DIR, f"{tag}.pt"),
    )
    s15.log(
        f"RESULT {tag}: ROC-AUC {metrics['roc_auc']:.4f} "
        f"PR-AUC {metrics['pr_auc']:.4f} F1 {metrics['f1']:.4f} "
        f"P@1% {metrics['precision_at_topk']:.4f} "
        f"({info['train_seconds']:.0f}s, {info['epochs_run']} ep, "
        f"best {info['best_epoch']})"
    )


if __name__ == "__main__":
    main()
