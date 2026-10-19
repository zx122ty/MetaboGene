#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""GraphDTA-inspired interaction model optimized for the MetaboGene split.

The control experiments showed that GraphDTA with frozen ESM-2 is stronger
than either native GraphDTA or the current ECFP4+ESM MLP. This script tests a
controlled three-way representation:

    RDKit molecular graph (GIN) + ECFP4 + frozen ESM-2 -> MLP

The data split, row order, test metrics and early-stopping criterion are shared
with the existing MetaboGene and SOTA baseline runs.
"""
import argparse
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


s15 = _load_module("sota_step15_for_hybrid", os.path.join(BASE_DIR, "15_sota_baselines.py"))


class GraphECFPESM(nn.Module):
    def __init__(self, ecfp_dim=2048, atom_dim=48, graph_hidden=128,
                 graph_layers=4, ecfp_hidden=512, esm_dim=2560,
                 esm_hidden=1024, hidden=1024, dropout=0.1, n_outputs=1,
                 graph_dropout=0.0):
        super().__init__()
        self.atom_proj = nn.Linear(atom_dim, graph_hidden)
        self.gins = nn.ModuleList(
            [s15.GINLayer(graph_hidden) for _ in range(graph_layers)]
        )
        self.graph_out = nn.Sequential(
            nn.Linear(2 * graph_hidden, graph_hidden),
            nn.ReLU(),
        )
        if ecfp_hidden > 0:
            self.ecfp_branch = nn.Sequential(
                nn.Linear(ecfp_dim, ecfp_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            ecfp_out = ecfp_hidden
        else:
            # The original ECFP projection sweep showed that compressing the
            # 2048-bit fingerprint can discard useful sparse substructures.
            self.ecfp_branch = nn.Identity()
            ecfp_out = ecfp_dim
        self.esm_branch = nn.Sequential(
            nn.Linear(esm_dim, esm_hidden),
            nn.ReLU(),
        )
        fused_dim = graph_hidden + ecfp_out + esm_hidden
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, n_outputs),
        )
        self.n_outputs = int(n_outputs)
        self.graph_dropout = float(graph_dropout)

    def encode_graph(self, atom_feats, edge_index, graph_batch, n_graphs):
        h = self.atom_proj(atom_feats)
        for gin in self.gins:
            h = gin(h, edge_index)
        sum_p = h.new_zeros(n_graphs, h.shape[1]).index_add_(
            0, graph_batch, h
        )
        count = h.new_zeros(n_graphs).index_add_(
            0, graph_batch, h.new_ones(h.shape[0])
        )
        mean_p = sum_p / count.unsqueeze(1).clamp_min(1)
        return self.graph_out(torch.cat([sum_p, mean_p], dim=1))

    def forward(self, graphs, ecfp, esm):
        atom_feats, edge_index, graph_batch, n_graphs = graphs
        graph = self.encode_graph(
            atom_feats, edge_index, graph_batch, n_graphs
        )
        if self.training and self.graph_dropout > 0:
            keep = 1.0 - self.graph_dropout
            mask = (torch.rand(
                graph.shape[0], 1, device=graph.device
            ) < keep).to(graph.dtype) / keep
            graph = graph * mask
        out = self.fc(torch.cat(
            [graph, self.ecfp_branch(ecfp), self.esm_branch(esm)], dim=1
        ))
        return out.squeeze(-1) if self.n_outputs == 1 else out


def _model_inputs(data, x_ecfp, rows, device):
    graphs = s15.collate_graphs(data.graphs_from(rows), device)
    target_idx = data.row_tar[rows]
    ecfp = torch.from_numpy(x_ecfp[rows].astype(np.float32, copy=False)).to(device)
    esm = torch.from_numpy(
        np.asarray(data.esm[target_idx], dtype=np.float32)
    ).to(device)
    return graphs, ecfp, esm


def predict(model, data, x_ecfp, rows, device, batch_size, task="binary"):
    model.eval()
    out = []
    with torch.no_grad():
        for i in range(0, len(rows), batch_size):
            r = rows[i:i + batch_size]
            graphs, ecfp, esm = _model_inputs(data, x_ecfp, r, device)
            logits = model(graphs, ecfp, esm)
            if task == "binary":
                logits = torch.sigmoid(logits)
            out.append(logits.float().cpu().numpy())
    return np.concatenate(out)


def _ecfp_from_smiles(smiles):
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        raise ValueError(f"invalid external SMILES: {smiles!r}")
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
    return np.asarray(fp, dtype=np.float32)


def score_external(args):
    ckpt = args.checkpoint or os.path.join(
        s15.CKPT_DIR, "graph_ecfp_esm_row_s42_b512.pt"
    )
    if not os.path.exists(ckpt):
        raise SystemExit(f"Missing checkpoint: {ckpt}")
    model = GraphECFPESM().to(s15.DEVICE)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = state.get("meta", {})
    if int(meta.get("ecfp_hidden", 512)) != 512:
        model = GraphECFPESM(
            ecfp_hidden=int(meta["ecfp_hidden"]),
            dropout=float(meta.get("dropout", 0.1)),
        ).to(s15.DEVICE)
    model.load_state_dict(state["state_dict"])
    model.eval()

    fa = s15._load_fasta()
    smi_cache = s15._pubchem_smiles()
    frames = s15._external_frames()
    esm_dir = s15.cfg.get_path("paths.esm_cache_dir")
    esm_emb = np.load(os.path.join(esm_dir, "embeddings.npy"), mmap_mode="r")
    esm_ids = json.load(open(os.path.join(esm_dir, "ids.json")))
    seq2idx = {r["seq"]: int(r["index"]) for r in esm_ids}
    prob_store = {}

    def smiles_of(df, is_chembl):
        if is_chembl and "canonical_smiles" in df.columns:
            vals = df["canonical_smiles"].tolist()
        else:
            vals = [None] * len(df)
        cids = df["metabolite_pubchem_cid"].astype("int64").astype(str).tolist()
        return [
            v if isinstance(v, str) else smi_cache.get(c)
            for v, c in zip(vals, cids)
        ]

    def score_subset(d, smis, seqs, ii, valid, save_key=None):
        ps = []
        bs = 256
        with torch.no_grad():
            for start in range(0, len(ii), bs):
                idx = ii[start:start + bs]
                graphs = s15.collate_graphs(
                    [s15._build_graph(smis[k]) for k in idx],
                    s15.DEVICE,
                )
                ecfp = torch.from_numpy(np.stack(
                    [_ecfp_from_smiles(smis[k]) for k in idx]
                )).to(s15.DEVICE)
                esm = np.zeros((len(idx), esm_emb.shape[1]), dtype=np.float32)
                for j, k in enumerate(idx):
                    e = seq2idx.get(seqs[k])
                    if e is not None:
                        esm[j] = esm_emb[e]
                esm_t = torch.from_numpy(esm).to(s15.DEVICE)
                logits = model(graphs, ecfp, esm_t)
                ps.append(torch.sigmoid(logits).float().cpu().numpy())
        probs = np.concatenate(ps)
        y = d["is_active"].values[ii].astype(int)
        if save_key is not None:
            prob_store[save_key] = {"y": y, "p": probs, "rows": ii}
        met = s15.compute_metrics(y, probs)
        met.update({
            "n_scored": int(len(ii)),
            "n_rows": int(len(d)),
            "coverage": round(float(valid.mean()), 4),
            "smiles_cov": round(float(np.mean(
                [s is not None for s in smis]
            )), 4),
            "seq_cov": round(float(np.mean(
                [q is not None for q in seqs]
            )), 4),
        })
        return met

    def score_frame(name, df, chemistry=False):
        d = df.reset_index(drop=True)
        smis = smiles_of(d, chemistry)
        seqs = [fa.get(str(u)) for u in d["target_uniprot"].tolist()]
        valid = np.array([
            s is not None and q is not None
            for s, q in zip(smis, seqs)
        ])
        ii = np.where(valid)[0]
        if not len(ii):
            return {"n_scored": 0}
        out = score_subset(d, smis, seqs, ii, valid, f"{name}:all")
        esm_ok = np.array([q in seq2idx for q in seqs])
        ii2 = np.where(valid & esm_ok)[0]
        if len(ii2) > 20 and len(np.unique(d["is_active"].values[ii2])) > 1:
            sub = score_subset(
                d, smis, seqs, ii2, valid & esm_ok, f"{name}:esm"
            )
            sub["coverage_of_all"] = round(float(len(ii2) / len(d)), 4)
            out["esm_subset"] = sub
        return out

    out = {
        "model": "graph_ecfp_esm",
        "checkpoint": ckpt,
        "seed": int(meta.get("seed", 42)),
        "sources": {},
    }
    if "chembl" in frames:
        out["sources"]["chembl"] = score_frame(
            "chembl", frames["chembl"], chemistry=True
        )
        s15.log(f"chembl: {out['sources']['chembl']}")
    if "hmdb_assoc" in frames:
        hm = frames["hmdb_assoc"]
        a, n0 = hm[hm["is_active"] == 1], hm[hm["is_active"] == 0]
        if len(a) > len(n0):
            a = a.sample(len(n0), random_state=42)
        hm = pd.concat([a, n0], ignore_index=True)
        out["sources"]["hmdb_assoc"] = score_frame("hmdb_assoc", hm)
        s15.log(f"hmdb_assoc: {out['sources']['hmdb_assoc']}")
    if "pdb" in frames:
        out["sources"]["pdb"] = score_frame("pdb", frames["pdb"])
        s15.log(f"pdb: {out['sources']['pdb']}")
    if "drugbank" in frames:
        db = frames["drugbank"]
        out["sources"]["drugbank"] = score_frame("drugbank", db)
        s15.log(f"drugbank: {out['sources']['drugbank']}")
        sub = db[db["in_hmdb"] == True] if "in_hmdb" in db.columns else db.iloc[0:0]
        a, n0 = sub[sub["is_active"] == 1], sub[sub["is_active"] == 0]
        if len(a) > len(n0):
            a = a.sample(len(n0), random_state=42)
        sub = pd.concat([a, n0], ignore_index=True)
        out["sources"]["drugbank_in_hmdb"] = score_frame(
            "drugbank_in_hmdb", sub
        )
        s15.log(f"drugbank_in_hmdb: {out['sources']['drugbank_in_hmdb']}")

    out_name = args.tag or "sota_graph_ecfp_esm_external_metrics"
    path = os.path.join(s15.RES_DIR, f"{out_name}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    np.savez_compressed(
        os.path.join(s15.RES_DIR, f"{out_name}_probs.npz"),
        **{
            f"{key}:y": value["y"]
            for key, value in prob_store.items()
        },
        **{
            f"{key}:p": value["p"]
            for key, value in prob_store.items()
        },
        **{
            f"{key}:rows": value["rows"]
            for key, value in prob_store.items()
        },
    )
    s15.log(f"saved {path}")


def ensemble_external(args):
    """Equal-weight external ensemble of GraphECFPESM seeds 0/1/2/42."""
    seeds = (0, 1, 2, 42)
    paths = [
        os.path.join(s15.RES_DIR, f"metabogene_graph_s{s}_external_probs.npz")
        for s in seeds
    ]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        raise SystemExit(f"Missing external seed probabilities: {missing}")
    members = [np.load(p) for p in paths]
    files = [f for f in members[0].files if f.endswith(":y")]
    probs = {}
    metrics = {
        "model": "MetaboGene-Graph-E4",
        "seeds": list(seeds),
        "aggregation": "equal mean probability",
        "sources": {},
    }
    for ykey in files:
        base = ykey[:-2]
        y = members[0][ykey]
        rows = members[0][base + ":rows"]
        if any(not np.array_equal(y, z[ykey]) for z in members):
            raise RuntimeError(f"Label mismatch for {base}")
        if any(not np.array_equal(rows, z[base + ":rows"]) for z in members):
            raise RuntimeError(f"Row mismatch for {base}")
        p = np.mean([z[base + ":p"].astype(np.float64) for z in members], axis=0)
        probs[ykey] = y
        probs[base + ":p"] = p
        probs[base + ":rows"] = rows
        met = s15.compute_metrics(y, p)
        met["n_scored"] = int(len(y))
        metrics["sources"][base] = met
    out_name = args.tag or "metabogene_graph_e4_external_metrics"
    with open(os.path.join(s15.RES_DIR, f"{out_name}.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    np.savez_compressed(
        os.path.join(s15.RES_DIR, f"{out_name}_probs.npz"),
        **probs,
    )
    s15.log(f"saved external E4 ensemble: {out_name}")


def train(model, data, x_ecfp, y, tr, va, args):
    device = s15.DEVICE
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = (nn.BCEWithLogitsLoss() if args.task == "binary"
            else nn.CrossEntropyLoss())
    amp_dtype = torch.bfloat16 if args.amp == "bf16" else torch.float16
    use_amp = args.amp != "none"
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and args.amp == "fp16"))
    rng = np.random.RandomState(args.seed)
    best_auc, best_ep, bad = -1.0, 0, 0
    best_state = None
    curves = {"train_loss": [], "val_auc": [], "val_acc": []}
    t0 = time.time()

    for ep in range(args.max_epochs):
        model.train()
        order = tr[rng.permutation(len(tr))]
        losses = []
        for i in range(0, len(order), args.batch_size):
            rows = order[i:i + args.batch_size]
            graphs, ecfp, esm = _model_inputs(data, x_ecfp, rows, device)
            yb = torch.from_numpy(y[rows])
            yb = yb.float().to(device) if args.task == "binary" else yb.long().to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                loss = crit(model(graphs, ecfp, esm), yb)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True)
                continue
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                gn = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                if torch.isfinite(gn):
                    scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip
                )
                if torch.isfinite(gn):
                    opt.step()
            losses.append(float(loss.detach()))
        if not losses:
            raise RuntimeError("No finite training batches")

        p_va = predict(
            model, data, x_ecfp, va, device,
            batch_size=min(args.eval_batch_size, args.batch_size),
            task=args.task,
        )
        if args.task == "binary":
            va_auc = float(s15.roc_auc_score(y[va], p_va))
            va_acc = float(((p_va > 0.5).astype(int) == y[va]).mean())
            monitor = va_auc
        else:
            va_auc = float("nan")
            va_pred = p_va.argmax(axis=1)
            va_acc = float((va_pred == y[va]).mean())
            monitor = va_acc
        curves["train_loss"].append(float(np.mean(losses)))
        curves["val_auc"].append(va_auc)
        curves["val_acc"].append(va_acc)
        if monitor > best_auc:
            best_auc, best_ep, bad = monitor, ep + 1, 0
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
        else:
            bad += 1
        s15.log(
            f"    ep {ep + 1:03d}/{args.max_epochs} "
            f"loss={curves['train_loss'][-1]:.4f} "
            f"val_auc={va_auc:.4f} val_acc={va_acc:.4f} "
            f"best_ep={best_ep} bad={bad}"
        )
        if bad >= args.patience:
            s15.log(f"    early stop at epoch {ep + 1} (patience {args.patience})")
            break

    model.load_state_dict(best_state)
    return model, {
        "train_seconds": round(time.time() - t0, 1),
        "epochs_run": len(curves["train_loss"]),
        "best_epoch": int(best_ep),
        "best_monitor": round(float(best_auc), 4),
    }, curves


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="row", choices=["row", "strict"])
    ap.add_argument("--task", default="binary", choices=["binary", "direction"])
    ap.add_argument("--seed", type=int, default=s15.SEED0)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--grad_clip", type=float, default=5.0)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "none"])
    ap.add_argument("--ecfp_hidden", type=int, default=512,
                    help="ECFP projection width; 0 keeps the raw 2048-bit block")
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--graph_dropout", type=float, default=0.0)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--external", choices=["all", "ensemble"], default=None)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    s15.set_seed(args.seed)
    s15.build_cache()
    data = s15.TokenData()
    if args.external == "all":
        score_external(args)
        return
    if args.external == "ensemble":
        ensemble_external(args)
        return
    label_file = np.load(s15.cfg.sidecar_npz(
        "labels", s15.cfg.get("training.feat_file_map")["esm"]
    ))
    split_file = np.load(s15.cfg.sidecar_npz(
        "split", s15.cfg.get("training.feat_file_map")["esm"]
    ))
    if args.task == "binary":
        y = label_file["y_binary"].astype(np.int64)
        tr, va, te, drop = s15.get_split(args.split, data, args.seed)
    else:
        y = label_file["y_direction"].astype(np.int64)
        if args.split == "row":
            tr, va, te = split_file["d_tr"], split_file["d_va"], split_file["d_te"]
        else:
            b_tr, b_va, b_te, _ = s15.get_split("strict", data, args.seed)
            n_pos = int(data.n_pos)
            tr, va, te = (b_tr[b_tr < n_pos], b_va[b_va < n_pos],
                          b_te[b_te < n_pos])
        drop = None

    ecfp_cache = os.path.join(s15.FEAT_DIR, "ecfp4_esmrows.npy")
    if os.path.exists(ecfp_cache):
        s15.log(f"loading ECFP4 memory map: {ecfp_cache}")
        x_ecfp = np.load(ecfp_cache, mmap_mode="r")
    else:
        s15.log("loading ECFP4 block from features.npz (first 2048 columns)")
        feat = np.load(
            os.path.join(s15.FEAT_DIR, "features.npz"), allow_pickle=True
        )["X_binary"]
        x_ecfp = feat[:, :2048]
    if len(x_ecfp) != len(label_file["y_binary"]):
        raise RuntimeError("ECFP row count does not match binary feature rows")

    model = GraphECFPESM(
        n_outputs=1 if args.task == "binary" else 3,
        ecfp_hidden=args.ecfp_hidden,
        dropout=args.dropout,
        graph_dropout=args.graph_dropout,
    ).to(s15.DEVICE)
    s15.log(
        f"GraphECFPESM task={args.task} split={args.split} seed={args.seed}: "
        f"train {len(tr):,} / val {len(va):,} / test {len(te):,}; "
        f"params={s15.n_params(model):,}"
    )
    model, info, curves = train(model, data, x_ecfp, y, tr, va, args)
    p_te = predict(
        model, data, x_ecfp, te, s15.DEVICE,
        batch_size=min(args.eval_batch_size, args.batch_size),
        task=args.task,
    )
    if args.task == "binary":
        metrics = s15.compute_metrics(y[te], p_te)
    else:
        from sklearn.metrics import confusion_matrix, f1_score
        pred = p_te.argmax(axis=1)
        metrics = {
            "accuracy": round(float((pred == y[te]).mean()), 4),
            "macro_f1": round(float(f1_score(y[te], pred, average="macro")), 4),
            "confusion_matrix_label_order": s15.cfg.direction_label_names(),
            "cm": confusion_matrix(y[te], pred).tolist(),
        }
    tag = args.tag or (
        f"graph_ecfp_esm_{args.task}_{args.split}_s{args.seed}"
    )
    result = {
        "model": "graph_ecfp_esm",
        "task": args.task,
        "protocol": "own",
        "split": args.split,
        "seed": int(args.seed),
        "max_len_smi": 128,
        "max_len_prot": 1500,
        "n_params": s15.n_params(model),
        "batch_size": int(args.batch_size),
        "lr": float(args.lr),
        "ecfp_hidden": int(args.ecfp_hidden),
        "dropout": float(args.dropout),
        "graph_dropout": float(args.graph_dropout),
        "patience": int(args.patience),
        "max_epochs": int(args.max_epochs),
        "n_train": int(len(tr)),
        "n_val": int(len(va)),
        "n_test": int(len(te)),
        "drop_rate": float(drop) if drop is not None else 0.0,
        **info,
        **metrics,
    }
    with open(os.path.join(s15.RUNS_DIR, f"{tag}.json"), "w") as f:
        json.dump(result, f, indent=2)
    np.savez_compressed(
        os.path.join(s15.PRED_DIR, f"{tag}.npz"),
        y=y[te], p=p_te, rows=te,
    )
    np.savez_compressed(
        os.path.join(s15.CURVES_DIR, f"{tag}.npz"),
        train_loss=np.asarray(curves["train_loss"]),
        val_auc=np.asarray(curves["val_auc"]),
        val_acc=np.asarray(curves["val_acc"]),
        best_epoch=np.asarray([info["best_epoch"]]),
        best_monitor=np.asarray([info["best_monitor"]]),
        seed=np.asarray([args.seed]),
        split=np.asarray([args.split]),
    )
    torch.save(
        {"state_dict": model.state_dict(), "meta": result},
        os.path.join(s15.CKPT_DIR, f"{tag}.pt"),
    )
    if args.task == "binary":
        detail = (
            f"ROC-AUC {metrics['roc_auc']:.4f} PR-AUC {metrics['pr_auc']:.4f} "
            f"F1 {metrics['f1']:.4f} P@1% {metrics['precision_at_topk']:.4f}"
        )
    else:
        detail = (
            f"Acc {metrics['accuracy']:.4f} "
            f"macro-F1 {metrics['macro_f1']:.4f}"
        )
    s15.log(
        f"RESULT {tag}: {detail} "
        f"({info['train_seconds']:.0f}s, {info['epochs_run']} ep, "
        f"best {info['best_epoch']})"
    )


if __name__ == "__main__":
    main()
