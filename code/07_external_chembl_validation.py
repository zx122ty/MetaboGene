#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Step 7 — Independent external validation against ChEMBL experimental binding data

Addresses main-text limitation §5.2(1): "negatives are unobserved pairs, not
confirmed non-interactions; no independent experimental resource was used."

External set design (independent of our literature-mined gold standard):
  * Positives ("externally validated interactors"): unique (metabolite CID, human
    UniProt target) pairs in ChEMBL with an experimental binding activity
    pChEMBL >= 6.0 (Kd/IC50/Ki <= 1 uM), where the metabolite belongs to the HMDB
    endogenous-metabolite set (local HMDB parquet -> PubChem CID).
  * Negatives: random (HMDB metabolite, ChEMBL-observed human target) pairs that are
    NOT in our gold positives and NOT among ChEMBL actives — same "unreported"
    semantics as the internal negative model (mirrors Step 3 logic on an external
    universe), 1:1.

Two evaluation views:
  A) Unseen-pair ranking: pairs with NO overlap to our training pairs. We report
     Precision@K (K = 1/5/10% of actives) and the fraction of actives ranked above
     the median negative score — a rank-based metric robust to threshold choice.
  B) External binary AUC: actives vs external negatives, scored by the same model.

Model: retrain the best configuration (ECFP4 + ESM-2, Linear(2560->1024) protein
projection, DualTaskMLP) on the SAME internal split/seed as Step 5 so that scoring
is exactly the released pipeline; only the evaluation set is external.

Stages (run separately, resumable):
  python code/07_external_chembl_validation.py --stage build     # ChEMBL -> dataset parquet + stats
  python code/07_external_chembl_validation.py --stage score     # features + model + metrics
Deliverables:
  data/external/chembl_ext_dataset.parquet, chembl_ext_stats.json
  results/step7_external_chembl_metrics.json, step7_external_audit.csv
"""
import argparse
import json
import os
import random
import sqlite3
import time

import numpy as np
import pandas as pd

import config as C

cfg = C.load()
BASE = cfg.base_dir
EXT_DIR = cfg.get_path("paths.external_dir")
os.makedirs(EXT_DIR, exist_ok=True)
RES_DIR = cfg.get_path("paths.results_dir")

SEED = 42
PACHEMBL_MIN = 6.0          # active threshold: Kd/IC50/Ki <= 1 uM
STANDARD_TYPES = ("IC50", "KI", "KD", "EC50", "AC50")
ACTIVITY_CODES = (1, 2)     # ChEMBL activity_property_value standard_relation: '=' / '<'

HMDB_PARQUET = cfg.get_path("paths.hmdb_metabolites")
CHEMBL_DB = os.path.join(EXT_DIR, "chembl_37.db")
DATASET = os.path.join(EXT_DIR, "chembl_ext_dataset.parquet")


# --------------------------------------------------------------------------- #
# stage: build
# --------------------------------------------------------------------------- #
def load_hmdb():
    """Return (cid_set, inchikey_to_cid) from the local HMDB metabolite library.

    inchikey_to_cid: deterministic first-CID per InChIKey (sorted by CID).
    """
    hmdb = pd.read_parquet(HMDB_PARQUET, columns=["pubchem_compound_id", "inchikey"])
    df = hmdb.copy()
    df["cid"] = pd.to_numeric(df["pubchem_compound_id"].astype(str).str.strip(), errors="coerce")
    df["ik"] = df["inchikey"].astype(str).str.strip()
    df = df[df["cid"].notna() & ~df["ik"].isin(["", "nan", "None"])]
    cid_set = set(df["cid"].astype("int64").unique())
    ik_map = (df.dropna(subset=["ik"]).sort_values("cid")
              .drop_duplicates("ik").set_index("ik")["cid"].astype("int64").to_dict())
    return cid_set, ik_map


def _preflight(con):
    """Verify required tables/columns exist in this ChEMBL release (schema drift guard)."""
    cur = con.cursor()
    def cols(table):
        return {r[1].lower() for r in cur.execute(f"PRAGMA table_info({table})")}
    need = {
        "activities": {"molregno", "assay_id", "standard_type", "pchembl_value"},
        "assays": {"assay_id", "tid"},
        "target_dictionary": {"tid", "organism"},
        "target_components": {"tid", "component_id"},
        "component_sequences": {"component_id", "accession", "component_type"},
        "compound_structures": {"molregno", "canonical_smiles", "standard_inchi_key"},
    }
    for table, req in need.items():
        have = cols(table)
        missing = req - have
        assert not missing, f"table {table} missing columns {missing}; have={sorted(have)[:12]}..."
    print("[07-build] preflight OK (InChIKey-based metabolite join; no hetxref needed)")


def build_dataset():
    assert os.path.exists(CHEMBL_DB), f"missing {CHEMBL_DB}; extract the sqlite tarball first"
    cid_set, ik_map = load_hmdb()
    print(f"[07-build] HMDB metabolite CIDs: {len(cid_set):,}; InChIKey->CID map: {len(ik_map):,}")

    con = sqlite3.connect(CHEMBL_DB)
    _preflight(con)

    # human protein targets with UniProt accessions (universe for negative sampling)
    q_targets = """
        SELECT DISTINCT td.tid, cs.accession AS uniprot_acc
        FROM target_dictionary td
        JOIN target_components tc ON td.tid = tc.tid
        JOIN component_sequences cs ON tc.component_id = cs.component_id
        WHERE td.organism LIKE '%Homo sapiens%'
          AND cs.component_type = 'PROTEIN'
          AND cs.accession IS NOT NULL
    """
    targets = pd.read_sql_query(q_targets, con)
    print(f"[07-build] human protein (tid, UniProt) components in ChEMBL: {len(targets):,}")

    # binding activities on human protein targets.
    # NOTE: (a) in the SQLite export activities.toid is mostly NULL, so the target
    # link goes through assays (activities.assay_id -> assays.tid); (b) this shared
    # network filesystem makes indexed random I/O pathologically slow, so we run a
    # fast 3-table SEQUENTIAL scan (no component/structure joins in SQL) and do the
    # tid->accession + structure mapping in pandas on the small active set.
    q_act = f"""
        SELECT a.molregno, td.tid AS tid, a.pchembl_value
        FROM activities a
        JOIN assays asy ON a.assay_id = asy.assay_id
        JOIN target_dictionary td ON asy.tid = td.tid
        WHERE a.standard_type IN {STANDARD_TYPES}
          AND a.pchembl_value >= {PACHEMBL_MIN}
          AND td.organism LIKE '%Homo sapiens%'
    """
    t0 = time.time()
    act = pd.read_sql_query(q_act, con)
    print(f"[07-build] human-target binding activities at pChEMBL>={PACHEMBL_MIN}: "
          f"{len(act):,} (SQL {time.time()-t0:.1f}s)")

    # keep only protein targets with a UniProt accession (tid -> accession)
    tid2acc = (targets.drop_duplicates("tid")
               .set_index("tid")["uniprot_acc"].astype(str).to_dict())
    act["uniprot_acc"] = act["tid"].map(tid2acc)
    act = act[act["uniprot_acc"].notna()].copy()
    print(f"[07-build] on human protein targets (UniProt accession): {len(act):,}")

    # structure + InChIKey for the active molregnos only (small)
    mols = sorted(act["molregno"].unique().tolist())
    cstr_parts = []
    CHUNK = 5000  # stay under SQLite's bound-parameter limit
    for i in range(0, len(mols), CHUNK):
        chunk = mols[i:i + CHUNK]
        ph = ",".join("?" * len(chunk))
        cstr_parts.append(pd.read_sql_query(
            f"SELECT molregno, canonical_smiles, standard_inchi_key "
            f"FROM compound_structures WHERE molregno IN ({ph})", con, params=chunk))
    cstr = pd.concat(cstr_parts, ignore_index=True) if cstr_parts else pd.DataFrame(
        columns=["molregno", "canonical_smiles", "standard_inchi_key"])
    act = act.merge(cstr, on="molregno", how="left")

    # metabolite membership via InChIKey -> PubChem CID (local HMDB map)
    act["cid"] = act["standard_inchi_key"].map(ik_map).astype("Int64")
    n_ik_hit = int(act["cid"].notna().sum())
    print(f"[07-build] activities on HMDB-metabolite compounds (InChIKey hit): {n_ik_hit:,} "
          f"({100*n_ik_hit/max(1,len(act)):.2f}%)")
    act = act[act["cid"].notna()].copy()

    # all remaining rows are active by construction (threshold applied in SQL)
    act["is_active"] = 1
    actives = act.copy()
    actives = actives.sort_values("pchembl_value", ascending=False)
    keep = ~actives.duplicated(subset=["cid", "uniprot_acc"], keep="first")
    actives = actives[keep]
    print(f"[07-build] active (CID, UniProt) pairs: {len(actives):,}")

    # contamination control: exclude pairs already in internal gold positives
    gold = pd.read_parquet(cfg.get_path("paths.gold_positive"),
                           columns=["metabolite_pubchem_cid", "gene_entrez_id"])
    gold_pairs = set(zip(gold["metabolite_pubchem_cid"].astype("int64"),
                         gold["gene_entrez_id"].astype("int64")))
    up_map_path = cfg.get_path("paths.gold_with_uniprot")
    u2e = {}
    if os.path.exists(up_map_path):
        gu = pd.read_parquet(up_map_path, columns=["uniprot_id", "gene_entrez_id"]).dropna()
        u2e = dict(zip(gu["uniprot_id"].astype(str), gu["gene_entrez_id"].astype("int64")))
    actives["entrez"] = actives["uniprot_acc"].map(u2e)
    actives["in_internal_gold"] = [
        (bool(pd.notna(e)) and (c, int(e)) in gold_pairs)
        for c, e in zip(actives["cid"], actives["entrez"])]
    n_overlap = int(actives["in_internal_gold"].sum())
    print(f"[07-build] active pairs overlapping internal gold: {n_overlap:,} "
          f"({100*n_overlap/max(1,len(actives)):.2f}%)")
    unseen = actives[~actives["in_internal_gold"]].copy()

    # negative universe: HMDB-metabolite CIDs x ChEMBL human protein targets
    neg_targets = targets[targets["uniprot_acc"].notna()] \
        .drop_duplicates("uniprot_acc")[["tid", "uniprot_acc"]]
    con.close()

    rng = random.Random(SEED)
    n_neg = len(unseen)
    active_pair_set = set(zip(actives["cid"].astype("int64"), actives["uniprot_acc"].astype(str)))
    neg_rows, seen_neg, attempts = [], set(), 0
    while len(neg_rows) < n_neg and attempts < n_neg * 40:
        attempts += 1
        c = rng.choice(list(cid_set))
        t = neg_targets.iloc[rng.randrange(len(neg_targets))]
        key = (int(c), str(t["uniprot_acc"]))
        if key in seen_neg or key in active_pair_set:
            continue
        e = u2e.get(str(t["uniprot_acc"]))
        if pd.notna(e) and (c, int(e)) in gold_pairs:
            continue
        seen_neg.add(key)
        neg_rows.append({"cid": int(c), "uniprot_acc": str(t["uniprot_acc"]),
                         "pchembl_value": np.nan, "is_active": 0})
    print(f"[07-build] external negatives sampled: {len(neg_rows):,} (attempts {attempts:,})")

    act_out = unseen.rename(columns={"cid": "metabolite_pubchem_cid",
                                     "uniprot_acc": "target_uniprot"}) \
                    [["metabolite_pubchem_cid", "target_uniprot", "pchembl_value",
                      "canonical_smiles", "entrez"]] \
                    .assign(is_active=1)
    neg_out = pd.DataFrame(neg_rows).rename(columns={"cid": "metabolite_pubchem_cid",
                                                     "uniprot_acc": "target_uniprot"}) \
                    [["metabolite_pubchem_cid", "target_uniprot", "pchembl_value"]] \
                    .assign(canonical_smiles=None, entrez=None, is_active=0)
    ds = pd.concat([act_out, neg_out], ignore_index=True)
    ds.to_parquet(DATASET, index=False)

    stats = {
        "chembl_release": "37",
        "pchembl_min": PACHEMBL_MIN,
        "standard_types": list(STANDARD_TYPES),
        "hmdb_metabolite_cids": len(cid_set),
        "hmdb_inchikey_to_cid_map": len(ik_map),
        "human_protein_components_chembl": int(len(targets)),
        "external_actives": int(len(unseen)),
        "actives_overlap_internal_gold_excluded": n_overlap,
        "external_negatives": int(len(neg_rows)),
        "seed": SEED,
        "metabolite_membership": ("InChIKey join: compound_structures.standard_inchi_key -> "
                                  "local HMDB inchikey -> PubChem CID (deterministic first CID)"),
        "note": ("positives = ChEMBL experimental binding pChEMBL>=6 on HMDB endogenous "
                 "metabolites x human UniProt targets, disjoint from internal gold pairs; "
                 "negatives = random unreported (HMDB metabolite, ChEMBL-observed human "
                 "protein target) pairs excluding internal-gold and ChEMBL-active pairs"),
    }
    with open(os.path.join(EXT_DIR, "chembl_ext_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"[07-build] saved {DATASET} + chembl_ext_stats.json")


def pubchem_smiles_to_cids(smiles_list, sleep_s=0.15):
    """SMILES -> PubChem CID via PUG REST (one GET per unique SMILES; PubChem has
    no batched SMILES->CID endpoint). Returns {smiles: cid} for resolved entries."""
    import time, urllib.parse, urllib.request
    UA = {"User-Agent": "Mozilla/5.0 (metabo-gene-research)"}
    out = {}
    uniq = list(dict.fromkeys(smiles_list))
    for j, s in enumerate(uniq):
        url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
               f"smiles/{urllib.parse.quote(s, safe='')}/cids/JSON")
        ok = False
        for attempt in range(3):
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                            timeout=60) as r:
                    d = json.load(r)
                cids = d.get("IdentifierList", {}).get("CID", [])
                if isinstance(cids, int):
                    out[s] = cids
                elif cids:
                    out[s] = cids[0]
                ok = True
                break
            except Exception as ex:
                time.sleep(3)
        if not ok:
            pass  # unresolved SMILES simply stay absent from `out`
        if (j + 1) % 200 == 0:
            print(f"  smiles->cids progress: {j+1}/{len(uniq)} (resolved {len(out)})")
        time.sleep(sleep_s)
    return out


# --------------------------------------------------------------------------- #
# stage: score  (imports torch lazily to keep build stage lightweight)
# --------------------------------------------------------------------------- #
def score_external():
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    import torch.nn as nn
    from sklearn.metrics import roc_auc_score, average_precision_score

    # --- features (reuse Step 4 components) ---
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "enc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "04_encode_features.py"))
    enc = importlib.util.module_from_spec(spec)
    # avoid running 04 main: it only runs under __main__, safe to exec module
    spec.loader.exec_module(enc)

    ds = pd.read_parquet(DATASET)
    print(f"[07-score] external dataset: {len(ds):,} rows "
          f"({int(ds['is_active'].sum()):,} actives)")

    # SMILES for negatives (actives carry ChEMBL canonical SMILES; negatives fetched)
    need = ds[ds["canonical_smiles"].isna()]["metabolite_pubchem_cid"].astype("int64").unique()
    cache_path = os.path.join(EXT_DIR, "pubchem_smiles_ext.json")
    cache = {}
    if os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    missing = [int(c) for c in need if str(int(c)) not in cache]
    if missing:
        import time, urllib.parse, urllib.request
        print(f"[07-score] fetching {len(missing):,} SMILES from PubChem (POST batches) ...")
        UA = {"User-Agent": "Mozilla/5.0 (metabo-gene-research)"}
        for i in range(0, len(missing), 200):
            batch = missing[i:i+200]
            # GET with comma-joined CIDs in path (matches internal Step 2; POST form
            # is rejected by PubChem PUG REST).
            url = ("https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                   f"cid/{','.join(map(str, batch))}/property/CanonicalSMILES/JSON")
            ok = False
            for attempt in range(3):
                try:
                    req = urllib.request.Request(url, headers=UA)
                    with urllib.request.urlopen(req, timeout=60) as r:
                        d = json.load(r)
                    props = d.get("PropertyTable", {}).get("Properties", [])
                    if not props:  # no CIDs resolved at all -> retry is pointless
                        ok = True
                        break
                    for rec in props:
                        smi = rec.get("CanonicalSMILES") or rec.get("ConnectivitySMILES")
                        if smi:
                            cache[str(rec["CID"])] = smi
                    ok = True
                    break
                except Exception as ex:
                    print(f"  batch {i//200} attempt {attempt+1} failed: {ex}")
                    time.sleep(5)
            if not ok:
                print(f"  batch {i//200}: giving up")
            json.dump(cache, open(cache_path, "w"))
            time.sleep(0.3)
    pubcache = {str(k): v for k, v in cache.items()}
    cid_str = ds["metabolite_pubchem_cid"].astype("int64").astype(str)
    canon = ds["canonical_smiles"]
    ds["smiles"] = [c if isinstance(c, str) else pubcache.get(cs)
                    for c, cs in zip(canon.tolist(), cid_str)]

    # protein sequences from local UniProt FASTA (reuse 02b loader)
    print("[07-score] loading local UniProt FASTA ...")
    fasta = enc.load_fasta() if hasattr(enc, "load_fasta") else None
    if fasta is None:
        import gzip
        fasta = {}
        acc, buf = None, []
        with gzip.open(cfg.get_path("paths.uniprot_fasta"), "rt") as f:
            for line in f:
                if line.startswith(">"):
                    if acc:
                        fasta[acc] = "".join(buf)
                    acc = line[1:].split("|")[1]
                    buf = []
                elif acc:
                    buf.append(line.strip())
        if acc:
            fasta[acc] = "".join(buf)
    ds["protein_seq"] = ds["target_uniprot"].map(fasta)
    print(f"[07-score] smiles cov {ds['smiles'].notna().mean():.3f}  "
          f"seq cov {ds['protein_seq'].notna().mean():.3f}")

    # encode: ECFP4 (2048) + ESM-2 (2560, cached; L2 norm) — identical to features_esm.npz
    X_mol, has_mol = enc.encode_smiles_fp(ds["smiles"].tolist())
    emb_path = os.path.join(cfg.get_path("paths.esm_cache_dir"), "embeddings.npy")
    ids_path = os.path.join(cfg.get_path("paths.esm_cache_dir"), "ids.json")
    esm_emb = np.load(emb_path)
    seq_to_idx = {r["seq"]: r["index"] for r in json.load(open(ids_path))}
    X_seq, has_seq = enc.encode_seq(ds["protein_seq"].tolist(), protein_type="esm",
                                    esm_cache=(esm_emb, seq_to_idx), l2_norm=True)
    X = np.hstack([X_mol, X_seq]).astype("float32")
    print(f"[07-score] feature matrix {X.shape}")

    # --- retrain best model on the SAME internal split (deterministic) ---
    import numpy.random as npr
    random.seed(SEED); npr.seed(SEED); torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED) if torch.cuda.is_available() else None
    device = "cuda" if torch.cuda.is_available() else "cpu"

    feat_name = cfg.get("training.feat_file_map")["esm"]
    d = np.load(os.path.join(cfg.get_path("paths.features_dir"), feat_name))
    l = np.load(cfg.sidecar_npz("labels", feat_name))
    s = np.load(cfg.sidecar_npz("split", feat_name))
    Xtr = torch.tensor(d["X_binary"][s["b_tr"]]).float().to(device)
    ytr = torch.tensor(l["y_binary"][s["b_tr"]]).to(device)
    Xva = torch.tensor(d["X_binary"][s["b_va"]]).float().to(device)
    yva = torch.tensor(l["y_binary"][s["b_va"]]).to(device)

    BS = cfg.get("training.batch_size"); LR = cfg.get("training.lr")
    EPOCHS = cfg.get("training.epochs"); PATIENCE = cfg.get("training.patience")
    HIDDEN = cfg.get("training.hidden_dim"); DROP = cfg.get("training.dropout")
    FP_DIM, PROT_DIM, PROJ = 2048, 2560, 1024

    class DualTaskMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.esm_proj = nn.Linear(PROT_DIM, PROJ)
            self.trunk = nn.Sequential(
                nn.Linear(FP_DIM + PROJ, HIDDEN), nn.ReLU(), nn.Dropout(DROP),
                nn.Linear(HIDDEN, HIDDEN), nn.ReLU(), nn.Dropout(DROP))
            self.binary_head = nn.Linear(HIDDEN, 1)

        def forward(self, x):
            h = self.trunk(torch.cat([x[:, :FP_DIM], self.esm_proj(x[:, FP_DIM:])], dim=1))
            return self.binary_head(h).squeeze(-1)

    model = DualTaskMLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    crit = nn.BCEWithLogitsLoss()
    best_va, best_state, bad = -1, None, 0
    for ep in range(EPOCHS):
        model.train()
        order = npr.permutation(len(Xtr))
        for i in range(0, len(order), BS):
            b = order[i:i+BS]
            opt.zero_grad()
            loss = crit(model(Xtr[b]), ytr[b].float())
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            ova = model(Xva)
            va_acc = float(((ova.sigmoid() > 0.5).float() == yva).float().mean())
        if va_acc > best_va:
            best_va, bad = va_acc, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    model.load_state_dict(best_state)
    model.eval()

    # internal sanity check on the same test split (must reproduce ~0.9217)
    Xte = torch.tensor(d["X_binary"][s["b_te"]]).float().to(device)
    yte = torch.tensor(l["y_binary"][s["b_te"]]).cpu().numpy()
    with torch.no_grad():
        p_int = model(Xte).sigmoid().cpu().numpy()
    int_roc = roc_auc_score(yte, p_int)
    print(f"[07-score] internal test ROC-AUC sanity: {int_roc:.4f} (expect ~0.9217)")

    # --- score external set ---
    Xe = torch.tensor(X).float().to(device)
    with torch.no_grad():
        pe = model(Xe).sigmoid().cpu().numpy()
    ds["pred_prob"] = pe
    act = ds[ds["is_active"] == 1]
    neg = ds[ds["is_active"] == 0]

    # A) unseen-pair ranking metrics (all external actives are unseen by construction)
    scores_act, scores_neg = act["pred_prob"].values, neg["pred_prob"].values
    res = {"internal_test_roc_auc_sanity": round(float(int_roc), 4),
           "n_external_actives": int(len(act)), "n_external_negatives": int(len(neg))}
    # precision@K defined over the ranked external set (actives + negatives):
    for kfrac in (0.01, 0.05, 0.10):
        k = max(1, int(kfrac * len(act)))
        order = ds.reindex(ds["pred_prob"].sort_values(ascending=False).index)
        topk = order.head(k)
        res[f"precision_at_top{int(kfrac*100)}pct_of_actives"] = \
            round(float(topk["is_active"].mean()), 4)
    # rank statistic: fraction of actives with score > median negative score
    med_neg = float(np.median(scores_neg))
    res["frac_actives_above_median_neg"] = round(float((scores_act > med_neg).mean()), 4)
    # B) external binary AUC / PR-AUC
    y_ext = ds["is_active"].values
    res["external_roc_auc"] = round(float(roc_auc_score(y_ext, pe)), 4)
    res["external_pr_auc"] = round(float(average_precision_score(y_ext, pe)), 4)

    with open(os.path.join(RES_DIR, "step7_external_chembl_metrics.json"), "w") as f:
        json.dump(res, f, indent=2)
    print(f"[07-score] metrics: {json.dumps(res, indent=2)}")

    # audit table: top-50 external actives by predicted probability + 20 random negatives
    audit = act.nlargest(50, "pred_prob").copy()
    audit = pd.concat([audit, neg.sample(20, random_state=SEED)], ignore_index=True)
    audit[["metabolite_pubchem_cid", "target_uniprot", "pchembl_value",
           "is_active", "pred_prob"]].to_csv(os.path.join(RES_DIR, "step7_external_audit.csv"),
                                             index=False)
    print("[07-score] audit table saved -> results/step7_external_audit.csv")


def api_smoke_test():
    """Thin validation: run build_dataset() on the local ChEMBL SQLite DB.

    (The earlier REST-API smoke test was replaced by this DB-based build, which is
    deterministic and far faster; InChIKey join avoids PubChem rate limits.)
    """
    build_dataset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["build", "score", "api-smoke"], required=True)
    args = ap.parse_args()
    if args.stage == "build":
        build_dataset()
    elif args.stage == "score":
        score_external()
    else:
        api_smoke_test()


if __name__ == "__main__":
    main()
