# MetaboGene

Direction- and uncertainty-aware metabolite–target interaction prediction with literature-traceable supervision.

MetaboGene predicts whether a metabolite and a protein interact, and whether the reported regulation is activation, inhibition, or neutral. Inputs are a metabolite SMILES string and a protein sequence. The model family is trained on PMDdb, a literature-traceable metabolite–target knowledge graph.

This repository contains the code used to build the dataset, train and evaluate the models, run SOTA baselines, perform external validation, and compute uncertainty. Datasets, feature caches, model checkpoints, ESM-2 weights, and generated results are distributed through Zenodo.

## Current results

| Task | Model | Main metric |
|---|---|---:|
| Interaction, row-level | MetaboGene-Graph-E4 | ROC-AUC 0.9449, PR-AUC 0.9528 |
| Interaction, pair-disjoint | MetaboGene-Graph-E3 | ROC-AUC 0.9307, PR-AUC 0.9395 |
| Direction, row-level | MetaboGene-Graph-E4 | Accuracy 0.8492, macro-F1 0.8441 |
| Direction, pair-disjoint | MetaboGene-Graph-D30 | Accuracy 0.6915, macro-F1 0.6799 |
| ChEMBL external validation | MetaboGene-Graph-E4 | ROC-AUC 0.8837, PR-AUC 0.8931 |
| Error detection, row-level | MetaboGene-Graph-E4 | Predictive-entropy AUROC 0.8111 |

`MetaboGene-Graph` combines a GIN molecular graph, ECFP4 fingerprints, and frozen ESM-2 protein embeddings. `-E4` and `-E3` denote equal-probability ensembles over four and three seeds. Interaction and direction models use separate task-specific weights.

The Graph-E4 ensemble is numerically best, but its ROC-AUC difference from GraphDTA+ESM-E4 is small and not statistically significant (`+0.0006`, DeLong `p=0.47`). The matched single-model comparison supports the architecture gain; `code/19_equal_ensemble_comparison.py` reproduces both comparisons.

## Repository layout

```text
code/
  01-04*       Dataset construction and feature generation
  05-06        Reference MLP models and audit checks
  07-12        External dataset construction and validation
  13-14        Uncertainty analysis
  15*          DeepDTA, GraphDTA, MolTrans, and entity baselines
  16           MetaboGene-Graph training and external scoring
  17           Seed ensembles
  18           Graph uncertainty evaluation
  19           Equal-aggregation SOTA comparison
  config.yaml  Paths and experiment settings
README.md
ZENODO_UPLOAD.md
requirements.txt
```

No data or model weights are stored in Git. See [ZENODO_UPLOAD.md](ZENODO_UPLOAD.md) for the release inventory.

## 1. Install

Python 3.10 or newer is recommended. A CUDA GPU is optional for the reference MLP and required in practice for ESM-2 3B and full Graph-E4 training.

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA version from the PyTorch package index if the default wheel does not detect your GPU.

## 2. Download project assets

The release is distributed as one Zenodo record. After publishing it, replace `<ZENODO_RECORD_ID>` with the numeric record ID.

```bash
export ZENODO_RECORD_ID="your-record-id"

wget -O MetaboGene-zenodo-v1.zip \
  "https://zenodo.org/records/${ZENODO_RECORD_ID}/files/MetaboGene-zenodo-v1.zip?download=1"
mkdir -p zenodo_upload
unzip -q MetaboGene-zenodo-v1.zip -d zenodo_upload

mkdir -p data results external ESM2
rsync -a zenodo_upload/data/ data/
rsync -a zenodo_upload/results/ results/
rsync -a zenodo_upload/external/ external/
rsync -a zenodo_upload/esm2/esm2_t36_3B_UR50D/ ESM2/esm2_t36_3B_UR50D/
```

The archive preserves the `data/`, `results/`, `external/`, `ESM2/`, and `paper/` paths listed in [ZENODO_UPLOAD.md](ZENODO_UPLOAD.md). The frozen ESM-2 3B weights are included so the project does not depend on access to an external model hub.

The source PMDdb parquet and reviewed UniProt FASTA are included in the Zenodo archive.

## 3. Configure local paths

Open `code/config.yaml` and update these entries:

| Key | Expected location |
|---|---|
| `paths.source_parquet` | `data/source/merged_standardized_tree.parquet` |
| `paths.uniprot_fasta` | `external/uniprot/uniprot_sprot.fasta.gz` |
| `paths.hmdb_metabolites` | `external/HMDB/hmdb_metabolites.parquet` |
| `paths.esm_model_path` | `ESM2/esm2_t36_3B_UR50D` |
| `unimol2_embed.ckpt` | Uni-Mol2 1.1B checkpoint, only if reproducing Uni-Mol2 experiments |

Then check the configuration and required files:

```bash
python code/config.py
```

The Zenodo archive includes the reviewed UniProt FASTA, HMDB metabolite parquet, and the compressed ChEMBL 37 SQLite archive. Extract ChEMBL after copying the archive:

```bash
mkdir -p data/external
tar -xzf external/chembl/chembl_37_sqlite.tar.gz -C data/external
ln -sfn chembl_37/chembl_37_sqlite/chembl_37.db data/external/chembl_37.db
```

The external dataset builders may also use PDB, Reactome, and CTD. DrugBank files are not redistributed because access is licence-controlled; obtain an authorised local export and follow `code/07-10*.py`.

## 4. Verify the released predictions

This path uses the saved predictions in `results/` and does not retrain a model.

```bash
python code/17_make_ensembles.py
python code/18_graph_uncertainty_eval.py
python code/19_equal_ensemble_comparison.py
python code/_verify_optimized_model.py
```

Expected headline outputs include:

```text
Graph-E4 interaction ROC-AUC: 0.9449
Graph-E4 interaction PR-AUC:  0.9528
Graph-E4 direction accuracy:  0.8492
Graph-E3 strict ROC-AUC:      0.9307
Graph-E4 uncertainty AUROC:   0.8111
Graph-E4 vs GraphDTA+ESM-E4:  Delta ROC-AUC +0.000575, p=0.468
```

## 5. Reproduce Graph-E4

The Graph model reads ECFP4 features and the cached ESM-2 embeddings. The first run creates `data/features/sota_cache/`; the uploaded cache shortens this step.

```bash
for seed in 0 1 2 42; do
  python code/16_graph_fusion_optimize.py \
    --split row --task binary --seed "$seed" \
    --batch_size 512 --patience 10 \
    --tag "graph_ecfp_esm_row_s${seed}_b512"
done

for seed in 0 1 2 42; do
  python code/16_graph_fusion_optimize.py \
    --split row --task direction --seed "$seed" \
    --batch_size 128 --patience 10 \
    --tag "graph_ecfp_esm_direction_row_s${seed}_b128"
done

python code/17_make_ensembles.py
```

For the pair-disjoint interaction ensemble:

```bash
for seed in 0 1 42; do
  python code/16_graph_fusion_optimize.py \
    --split strict --task binary --seed "$seed" \
    --batch_size 128 --patience 10 \
    --tag "graph_ecfp_esm_strict_s${seed}_b128"
done

python code/17_make_ensembles.py
```

For the strict direction model with standard dropout 0.30:

```bash
python code/16_graph_fusion_optimize.py \
  --split strict --task direction --seed 42 \
  --batch_size 128 --dropout 0.3 --patience 10 \
  --tag graph_ecfp_esm_direction_strict_s42_drop03
```

`MetaboGene-Graph-D30` in the manuscript refers to this model. `MetaboGene-Graph-GD30` is a different variant that uses graph-branch dropout during interaction training.

## 6. Rebuild the dataset from source

The full path starts from the PMDdb source parquet and reviewed UniProt sequences:

```bash
python code/01_build_gold_dataset.py
python code/02_fetch_sequences.py
python code/02b_fill_protein_seq.py
python code/03_negative_sampling.py
python code/03b_fill_neg_struct.py

python code/04b_esm_embed.py
python code/04_encode_features.py
python code/04_encode_features.py --protein_encoder_type esm
```

The resulting files are written under `data/gold/`, `data/neg/`, and `data/features/`. Rebuilding the full ESM cache requires the ESM-2 3B weights and takes substantially longer than downloading the released cache.

## 7. Baselines and external validation

Baseline examples:

```bash
python code/15c_moltrans_official.py --help
python code/15_sota_baselines.py \
  --model graphdta --protein esm_frozen \
  --split row --seed 42
python code/15_sota_baselines.py --summarize
```

MolTrans code and ESPF vocabularies are not vendored. Clone the official repository at the tested commit:

```bash
git clone https://github.com/kexinhuang12345/MolTrans.git third_party/MolTrans_official
git -C third_party/MolTrans_official checkout 47ac16b8c158b080ba6cdaec74cd7aa9c1332b73
```

ChEMBL validation:

```bash
python code/07_external_chembl_validation.py --stage build
python code/07_external_chembl_validation.py --stage score
```

The other external sources use `code/08-10*.py`. External Graph scoring:

```bash
for seed in 0 1 2 42; do
  python code/16_graph_fusion_optimize.py \
    --external all \
    --checkpoint "results/sota_ckpt/graph_ecfp_esm_row_s${seed}_b512.pt" \
    --tag "metabogene_graph_s${seed}_external"
done

python code/16_graph_fusion_optimize.py --external ensemble
```

## Data and code availability

- PMDdb browser: <https://pmddb.qfxulab.com/>
- Code repository: `https://github.com/<GITHUB_OWNER>/MetaboGene`
- Data, model and paper archive: `https://doi.org/<ZENODO_DOI>`
- Release inventory: [ZENODO_UPLOAD.md](ZENODO_UPLOAD.md)

Absolute scores depend on the published negative-sampling model. Internal negatives are unreported pairs, not experimentally confirmed non-interactions. Direction labels also reflect statement-level literature semantics. These limitations are described in the manuscript and should be considered when applying the model outside the released evaluation sets.
