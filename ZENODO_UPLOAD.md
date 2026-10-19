# Zenodo upload package

The release is organized as one Zenodo record named `MetaboGene-zenodo-v1`. Upload the completed `zenodo_upload/` directory as `MetaboGene-zenodo-v1.zip`. A single archive keeps the directory structure intact when the record is downloaded.

The intended record contains:

- processed and source data;
- frozen ESM-2 3B weights needed for local reproduction;
- the reviewed UniProt FASTA;
- current model checkpoints and saved predictions;
- result tables needed to verify the reported metrics;
- the manuscript, supplementary material, figures, and supplementary tables.

The required package is about 22 GB. It stays below Zenodo's standard 50 GB record limit.

## Repository mapping

After downloading and extracting the archive, copy its contents into the code repository:

```text
zenodo_upload/data/       -> data/
zenodo_upload/results/    -> results/
zenodo_upload/external/   -> external/
zenodo_upload/esm2/       -> ESM2/esm2_t36_3B_UR50D/
zenodo_upload/paper/      -> optional paper bundle
```

## Data files

| File in the archive | Approx. size | Purpose |
|---|---:|---|
| `data/source/merged_standardized_tree.parquet` | 1.2 GB | Raw 1.61M-statement PMDdb source table |
| `data/gold/gold_positive.parquet` | 5 MB | Curated interaction positives |
| `data/gold/gold_positive_with_uniprot.parquet` | 5.5 MB | Positives with UniProt mapping |
| `data/gold/gold_with_struct_fullseq.parquet` | 118 MB | Positives with SMILES and full protein sequences |
| `data/gold/gold_stats.json` | 684 B | Curation statistics |
| `data/gold/seq_coverage.json` | 95 B | Sequence coverage |
| `data/neg/positives.parquet` | 72 MB | Model-ready positive rows |
| `data/neg/negatives.parquet` | 162 MB | Hard and random negative rows |
| `data/neg/neg_stats.json` | 880 B | Negative-sampling statistics |
| `data/neg/gen_struct_coverage.json` | 83 B | Negative structural coverage |

## Feature files

| File in the archive | Approx. size | Purpose |
|---|---:|---|
| `data/features/features.npz` | 186 MB | ECFP4 and reference features |
| `data/features/esm_3B/embeddings.npy` | 269 MB | Frozen ESM-2 protein embeddings |
| `data/features/esm_3B/ids.json` | 17 MB | Sequence-to-embedding index |
| `data/features/esm_3B/esm_stats.json` | 141 B | Embedding statistics |
| `data/features/labels_features_esm.npz` | 57 KB | Binary and direction labels |
| `data/features/split_features_esm.npz` | 1.8 MB | Row-level splits |
| `data/features/split_strict_features_esm.npz` | 3.1 MB | Pair-disjoint split |
| `data/features/sota_cache/` | 673 MB | Token, graph, FASTA, vocabulary, and ESM lookup caches |
| `data/cache/pubchem_smiles.json` | 1.9 MB | CID-to-SMILES lookup |

## External resources

| File in the archive | Approx. size | Purpose |
|---|---:|---|
| `external/uniprot/uniprot_sprot.fasta.gz` | 90 MB | Reviewed UniProt FASTA used for sequence mapping |
| `external/HMDB/hmdb_metabolites.parquet` | 131 MB | HMDB metabolite library used for external validation |
| `external/chembl/chembl_37_sqlite.tar.gz` | 5.7 GB | Official ChEMBL 37 SQLite archive |
| `data/external/chembl_ext_dataset.parquet` | 131 KB | Prepared ChEMBL external evaluation set |
| `data/external/chembl_ext_stats.json` | 813 B | ChEMBL external-set statistics |
| `data/external/unified_ext/build_stats.json` | 951 B | External-source build statistics |
| `data/external/unified_ext/ctd_pairs.parquet` | 36 KB | Prepared CTD direction pairs |
| `data/external/unified_ext/hmdb_assoc_pairs.parquet` | 541 KB | Prepared HMDB association pairs |
| `data/external/unified_ext/pdb_pairs.parquet` | 56 KB | Prepared PDB co-crystal pairs |
| `data/external/unified_ext/reactome_pairs.parquet` | 4.6 KB | Prepared Reactome direction pairs |

The archive does not contain DrugBank source files or DrugBank-derived pair tables because access is licence-controlled. Obtain an authorised DrugBank export and run the scripts in `code/07b_drugbank_smiles.py` and `code/10_external_sources_validation.py`.

After extracting ChEMBL:

```bash
mkdir -p data/external
tar -xzf external/chembl/chembl_37_sqlite.tar.gz -C data/external
ln -sfn chembl_37/chembl_37_sqlite/chembl_37.db data/external/chembl_37.db
```

## ESM-2 weights

The archive includes the PyTorch ESM-2 3B files from `facebook/esm2_t36_3B_UR50D` (MIT licence):

```text
esm2/esm2_t36_3B_UR50D/config.json
esm2/esm2_t36_3B_UR50D/pytorch_model.bin.index.json
esm2/esm2_t36_3B_UR50D/pytorch_model-00001-of-00002.bin
esm2/esm2_t36_3B_UR50D/pytorch_model-00002-of-00002.bin
esm2/esm2_t36_3B_UR50D/special_tokens_map.json
esm2/esm2_t36_3B_UR50D/tokenizer_config.json
esm2/esm2_t36_3B_UR50D/vocab.txt
esm2/esm2_t36_3B_UR50D/README.md
esm2/esm2_t36_3B_UR50D/.gitattributes
```

TensorFlow copies and `.hfd` download metadata are excluded.

## Model checkpoints

The archive includes:

```text
results/sota_ckpt/graph_ecfp_esm_row_s0_b512.pt
results/sota_ckpt/graph_ecfp_esm_row_s1_b512.pt
results/sota_ckpt/graph_ecfp_esm_row_s2_b512.pt
results/sota_ckpt/graph_ecfp_esm_row_s42_b512.pt

results/sota_ckpt/graph_ecfp_esm_direction_row_s0_b128.pt
results/sota_ckpt/graph_ecfp_esm_direction_row_s1_b128.pt
results/sota_ckpt/graph_ecfp_esm_direction_row_s2_b128.pt
results/sota_ckpt/graph_ecfp_esm_direction_row_s42_b128.pt

results/sota_ckpt/graph_ecfp_esm_strict_s0_b128.pt
results/sota_ckpt/graph_ecfp_esm_strict_s1_b128.pt
results/sota_ckpt/graph_ecfp_esm_strict_s42_b128.pt
results/sota_ckpt/graph_ecfp_esm_direction_strict_s42_drop03.pt
```

The baseline checkpoints under `results/sota_ckpt/` are also included, except smoke-test files. They allow the DeepDTA, GraphDTA, and MolTrans comparisons to be reproduced without retraining.

## Predictions and metrics

Upload these directories in full:

```text
results/sota_preds/
results/sota_runs/
results/sota_curves/
```

The archive also contains the metric files needed by `code/17_make_ensembles.py`, `code/18_graph_uncertainty_eval.py`, `code/19_equal_ensemble_comparison.py`, and `code/_verify_optimized_model.py`.

## Paper bundle

```text
paper/manuscript_261017.md
paper/supplementary.md
paper/figures/
paper/supplementary_tables/
```

Only final figure files are included. Alignment overlays, collision-audit PDFs, old versions, and superseded figure panels are excluded.

## Excluded large resources

The package does not include:

- `data/features/features_esm.npz` and other historical 3.6–5.6 GB feature matrices;
- `data/features/cache_unimol2/` or the Uni-Mol2 1.1B checkpoint;
- extracted ChEMBL SQLite (`31.6 GB`), because the compressed official archive is included;
- DrugBank source data or DrugBank-derived pair files;
- logs, smoke runs, `.polluted.bak` files, and temporary caches.

These exclusions keep the record below the standard Zenodo size limit. The required current Graph workflow can be run from the included files.

## Zenodo metadata checklist

Before publishing, complete these fields in Zenodo:

| Field | Suggested value |
|---|---|
| Title | `MetaboGene: a direction- and uncertainty-aware model for metabolite–target interaction prediction with literature-traceable supervision` |
| Resource type | Dataset |
| Version | `1.0.0` |
| Creators | Author names and affiliations in publication order |
| Description | Copy the project description from `README.md` and mention the GitHub code repository |
| Keywords | metabolite–target interaction; regulatory direction; knowledge graph; ESM-2; uncertainty; drug–target interaction |
| Related identifiers | GitHub repository URL; manuscript DOI when available |
| Licence | Select the authors' chosen licence and confirm it is compatible with third-party data |

Do not publish the Zenodo record until the HMDB, ChEMBL, UniProt, ESM-2, and manuscript distribution terms have been checked.

## Prepared archive

The upload-ready files are:

```text
../zenodo_upload/
../MetaboGene-zenodo-v1.zip
../MetaboGene-zenodo-v1.zip.sha256
```

The archive passed `unzip -t`, and `zenodo_upload/SHA256SUMS` passed file-by-file verification.
