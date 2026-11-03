# FinTriage AI — project status

Updated: 2026-11-03

## COMPLETED

### Phase 1 — original project investigation
- Resolved both LinkedIn shortlinks. Verified the original repository as
  **`Razan3ssam/Consumer-Complaint-Classifier`** through the author's own post
  (ten lookalike repos exist from the same cohort). Full audit in
  `docs/ORIGINAL_PROJECT_AUDIT.md`. No LICENSE on the original → protocol reused,
  code written independently, attribution in the app and README.

### Phase 2/3 — data
- Dataset acquired: Kaggle `shashwatwork/consume-complaints-dataset-fo-nlp`,
  `complaints_processed.csv`, **162,421 rows** (matches the original's claim),
  SHA-256 `a120ac8450cb…`, 95 MB. Public endpoint, no Kaggle credentials needed.
  Provenance in `results/metrics/dataset_provenance.json`.
- Cleaning: 162,421 → **124,328** rows. Nulls 10 and exact duplicates 37,939 both
  match the original exactly; we additionally drop 144 complaints under 3 tokens
  and truncate 850 very long ones (the original kept them).
- **Near-duplicate finding:** 9,844 groups covering **30,752 rows (24.7%)** are
  near-identical but not exact duplicates. Under the original's protocol these
  scatter across train/val/test, which inflates the test score. We group them
  with MinHash/LSH and keep each group inside one split. Our test numbers may
  therefore come out *below* the published ones — honestly so.
- Splits: stratified 80/10/10, seed 42, group-aware →
  **99,458 / 12,435 / 12,435**. Class shares match within 0.1 pp across splits
  and match the original's reported distribution exactly
  (credit_reporting 45.2%, debt_collection 16.9%, mortgages_and_loans 15.1%,
  credit_card 12.1%, retail_banking 10.8%).
- Leakage assertions (group spread and identical-narrative overlap) pass.

### Phase 8 (partial) — verification
- `.venv` on Python 3.12 with pinned deps (1.4 GB). torch 2.5.1, MPS available.
- **27/27 unit tests pass** (`tests/test_preprocessing.py`, `tests/test_split.py`):
  cleaning, label mapping, near-duplicate detection, vectorizer padding/OOV/
  round-trip, split proportions, stratification, determinism, leakage detection.
- Smoke runs completed for both model families before any expensive training:
  GRU on 2,000 examples (18.5 s) and DistilBERT on 400 examples. Both smoke
  artifacts were deleted so they cannot be mistaken for trained models.

### Code written and syntax-clean
`configs/config.yaml` · `src/config.py` · `src/data/download.py` ·
`src/preprocessing/{quality,split,vectorizer}.py` · `src/models/rnn.py` ·
`src/training/common.py` · `src/evaluation/metrics.py` ·
`scripts/{prepare_data,train_rnn,train_distilbert,evaluate,select_model}.py` ·
`src/inference/{predictor,batch,examples}.py` ·
`src/explainability/integrated_gradients.py` · `src/{app_theme,ui_blocks,charts}.py` ·
`app.py` · `notebooks/kaggle_train_distilbert.ipynb`

### Split membership frozen and verified
- `data/processed/split_ids.csv.gz` (637 KB, committed) holds the row-level split
  assignment keyed by position in the raw CSV. `scripts/freeze_splits.py` proved
  it reconstructs the exact parquet files the recurrent models trained on —
  per-split content SHA-256 **train `87b13eec…`, val `ce29e921…`, test
  `816bd218…`, all MATCH**.
- `scripts/materialize_splits.py` rebuilds those splits anywhere: it verifies the
  raw CSV's SHA-256 first, applies only the normalisation cleaning applied,
  checks the label mapping, and refuses to write if any fingerprint disagrees.
  Verified: 0 duplicate groups span splits, 0 narratives shared train/test.

### GitHub
- **`https://github.com/fariskishtah/fintriage-ai` — PRIVATE**, 2 commits.
  Contains code, configs, docs and the frozen manifest only: no raw narratives,
  no trained weights, no credentials (912 KB total).

### Kaggle GPU path ready
- `notebooks/kaggle_train_distilbert.ipynb` (29 cells) rebuilds the frozen splits
  rather than re-deriving them, asserts the dataset hash and all three
  fingerprints before spending GPU time, smoke-runs on 512 examples, deletes the
  smoke artifacts, then fine-tunes. Needs no secrets when the code is attached as
  a dataset.
- `scripts/kaggle_launch.py` drives it headlessly: uploads the code as a private
  Kaggle dataset, pushes the kernel with GPU + both datasets attached, polls, and
  extracts the artifacts back into the project. It refuses to upload anything
  credential-shaped.

## IN PROGRESS

- **SimpleRNN / LSTM / GRU training** on MPS, up to 8 epochs each, early stopping
  on validation macro F1. Log: `/tmp/rnn_train.log`.
  - SimpleRNN **done**: best val macro F1 **0.7486**, 1,179 s, 8 epochs, 2.58 M params
  - LSTM: running
  - GRU: queued

## BLOCKED — one action needed

- **DistilBERT fine-tuning.** 49.6 s/step on M2 MPS → ~86 h locally. Everything
  for the Kaggle run is built and pushed; the only thing missing is Kaggle API
  credentials on this machine:
  `~/.kaggle/kaggle.json` from https://www.kaggle.com/settings → API → Create New
  Token. Once it is in place, `python scripts/kaggle_launch.py --push` runs the
  whole thing with no further interaction.
- **Hugging Face Spaces deployment.** Needs a write-scoped `HF_TOKEN`. The local
  `hf` CLI is broken (`~/.local/bin/hf` points at a deleted interpreter).
- **Hugging Face Spaces deployment.** Needs a write-scoped `HF_TOKEN`. The local
  `hf` CLI is broken (`~/.local/bin/hf` points at a deleted interpreter).

## NOT STARTED

- [ ] `scripts/make_figures.py` (static figures for the README)
- [ ] `tests/test_inference.py`, `tests/test_batch.py`, `tests/test_app.py`
- [ ] Launch the app locally and inspect it, desktop and mobile widths
- [ ] README with real results
- [ ] HF Space + live verification

## Honest status

No model is trained yet. **Every metric in `docs/ORIGINAL_PROJECT_AUDIT.md` is
the original author's published result, labelled as reference only.** No number
in this repository is ours until `results/metrics/model_comparison.csv` exists.
