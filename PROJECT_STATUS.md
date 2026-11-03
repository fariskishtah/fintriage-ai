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
  - SimpleRNN **done**: val macro F1 **0.7486**, 1,179 s, 8 epochs, 2.58 M params
  - LSTM **done**: val macro F1 **0.8147**, 3,382 s, 8 epochs, 2.61 M params
  - GRU: running
- **DistilBERT** fine-tuning on a Kaggle Tesla T4. The first run failed in 28 s
  at the split-verification assert — Kaggle's dataset pipeline decompresses
  `.gz` on upload, so the manifest arrived as `split_ids.csv`. The assert did
  its job: it refused to train against a manifest it could not find. The lookup
  now accepts either name (verified locally against both; all three fingerprints
  still MATCH) and version 2 is running.

## BLOCKED

Nothing. Both remaining trainings are running; everything else is prepared.

### Tests
**89 passing, 30 skipped.** The skips are the real-checkpoint tests, which skip
rather than pass on a mock so a missing model cannot look like a green run.
### Hugging Face — authenticated
- CLI repaired **inside the project venv**: `huggingface_hub[cli]` 2.x pulls the
  browser-login flow but breaks the pinned `transformers` 4.46.3 and
  `tokenizers` 0.20.3. Pinned **0.34.4** instead — the oldest release that still
  ships the `hf` command and satisfies both (`pip check` clean). The 2.x CLI
  lives in a throwaway venv used only for the browser login; both read the same
  token store at `~/.cache/huggingface/token`.
- Logged in via the OAuth device flow (no token ever pasted into chat, printed,
  logged or committed). `hf auth whoami` → **fariskishtah**.
- Write access confirmed by creating
  **`fariskishtah/fintriage-distilbert` (private)** — the model repo the Space
  will pull weights from.
- The Space itself is created at deploy time, private, once a checkpoint exists.
  Nothing is published without explicit approval; free CPU tier only.

### Application defect found and fixed
Launching the app returned **HTTP 500 on every page load**:
`TypeError: argument of type 'bool' is not iterable` in
`gradio_client/utils.py:887`. Cause: `gr.File`'s `api_info` carries
`meta.additionalProperties: false`, which gradio_client 1.5.2 cannot walk, and
gradio 5.9.1 computes `api_info` for the index route so `show_api=False` does
not avoid it. Fixed by upgrading to **gradio 5.50.0 / gradio_client 1.14.0**
(`get_api_info()` now succeeds); pinned in `requirements.txt`,
`deploy/requirements-space.txt` and the Space `sdk_version`.

Also fixed: long paths in the model-missing banner pushed the page sideways at
phone width. Notes, cards and tables now wrap or scroll inside themselves.

### UI verified in a browser at 1280px
Hero, degradation banner, all seven tabs, the real dataset tiles
(162,421 / 124,328 / 12,435 / 5) and the batch upload controls all render. The
app boots correctly with no checkpoint and says so, rather than crashing.
### Hugging Face — authenticated
- CLI repaired **inside the project venv**: `huggingface_hub[cli]` 2.x pulls the
  browser-login flow but breaks the pinned `transformers` 4.46.3 and
  `tokenizers` 0.20.3. Pinned **0.34.4** instead — the oldest release that still
  ships the `hf` command and satisfies both (`pip check` clean). The 2.x CLI
  lives in a throwaway venv used only for the browser login; both read the same
  token store at `~/.cache/huggingface/token`.
- Logged in via the OAuth device flow (no token ever pasted into chat, printed,
  logged or committed). `hf auth whoami` → **fariskishtah**.
- Write access confirmed by creating
  **`fariskishtah/fintriage-distilbert` (private)** — the model repo the Space
  will pull weights from.
- The Space itself is created at deploy time, private, once a checkpoint exists.
  Nothing is published without explicit approval; free CPU tier only.

### Application defect found and fixed
Launching the app returned **HTTP 500 on every page load**:
`TypeError: argument of type 'bool' is not iterable` in
`gradio_client/utils.py:887`. Cause: `gr.File`'s `api_info` carries
`meta.additionalProperties: false`, which gradio_client 1.5.2 cannot walk, and
gradio 5.9.1 computes `api_info` for the index route so `show_api=False` does
not avoid it. Fixed by upgrading to **gradio 5.50.0 / gradio_client 1.14.0**
(`get_api_info()` now succeeds); pinned in `requirements.txt`,
`deploy/requirements-space.txt` and the Space `sdk_version`.

Also fixed: long paths in the model-missing banner pushed the page sideways at
phone width. Notes, cards and tables now wrap or scroll inside themselves.

### UI verified in a browser at 1280px
Hero, degradation banner, all seven tabs, the real dataset tiles
(162,421 / 124,328 / 12,435 / 5) and the batch upload controls all render. The
app boots correctly with no checkpoint and says so, rather than crashing.

## NOT STARTED

- [ ] Pull the Kaggle artifacts and run select_model.py, then evaluate.py
- [ ] README with real results
- [ ] HF Space + live verification

## Honest status

No model is trained yet. **Every metric in `docs/ORIGINAL_PROJECT_AUDIT.md` is
the original author's published result, labelled as reference only.** No number
in this repository is ours until `results/metrics/model_comparison.csv` exists.
