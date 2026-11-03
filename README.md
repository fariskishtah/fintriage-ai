# FinTriage AI

**Intelligent Financial Complaint Classification & Routing**

Classifies English-language consumer financial complaints into five product
families and routes them, with a confidence-based recommendation for when a
human should look instead.

> ⚠️ **Results pending.** This README is updated with real measured metrics once
> all four models finish training. Nothing in this repository reports a metric we
> have not produced ourselves. Current state: `PROJECT_STATUS.md`.

## Categories

`credit_card` · `credit_reporting` · `debt_collection` · `mortgages_and_loans` · `retail_banking`

## Data

[Kaggle: shashwatwork/consume-complaints-dataset-fo-nlp](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp),
file `complaints_processed.csv` — 162,421 rows, derived from the US Consumer
Financial Protection Bureau complaint database (public domain; narratives are
scrubbed of personal information by the CFPB before publication).

Raw narratives are **not** committed here. `python -m src.data.download` fetches
them and records a SHA-256 provenance file.

### Preprocessing

| Step | Rows |
|---|---|
| As published | 162,421 |
| Drop nulls | −10 |
| Drop exact duplicate narratives | −37,939 |
| Drop complaints under 3 tokens | −144 |
| **Used** | **124,328** |

850 very long narratives are truncated at 600 tokens rather than dropped.

**Leakage control.** 9,844 near-duplicate groups covering 30,752 rows (24.7% of
the corpus) are near-identical without being exact duplicates. They are grouped
with MinHash + LSH over character 5-gram shingles and every member of a group is
kept inside a single split, so a complaint cannot appear in both training and
test in near-identical form.

**Splits.** Stratified 80/10/10, seed 42, group-aware →
99,458 / 12,435 / 12,435. Class shares agree across splits within 0.1 pp.

The exact membership is frozen in `data/processed/split_ids.csv.gz` and
fingerprinted in `results/metrics/split_fingerprint.json`, so every model —
local or on a GPU elsewhere — is trained and evaluated on provably identical
examples.

## Models

| | Architecture |
|---|---|
| SimpleRNN | Embedding(20k, 128) → RNN(64) → Dropout → Dense(64) → softmax |
| LSTM | same, LSTM cell |
| GRU | same, GRU cell |
| DistilBERT | `distilbert-base-uncased` fine-tuned, max_len 256, 2 epochs, lr 3e-5 |

All four in PyTorch, class-weighted loss (weights from the training split only),
early stopping on validation macro F1. The recurrent models mask padding by
packing sequences. Hyperparameters live in `configs/config.yaml`.

<!-- RESULTS:START -->

_Results are written here by `scripts/update_readme.py` once the models have been evaluated._

<!-- RESULTS:END -->

## Reproduce

```bash
python -m venv .venv && .venv/bin/pip install -r requirements-dev.txt
```

```bash
.venv/bin/python -m src.data.download
```

```bash
.venv/bin/python scripts/materialize_splits.py
```

```bash
for c in SimpleRNN LSTM GRU; do .venv/bin/python scripts/train_rnn.py --cell $c; done
```

DistilBERT needs a GPU — on an Apple M2 it benchmarks at 49.6 s/step, about 86
hours for the full run. Use `notebooks/kaggle_train_distilbert.ipynb` on a Kaggle
T4 instead.

```bash
.venv/bin/python scripts/select_model.py && .venv/bin/python scripts/evaluate.py
```

```bash
.venv/bin/python app.py
```

## Tests

```bash
.venv/bin/python -m pytest -q
```

## Attribution

The problem framing, the five-category label set and the four-model comparison
were inspired by
[Razan3ssam/Consumer-Complaint-Classifier](https://github.com/Razan3ssam/Consumer-Complaint-Classifier)
(found via the author's LinkedIn post). That project's evaluation protocol —
dataset, 80/10/10 stratified split at seed 42, and the model hyperparameters — is
deliberately reused so the numbers are comparable. An audit of what was reused,
what was redesigned and why is in `docs/ORIGINAL_PROJECT_AUDIT.md`.

The original repository carries no license, so no code was copied from it. This
implementation is independent work. Its published metrics are reproduced in the
audit document as **reference only** and are never presented as ours.

## Limitations

- English only, five product families, one label per complaint.
- Confidence scores are raw softmax outputs, not calibrated probabilities.
- Token attributions show local sensitivity, not the model's reasoning.
- Complaints spanning several products are inherently ambiguous.

## Licence

MIT, for this implementation. The dataset keeps its own terms.
