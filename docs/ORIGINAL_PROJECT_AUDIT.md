# Audit of the referenced project

## Identity verification

| Step | Result |
|---|---|
| `https://lnkd.in/p/d7cpemz3` | 301 → `linkedin.com/posts/razan-essam-1a5a0331a_...ugcPost-7522736399595417602-bsPY` |
| `https://lnkd.in/dphMqUew` | LinkedIn interstitial whose outbound button targets `https://github.com/Razan3ssam/Consumer-Complaint-Classifier` |
| Owner match | LinkedIn handle `razan-essam-1a5a0331a` ↔ GitHub login `Razan3ssam` (display name `𝓡𝓪𝔃𝓪𝓷`) |

**Verified original repository: <https://github.com/Razan3ssam/Consumer-Complaint-Classifier>**
(created 2026-11-01, pushed 2026-11-01, 0 stars, no license file).

This matters: a GitHub search for the project description returns at least ten
near-identically named repositories from what appears to be the same cohort
(`Marinastudent-hub`, `Medhat-66`, `uvez101`, `rAmeZ-m`, `basmalakhaled20`,
`mohamedalaaabdella-maker`, `ahmedtame-r`, `mohammedmokhtar77`, …). None of them
is the repository the post links to. We resolved the identity through the
author's own post rather than by name matching.

## What the repository contains

```
README.md                      4.5 KB
app.py                        14.2 KB   Gradio app, single-complaint only
requirements.txt                36 B    gradio / transformers / torch / nltk, unpinned
notebook/ccd-fornlp.ipynb      464 KB   the whole pipeline, Kaggle notebook export
results/model_comparison.csv            4 rows of headline metrics
results/results.json                    the same 4 rows
assets/*.mp4                  19.6 MB   screen recording of the app
Releases v1.0                  268 MB   fine-tuned DistilBERT (config/safetensors/tokenizer)
```

## Techniques and assumptions, read off the notebook

| Item | Value in the original |
|---|---|
| Dataset | Kaggle `shashwatwork/consume-complaints-dataset-fo-nlp`, file `complaints_processed.csv` |
| Raw rows | 162,421 |
| Cleaning | `dropna()` (10 rows), `drop_duplicates(subset='narrative')` (37,939 rows) → 124,472 |
| Labels | 5 classes, `LabelEncoder` (alphabetical order) |
| Split | `train_test_split` 80/10/10, `stratify=label`, `random_state=42` |
| Imbalance | `compute_class_weight('balanced')` on the training split |
| RNN input | Keras `Tokenizer(num_words=20000, oov_token="<OOV>")`, `maxlen=200`, post pad/truncate |
| RNN architecture | `Embedding(20000, 128, mask_zero=True)` → recurrent(64) → Dropout → Dense(64) → softmax |
| RNN training | Adam, batch 128, up to 8 epochs, early stopping, class weights |
| Transformer | `distilbert-base-uncased`, HF `Trainer`, max_len 256, 2 epochs, lr 3e-5, batch 32/64, weighted loss |
| Evaluation | accuracy, macro precision/recall/F1, confusion matrices, on the 10% test split |

## Published results (reference only — not ours)

| Model | Accuracy | P (macro) | R (macro) | F1 (macro) | Train time |
|---|---|---|---|---|---|
| SimpleRNN | 0.7953 | 0.7674 | 0.7841 | 0.7731 | 70 s |
| LSTM | 0.8206 | 0.7931 | 0.8398 | 0.8122 | 74 s |
| GRU | 0.8373 | 0.8067 | 0.8508 | 0.8257 | 47 s |
| DistilBERT | 0.8527 | 0.8287 | 0.8609 | 0.8433 | 2397 s |

## What works

- The problem framing and the four-way comparison are sound, and the protocol is
  stated precisely enough to reproduce.
- Exact duplicates are dropped **before** splitting, which rules out the most
  obvious leakage path.
- Class weights are computed on the training split only.
- The fine-tuned checkpoint is genuinely published, so the app is runnable.

## What is missing or weak

| Gap | Consequence |
|---|---|
| No license file | Code is "all rights reserved" by default. We reuse the *protocol*, write our own code, and attribute. |
| Only exact duplicates removed | The source has many near-duplicate narratives; they can straddle the split and inflate the test score. |
| No fixed label order on disk | `LabelEncoder` order is implicit. A column reordering silently breaks inference. |
| Metrics rounded to 4 dp in one file | No per-class table, no confusion matrix values, no latency, no model size, no parameter counts. |
| No seeds inside the training loops | The reported numbers are not reproducible run to run. |
| `requirements.txt` unpinned | A future `transformers` release can break checkpoint loading. |
| App is single-complaint only | No batch mode, no analytics, no explanation, no calibrated threshold. |
| Threshold 60% hard-coded | Not derived from validation data; softmax treated as a calibrated probability. |
| No tests | Nothing guards the label mapping or the preprocessing. |

## Independently verifiable vs. not

- **Verifiable:** every metric in the table, because the dataset, the split seed
  and the protocol are all stated. We re-run them ourselves.
- **Not verifiable as published:** the training times (unstated hardware — a
  Kaggle accelerator) and the exact number of epochs the early stopping chose.

## Reused conceptually / redesigned

- **Reused:** the five-class task, the dataset and its label set, the 80/10/10
  stratified protocol with seed 42, the RNN hyperparameters, and the DistilBERT
  fine-tuning recipe — so our numbers sit beside theirs honestly.
- **Redesigned:** PyTorch for all four models instead of Keras + PyTorch (one
  framework, one install, and the recurrent models mask padding by packing);
  near-duplicate grouping before splitting; a fixed label order in `configs/`;
  a validation-derived review threshold; a full metrics suite; tests; and the
  application itself.
