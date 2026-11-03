# FinTriage AI — project status

Updated: 2026-11-03

## COMPLETED

### Data
- Kaggle `shashwatwork/consume-complaints-dataset-fo-nlp`, `complaints_processed.csv`,
  **162,421 rows**, SHA-256 `a120ac8450cb3839…`. Public-domain CFPB source.
- Cleaned to **124,328** (nulls 10, exact duplicates 37,939, under 3 tokens 144,
  850 truncated at 600 tokens). Nulls and exact-duplicate counts match the
  reference project exactly.
- **9,844 near-duplicate groups over 30,752 rows (24.7%)** grouped by MinHash/LSH
  and kept inside one split each.
- Splits **99,458 / 12,435 / 12,435**, stratified 80/10/10, seed 42, group-aware.
  Frozen to `data/processed/split_ids.csv.gz` and fingerprinted. Zero narratives
  shared across splits; zero duplicate groups spanning splits.

### Models — all four trained, one Kaggle T4 session
Promoted only after `scripts/verify_artifacts.py` passed every provenance check.

**Held-out test, 12,435 complaints, read exactly once:**

| Model | Accuracy | Macro F1 | Weighted F1 | Train | CPU p50 | Size | Params |
|---|---|---|---|---|---|---|---|
| SimpleRNN | 0.7624 | 0.7415 | 0.7664 | 86 s | 1 ms | 11 MB | 2,576,901 |
| LSTM | 0.8245 | 0.8169 | 0.8267 | 76 s | 2 ms | 11 MB | 2,614,149 |
| GRU | 0.8220 | 0.8154 | 0.8242 | 54 s | 2 ms | 11 MB | 2,601,733 |
| **DistilBERT** | **0.8535** | **0.8472** | **0.8549** | 4,588 s | 45 ms | 269 MB | 66,957,317 |

- Production model **DistilBERT**, chosen on validation macro F1 (0.84376) before
  the test split was read.
- Review threshold **0.68**, calibrated on validation: auto-routes **87.3%** at
  **90.1%** accuracy, ECE **0.0415**.
- Reproducibility (validation only), local M2 MPS vs Kaggle T4:
  SimpleRNN +0.0002, LSTM −0.0047, GRU +0.0005.

### Application — verified against the real model
All seven tabs work. Verified in a browser:
- Live classification: Debt Collection at 98.2%, correct routing badge, top-3,
  calibration caveat.
- Batch CSV with the real model: all five labelled complaints classified into
  their intended categories; empty/short rows skipped with a warning.
- Session analytics: real counts, no fabricated data.
- Explainability: real Integrated Gradients; top tokens `debt, collection,
  agency, calling, owe`.
- Model Performance: real metrics, DistilBERT marked PRODUCTION.
- Mobile at 375 px: `scrollWidth == innerWidth`, no horizontal overflow.

### Tests
**121 passing, 0 skipped, 0 failing.**

### Bugs found by testing, and fixed
1. **HTTP 500 on every page load** — `gr.File`'s api_info has
   `meta.additionalProperties: false`, which gradio_client 1.5.2 cannot walk.
   Fixed by upgrading to gradio 5.50.0 / gradio_client 1.14.0.
2. **Empty CSV cells classified as the string "nan"** — `.astype(str)` turned
   NaN into `"nan"`, three characters, so it passed the length check. Fixed,
   two regression tests added.
3. **Space bundle would not have booted** — the deploy script prepended an
   import above `from __future__ import annotations`, a SyntaxError. Fixed; the
   staged app.py is now AST-parsed before upload.
4. **Kaggle artifact pull would have overwritten the local runs** — `--pull` did
   `extractall(ROOT)`, which would have mixed M2 and T4 timings in one table.
   Now stages to `artifacts_kaggle/` behind an explicit verification gate.

### Repositories
- GitHub **`fariskishtah/fintriage-ai` — PRIVATE**, synced. No weights, no raw
  narratives, no credentials tracked.
- Hugging Face model repo **`fariskishtah/fintriage-distilbert` — PRIVATE**,
  268.8 MB uploaded (safetensors + tokenizer + config).

## BLOCKED — one decision needed

**The Hugging Face Space cannot be created privately on the free tier.**
`create_repo` returned HTTP 402:

> Static Spaces are free for everyone, but hosting Gradio and Docker Spaces on
> free cpu-basic requires a PRO subscription.

No Space was created; nothing is public.

The deployment bundle itself is **verified working**: staged, run with no local
`artifacts/`, it pulled the weights from the private Hub repo, loaded the model,
predicted `Debt Collection` at 0.9801, and built the Gradio Blocks with a valid
api_info. The only missing piece is somewhere to host it.

Options, all requiring your decision:
1. Make the Space **public** (free). Needs explicit approval.
2. **HF PRO** (~$9/month) to keep it private. A paid charge; needs approval.
3. Leave it unhosted. The app runs locally and is fully verified.

## NOT DONE
- Live public URL verification (blocked by the above).
- README screenshots (the figures are in; app screenshots pending a decision).
