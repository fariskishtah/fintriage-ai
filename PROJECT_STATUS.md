# FinTriage AI — project status

Updated: 2026-11-03

## Status: DEPLOYED AND VERIFIED LIVE

Private Space: https://huggingface.co/spaces/fariskishtah/fintriage-ai — `RUNNING`
GitHub: https://github.com/fariskishtah/fintriage-ai — private
Model repo: https://huggingface.co/fariskishtah/fintriage-distilbert — private

## Final results — held-out test, 12,435 complaints, read exactly once

| Model | Accuracy | Macro F1 | Weighted F1 | Train | CPU p50 | Size | Params |
|---|---|---|---|---|---|---|---|
| SimpleRNN | 0.7624 | 0.7415 | 0.7664 | 86 s | 1 ms | 11 MB | 2,576,901 |
| LSTM | 0.8245 | 0.8169 | 0.8267 | 76 s | 2 ms | 11 MB | 2,614,149 |
| GRU | 0.8220 | 0.8154 | 0.8242 | 54 s | 2 ms | 11 MB | 2,601,733 |
| **DistilBERT** | **0.8535** | **0.8472** | **0.8549** | 4,588 s | 45 ms | 269 MB | 66,957,317 |

Production model **DistilBERT**, selected on validation macro F1 (0.84376)
before the test split was read. Review threshold **0.68**, calibrated on
validation: auto-routes 87.3% at 90.1% accuracy, ECE 0.0415.

All four trained and evaluated in one Kaggle T4 session and promoted only after
`scripts/verify_artifacts.py` passed: identical split hashes, identical label
map, all checkpoints loadable, selection free of any test metric, test split
read once. Local M2 MPS runs kept in `artifacts_local_mps/` as a validation-only
reproducibility record: SimpleRNN +0.0002, LSTM −0.0047, GRU +0.0005.

## Tests

**121 passing, 0 skipped, 0 failing.**

## Live verification

Performed against the running Space through its own endpoints, not against the
uploaded files.

- All five categories predicted correctly live.
- Live vs local confidence: max difference **0.048 pp** over six examples
  (live torch 2.8.0 vs local 2.5.1).
- Top-3, confidence, routing badge: correct. The ambiguous example returns
  53.3% and is routed to *Manual Review Recommended*.
- Batch CSV: 7 rows uploaded → 5 classified, 2 skipped, results file returned.
- Analytics: real session counts, 5 of 5 categories, overrides reflected.
- Explainability: Integrated Gradients, 19 token attributions.
- Validation: empty and too-short inputs rejected with clear messages.
- Restart: model reloads, inference unchanged, all 6 model files persist.
- Served CSS carries the brand tokens, the 640 px breakpoint and the overflow fix.

## Bugs found by testing, and fixed

1. **HTTP 500 on every page load** — `gr.File`'s api_info has
   `meta.additionalProperties: false`, which gradio_client 1.5.2 cannot walk.
   Fixed by upgrading to gradio 5.50.0 / gradio_client 1.14.0.
2. **Empty CSV cells classified as the string "nan"** — `.astype(str)` turned
   NaN into `"nan"`, three characters, so it passed the length check. Fixed,
   two regression tests.
3. **Space bundle would not boot** — the deploy script prepended an import above
   `from __future__ import annotations`. Fixed; staged app.py is AST-parsed
   before upload.
4. **Artifact pull would have overwritten the local runs** — `--pull` did
   `extractall(ROOT)`, mixing M2 and T4 timings in one table. Now staged behind
   a verification gate.
5. **ZeroGPU rejected torch 2.5.1** — `CONFIG_ERROR`. ZeroGPU bundle moved to
   2.8.0; the project and the cpu-basic bundle stay on 2.5.1.
6. **GPU warm-up wasted quota** — checked the `spaces` package: it enforces
   nothing at import, so a ZeroGPU Space serves fine with no `@spaces.GPU` call.
   The warm-up was reserving an A10G slice on every boot for CPU-only work.
   Removed.
7. **Broken upload retry** — the retry loop piped to `tail`, so it read `tail`'s
   exit code and gave up after one failure. Fixed; upload split into two stages.

## Known limitations

- The README has the seven generated result figures but no application
  screenshots.
- Live **pixel** QA was not possible: the Space is private and the automated
  browser has no Hugging Face session. Functional behaviour and the served
  CSS were verified live; the rendering of the identical bundle was verified
  locally at 1280 px and 375 px.
- On the first request after a cold start one reading came back 98.0% where
  steady state is 98.2%. Category unchanged and far from the 0.68 threshold.
- English only, five product families, one label per complaint.
- Confidence is raw softmax, not calibrated (ECE 0.0415).
- Free ZeroGPU accounts are limited to 2 Spaces; both slots are now used.
