# FinTriage AI — LinkedIn post (final draft)

**Status: NOT PUBLISHED.** Awaiting your review.

340 words. Images in filename order from `screenshots/`:
`01_fintriage_overview.png` → `05_batch_analytics.png`.

`«@Name»` marks the three places where a real LinkedIn tag must be inserted
**in the composer**. The plain-text version in `linkedin_post.txt` has the names
written out, so it can be pasted first and the tags added over them.

---

From raw consumer complaints to a deployed deep learning system.

FinTriage AI reads a written financial complaint and predicts which product family it belongs to — credit card, credit reporting, debt collection, mortgages and loans, or retail banking — so it reaches the right team instead of sitting in a queue. I built it as Sprint 3 of the «@INSTANT Software Solutions» AI & Data Science Training program.

The data work decided the outcome. From 162,421 CFPB complaints I kept 124,328 after removing nulls, exact duplicates and text too short to carry signal. The step that changed my numbers: 9,844 groups covering 24.7% of the corpus were near-duplicates — not identical, but close enough for a model to memorise one and "recognise" the other across a split. I grouped them with MinHash and LSH, kept each group inside one split, and froze the splits to a hashed row-level manifest, so all four models saw provably identical examples.

Four architectures on that data: SimpleRNN, LSTM, GRU, and a fine-tuned distilbert-base-uncased.

Held-out test set of 12,435 complaints, opened once:

SimpleRNN — 76.24% accuracy, 74.15% macro F1
LSTM — 82.45%, 81.69%
GRU — 82.20%, 81.54%
DistilBERT — 85.35%, 84.72%

DistilBERT went to production, selected on validation performance before the test set was touched.

It ships as a Gradio application: real-time classification, top-3 predictions with confidence scores, batch CSV processing, an analytics dashboard, and token-level explanations via Integrated Gradients. Predictions below a validation-calibrated confidence threshold are flagged for human review rather than routed automatically — a triage assistant, not a decision maker.

Built with PyTorch and Hugging Face Transformers, fine-tuned on a Kaggle GPU, deployed on Hugging Face Spaces, with 121 automated tests.

Live demo: https://huggingface.co/spaces/fariskishtah/fintriage-ai
Code: https://github.com/fariskishtah/fintriage-ai

Special thanks to «@INSTANT Software Solutions» and «@Jana Hatem» for the guidance and the opportunity throughout the AI & Data Science Training program. It turned deep learning concepts into a complete project I could build, evaluate honestly, and ship.

Data quality set the ceiling long before architecture did, and shipping turned out to be its own discipline.


---

## Mentions — must be inserted by you

| Where | Who | Verified URL |
|---|---|---|
| Opening paragraph | INSTANT Software Solutions | https://www.linkedin.com/company/instantsoftwaresolution/ |
| Acknowledgment | INSTANT Software Solutions | same as above |
| Acknowledgment | Eng. Jana Hatem | https://www.linkedin.com/in/janahatem |

**No tag could be inserted automatically.** No Chrome browser is connected to
this session (`list_connected_browsers` returned empty), so there is no
authenticated LinkedIn composer to drive — and publishing to your account
without your approval was out of scope regardless.

Worth knowing: **pasting a profile URL into a post does not create a mention.**
LinkedIn only turns text into a tag when you pick the entry from the `@`
autocomplete dropdown. The URLs above are for confirming you picked the right
one, not for pasting.

## Claims and where each comes from

| Claim | Source |
|---|---|
| 162,421 → 124,328 records | `results/metrics/data_quality_report.json` |
| 9,844 groups / 24.7% | `results/metrics/data_quality_report.json` |
| 12,435 test complaints | `results/metrics/model_comparison.json` |
| All four accuracy / macro F1 figures | `results/metrics/model_comparison.json` |
| DistilBERT selected on validation | `results/metrics/model_selection.json` |
| Validation-calibrated threshold | `results/metrics/review_threshold.json` |
| 121 automated tests | `pytest`: 121 passed, 0 skipped |

## Links in the post

Both verified reachable by an unauthenticated request:

- https://huggingface.co/spaces/fariskishtah/fintriage-ai — 200, Space `RUNNING`
- https://github.com/fariskishtah/fintriage-ai — 200

## Deliberately not claimed

- No production use, no financial decision-making, no advice.
- No claim that classification is perfect or that confidence is calibrated.
