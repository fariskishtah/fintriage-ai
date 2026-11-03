# FinTriage AI — LinkedIn post (draft)

**Status: NOT PUBLISHED.** Awaiting your review and approval.

Images, in order: `screenshots/01_fintriage_overview.png` → `05_batch_analytics.png`.

`[[MENTION: ...]]` marks where a real LinkedIn tag must be inserted **in the
LinkedIn composer** by typing `@` and selecting from autocomplete. Typing the
name as plain text does not create a working tag. See `verified_mentions.md`.

---

From raw consumer complaints to a deployed deep learning system.

I built FinTriage AI as Sprint 3 of the **[[MENTION: INSTANT organisation page]]** AI & Data Science Training Program. It reads a written financial complaint and predicts which product family it belongs to, so it can be routed to the team that actually handles it.

The problem is unglamorous and expensive. Complaints arrive as free text. Someone reads each one and decides whether it concerns a credit card, credit reporting, debt collection, mortgages and loans, or retail banking. Misroute it and the customer waits while it finds its way to the right desk.

The data work mattered more than the modelling.

I started from 162,421 CFPB consumer complaints and kept 124,328 after removing nulls, exact duplicates and complaints too short to carry signal. Then the part that changed my numbers: 9,844 groups covering 30,752 records — 24.7% of the corpus — were near-duplicates. Not identical, but close enough that a model could memorise one and "recognise" the other across a split. I grouped them with MinHash and LSH and kept every group inside a single split. The splits were stratified 80/10/10 and frozen to a row-level manifest with a content hash per split, so all four models were trained and evaluated on provably identical examples.

Four architectures on that same data: SimpleRNN, LSTM, GRU, and a fine-tuned distilbert-base-uncased.

Held-out test set of 12,435 complaints, opened once:

SimpleRNN — 76.24% accuracy, 74.15% macro F1
LSTM — 82.45%, 81.69%
GRU — 82.20%, 81.54%
DistilBERT — 85.35%, 84.72%

DistilBERT went to production, selected on validation performance before the test set was touched.

The project did not stop at a notebook. It ships as a Gradio application with real-time classification, top-3 predictions with confidence scores, batch CSV processing, an analytics dashboard, and token-level explanations via Integrated Gradients. Predictions below a confidence threshold calibrated on validation data are flagged for human review rather than routed automatically — the model is a triage assistant, not a decision maker.

Built with PyTorch and Hugging Face Transformers, fine-tuned on a Kaggle GPU, deployed on Hugging Face Spaces, with 121 automated tests covering preprocessing, leakage detection, inference and batch handling.

Thank you to **[[MENTION: INSTANT organisation page]]** and to my instructor, **[[MENTION: Jana Hatem]]**, whose guidance and structure turned deep learning concepts into something I could actually build and defend.

Three things I am taking with me. Data quality decides your ceiling long before architecture does — the near-duplicate work cost more than the fine-tuning and mattered more. A model is only as trustworthy as the evaluation protocol behind it, which is why the test set was opened exactly once. And shipping is its own discipline: the gap between a trained checkpoint and an application someone can use is wider than it looks.


---

## Length

465 words. Long for LinkedIn but appropriate for a technical portfolio post;
the first three lines carry the hook before the "see more" fold.

## Claims and where each comes from

| Claim in the post | Source file |
|---|---|
| 162,421 → 124,328 records | `results/metrics/data_quality_report.json` |
| 9,844 groups / 30,752 records / 24.7% | `results/metrics/data_quality_report.json` |
| 12,435 test complaints | `results/metrics/model_comparison.json` |
| All four accuracy / macro F1 figures | `results/metrics/model_comparison.json` |
| DistilBERT selected on validation | `results/metrics/model_selection.json` |
| Confidence threshold calibrated on validation | `results/metrics/review_threshold.json` |
| 121 automated tests | `pytest` run, 121 passed / 0 skipped |

Every figure was checked against these files before this draft was written.

## Deliberately not claimed

- No public demo link (all three repositories are private).
- No claim of production use, financial decision-making, or advice.
- No claim that classification is perfect or calibrated.
