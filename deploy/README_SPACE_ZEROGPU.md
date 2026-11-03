---
title: FinTriage AI
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
short_description: Financial complaint classification and intelligent routing
---

# FinTriage AI

**Intelligent Financial Complaint Classification & Routing**

Classifies English-language consumer financial complaints into five product
families, with a confidence-based recommendation for when a human should review
instead of routing automatically.

Production model: **DistilBERT**, fine-tuned on 99,458 complaint narratives.
Held-out test (12,435 complaints, read once): **accuracy 0.8535, macro F1 0.8472**.

| Model | Accuracy | Macro F1 | CPU p50 | Size |
|---|---|---|---|---|
| SimpleRNN | 0.7624 | 0.7415 | 1 ms | 11 MB |
| LSTM | 0.8245 | 0.8169 | 2 ms | 11 MB |
| GRU | 0.8220 | 0.8154 | 2 ms | 11 MB |
| **DistilBERT** | **0.8535** | **0.8472** | 45 ms | 269 MB |

## What you can do here

- **Classify** one complaint: category, top three candidates, probabilities and
  a routing recommendation you can override.
- **Batch** classify a CSV and download the predictions.
- **Analytics** over whatever you classified in this session.
- **Explainability** via Layer Integrated Gradients over the deployed model.
- **Model performance**: all four architectures on the same held-out test set.

## A note on the hardware

This Space runs on ZeroGPU. Inference is executed on CPU, where a single
complaint takes about 45 ms — fast enough that the GPU buys nothing for this
workload, and the CPU path is the one that was measured and tested. The GPU is
requested only for a short warm-up so the runtime is exercised.

## Data

[CFPB consumer complaint dataset](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp),
public-domain US government data; narratives are scrubbed of personal
information by the CFPB before publication. 162,421 rows → 124,328 after
removing nulls, exact duplicates and complaints under three tokens.
Near-duplicate narratives are grouped with MinHash + LSH and kept inside a
single split, so a complaint cannot appear in both training and test in
near-identical form.

Code and full metrics: [github.com/fariskishtah/fintriage-ai](https://github.com/fariskishtah/fintriage-ai)

## Privacy

Nothing you paste or upload is stored or logged. Batch files are processed in
memory for the request and handed back as a download.

## Please note

This tool provides automated complaint categorization for informational and
workflow-support purposes. Predictions may be incorrect and should be reviewed
before consequential decisions. It is not a financial, legal or credit advisor.
Confidence scores are raw softmax outputs, not calibrated probabilities.
