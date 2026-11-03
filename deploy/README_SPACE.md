---
title: FinTriage AI
emoji: 🏦
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
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

- **Classify** a single complaint: category, top three candidates, probabilities,
  and a routing recommendation you can override.
- **Batch** classify a CSV and download the predictions.
- **Analytics** over whatever you classified in this session.
- **Explainability** via Layer Integrated Gradients over the deployed model.
- **Model performance**: four architectures on the same held-out test set.

## Data and method

Trained on the [CFPB consumer complaint
dataset](https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp)
(public-domain US government data; narratives are scrubbed of personal
information by the CFPB before publication). 162,421 rows → 124,328 after
removing nulls, exact duplicates and complaints under three tokens.

Near-duplicate narratives are grouped with MinHash + LSH and kept inside a
single split, so a complaint cannot appear in both training and test in
near-identical form. Splits are stratified 80/10/10, seed 42, and their exact
membership is frozen and fingerprinted so all four models are compared on
identical examples.

Full code, metrics and the audit of the project that inspired this one:
[github.com/fariskishtah/fintriage-ai](https://github.com/fariskishtah/fintriage-ai)

## Privacy

Nothing you paste or upload is stored or logged. Batch files are processed in
memory for the request and handed back as a download.

## Please note

This tool provides automated complaint categorization for informational and
workflow-support purposes. Predictions may be incorrect and should be reviewed
before consequential decisions. It is not a financial, legal or credit advisor.
Confidence scores are raw softmax outputs, not calibrated probabilities.
