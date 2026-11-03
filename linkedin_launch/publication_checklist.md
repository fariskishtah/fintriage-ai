# Publication checklist — FinTriage AI

**The post is NOT published.** It needs your approval first.

## Ready

| # | Item | Status | Detail |
|---|---|---|---|
| 1 | Screenshot quality | **Ready** | 5 × 1080×1350 PNG, captured at 2× device scale; text sharp, charts readable |
| 2 | Screenshots are authentic | **Ready** | Captured by driving the running application with the real fine-tuned DistilBERT. No prediction, probability or metric was edited. Originals kept in `linkedin_assets/raw/` |
| 3 | Results correct | **Ready** | Every figure checked against `results/metrics/`; all four models match to 2 dp |
| 4 | Production model correct | **Ready** | DistilBERT, matching `model_selection.json` |
| 5 | Project name | **Ready** | "FinTriage AI" throughout |
| 6 | Privacy audit | **Ready** | Screenshots use synthetic complaints written for this project. No real narrative, token, password or account detail is visible |
| 7 | No overclaiming | **Ready** | No public demo promised, no production use claimed, explicitly described as triage support rather than decision-making |
| 8 | Spelling and grammar | **Ready** | Reviewed |
| 9 | Tone | **Ready** | No emoji, no hype phrasing, no engagement bait |

## Needs your decision

| # | Item | Status | What is needed |
|---|---|---|---|
| 10 | INSTANT mention | **URL confirmed, tag pending** | https://www.linkedin.com/company/instantsoftwaresolution/ — insert via composer autocomplete (2 places) |
| 11 | Jana Hatem mention | **URL confirmed, tag pending** | https://www.linkedin.com/in/janahatem — insert via composer autocomplete (1 place) |
| 12 | GitHub visibility | **Public** | Made public on your instruction. Full history audited first: no secrets, no credentials, no raw narratives, no weights in any commit |
| 13 | Hugging Face Space visibility | **Public** | `RUNNING` on ZeroGPU |
| 14 | Model repo visibility | **Public** | 268.8 MB checkpoint |
| 15 | Link accessibility | **Verified** | All four URLs return 200 to an anonymous request |
| 16 | Live check on the public URL | **Passed** | Real browser on the public Space: classification returned Mortgages and Loans at 99.1%, matching local. Mobile at 375 px: no horizontal overflow |
| 17 | Post length | **Ready** | 340 words, inside the 250–350 target |
| 18 | Final approval | **Pending** | Nothing is published to LinkedIn until you say so |

## Links now in the post

```
Live demo: https://huggingface.co/spaces/fariskishtah/fintriage-ai
Code: https://github.com/fariskishtah/fintriage-ai
Model: https://huggingface.co/fariskishtah/fintriage-distilbert
```

Each was checked with an unauthenticated request, which is what a reader gets.

## What going public exposed

Audited before flipping, across the **entire git history**, not just the current
files:

- No tokens, keys or credentials in any commit.
- No `kaggle.json`, `.env`, `.pem` or similar ever committed.
- No raw complaint narratives and no model weights in the repository.
- The one data file published, `data/processed/split_ids.csv.gz`, holds only row
  indices into the public Kaggle file plus label ids and split names. No text.

The Space carries the 268 MB checkpoint, which is now downloadable by anyone, as
is the model repository. That is the normal consequence of publishing a model
and was the intent of making it public.

## Publication order

1. Approve the post text.
2. Open the LinkedIn composer and upload the five images **in filename order** —
   they are sequenced as a story: product → prediction → evidence →
   explainability → scale.
3. Insert the two mentions via autocomplete and confirm each resolves.
4. Publish.
