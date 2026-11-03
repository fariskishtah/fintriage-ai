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
| 10 | INSTANT mention | **Blocked** | Organisation identified as INSTANT Software Solutions; LinkedIn page URL not verified. Insert via composer autocomplete |
| 11 | Jana Hatem mention | **Blocked** | Role confirmed; profile URL not verified and deliberately not guessed. Insert via composer autocomplete |
| 12 | GitHub visibility | **Private** | `github.com/fariskishtah/fintriage-ai`. Public readers cannot open it |
| 13 | Hugging Face Space visibility | **Private** | `huggingface.co/spaces/fariskishtah/fintriage-ai`. Public readers cannot open it |
| 14 | Model repo visibility | **Private** | `huggingface.co/fariskishtah/fintriage-distilbert` |
| 15 | Link accessibility | **Decision needed** | See below |
| 16 | Final approval | **Pending** | Nothing is published until you say so |

## About the links

The current draft deliberately contains **no project links**, because all three
repositories are private and a public audience would hit a 404. It stands on the
screenshots and the results.

Two options:

**A — keep everything private.** The draft is ready as written. Add a line such
as "Happy to walk through the code and the evaluation protocol on request."

**B — make the repositories public.** Then add the links. This needs your
explicit approval and some care:

- Review the GitHub repository before it goes public. It carries no weights, no
  raw narratives and no credentials, but it is still your call.
- Free accounts are limited to 2 ZeroGPU Spaces, and both are in use.
- A public Space exposes the app and its bundled checkpoint to anyone with the
  link.

**No visibility has been changed, and none will be without your explicit
instruction.**

## Publication order

1. Approve the post text.
2. Decide on visibility (option A or B above).
3. Open the LinkedIn composer and upload the five images **in filename order** —
   they are sequenced as a story: product → prediction → evidence →
   explainability → scale.
4. Insert the two mentions via autocomplete and confirm each resolves.
5. Publish.
