# Publication status: PUBLISHED and verified

**Live post:**
https://www.linkedin.com/feed/update/urn:li:ugcPost:7523425146909483008/

Published on the account of Faris Kishtah. One post, no duplicates — the
account's activity feed was checked beforehand and held no FinTriage AI post.

## Verified on the live post, not on the draft

Every check below was read off the published page.

### Mentions — both resolve to the exact profiles supplied

| Mention | Resolves to |
|---|---|
| Instant Software Solutions | `https://www.linkedin.com/company/instantsoftwaresolution/` |
| Jana Hatem | `https://www.linkedin.com/in/janahatem/` |

These were read from the live post's anchor hrefs, so they are real, clickable
LinkedIn tags, not styled text.

### Images — five attached

Five full-size images are on the post, in filename order:

1. `01_fintriage_overview.png` — application overview
2. `02_ai_classification.png` — live classification
3. `03_model_comparison.png` — four-model comparison
4. `04_explainable_ai.png` — Integrated Gradients
5. `05_batch_analytics.png` — batch processing and analytics

### Content

| Element | Present |
|---|---|
| Opening line | yes |
| Sprint 3 / INSTANT training programme | yes |
| Near-duplicate leakage finding (9,844 groups, 24.7%) | yes |
| SimpleRNN — 76.24% \| 74.15% | yes |
| LSTM — 82.45% \| 81.69% | yes |
| GRU — 82.20% \| 81.54% | yes |
| DistilBERT — 85.35% \| 84.72% | yes |
| Gradio application features | yes |
| 121 automated tests | yes |
| Acknowledgment | yes |
| Closing line | yes |
| Hashtags | yes |

### Links

LinkedIn rewrote both URLs to its own `lnkd.in` shortener, which is its normal
behaviour for outbound links. Both were followed to confirm the destinations:

| In the post | Destination |
|---|---|
| `lnkd.in/dqyHaxsA` | `https://huggingface.co/spaces/fariskishtah/fintriage-ai` |
| `lnkd.in/dtfFPsMf` | `https://github.com/fariskishtah/fintriage-ai` |

## How the images got attached

Not by automation. Every programmatic route was closed: the built-in browser has
no file-upload tool, LinkedIn's CSP blocked building `File` objects in-page from
both a local CORS server and the public GitHub raw URLs, and the native file
picker belongs to an application the desktop-control tool cannot be granted.
The five files were selected by hand in LinkedIn's own uploader, and the post
published from there.

The post text and both mentions were composed programmatically and were intact
in the composer at that point — 2,410 characters, both mentions verified as
`data-type="mention"` nodes before publishing.

## One structural difference from linkedin_post.txt

In the saved draft the two project links sit above the acknowledgment. In the
published post they are at the end, after the hashtags.

This was deliberate. Typing a URL mid-post makes LinkedIn insert a link preview
card, and that re-render moved the editor caret three separate times, once
dropping `@INSTANT` inside the GitHub URL and once truncating a sentence.
Typing the URLs last removed the problem. The wording is otherwise identical.
