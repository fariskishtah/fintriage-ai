# Publication status: BLOCKED — post fully composed, images cannot be attached

**Not published.** No duplicate post exists; the account's activity feed was
checked first and contains no FinTriage AI post.

## What is ready, right now, in the composer

The LinkedIn composer is open on **Faris Kishtah**, "Post to Anyone", holding
the complete post — 2,410 characters — verified in the DOM:

```
intact: true
mentions: ["Instant Software Solutions", "Jana Hatem"]
```

All sixteen content checks passed:

| Check | |
|---|---|
| Opening line | ok |
| Sprint 3 / INSTANT training programme | ok |
| Near-duplicate leakage finding (9,844 groups, 24.7%) | ok |
| SimpleRNN 76.24% / 74.15% | ok |
| LSTM 82.45% / 81.69% | ok |
| GRU 82.20% / 81.54% | ok |
| DistilBERT 85.35% / 84.72% | ok |
| Gradio application features | ok |
| 121 automated tests | ok |
| Acknowledgment with both mentions | ok |
| Closing line | ok |
| Hashtags | ok |
| Live demo URL | ok |
| GitHub URL | ok |
| No stray "@" text left behind | ok |
| No Markdown artifacts | ok |

**Both mentions are real LinkedIn entities**, not plain text — confirmed as
`data-type="mention"` nodes in the editor DOM, selected from the autocomplete:

- **Instant Software Solutions** — chosen from the company results for `@INSTANT`
- **Jana Hatem** — chosen from `@Jana`, the entry headlined "AI & Advanced AI &
  Data Science & Data Analysis Instructor | Training Program Manager", which is
  what distinguishes her from the others sharing the name

## The blocker: images cannot be attached from here

The five screenshots could not be added. Every available route is closed:

1. **The built-in browser has no file-upload tool.** Its toolset covers clicks,
   typing, JS and navigation, but nothing that can set an `<input type="file">`.
2. **In-page fetch is blocked by LinkedIn's CSP.** Serving the images from
   `http://127.0.0.1:8799` with permissive CORS and building `File` objects in
   the page failed with `TypeError: Failed to fetch`; an `<img>` load from the
   same origin also failed. Fetching them over HTTPS from the now-public GitHub
   repository failed the same way — LinkedIn restricts outbound connections.
3. **The native file picker cannot be driven.** LinkedIn's media editor offers
   "Upload from computer", which opens a macOS dialog belonging to the Claude
   desktop app. That app is not in the list of applications the desktop-control
   tool can be granted, so the dialog is unreachable.
4. **Embedding the images as data URIs** would mean passing ~1.6 MB of base64
   through a tool call. Not viable.

A post without the five screenshots is incomplete against
`publication_checklist.md`, so it was not published.

## What remains — two actions, about thirty seconds

The composer is open and ready. In the LinkedIn window:

1. Click the **image icon** in the toolbar, then **Upload from computer**.
2. Select all five files, in filename order, from:
   `/Users/fariskishtah/project/fintriage-ai/linkedin_launch/screenshots/`
   - `01_fintriage_overview.png`
   - `02_ai_classification.png`
   - `03_model_comparison.png`
   - `04_explainable_ai.png`
   - `05_batch_analytics.png`
3. Click **Post**.

Do not retype anything. The text and both mentions are already in place, and
retyping risks losing the tags.

### One note on the link preview

LinkedIn auto-generated a preview card for the Hugging Face link, which hides
the media buttons. It was dismissed with its "x" so the image icon is visible.
If it reappears after adding images, dismiss it again — a post can carry images
or a link card, not both. The URLs stay in the text either way.

### One structural change from the saved draft

In `linkedin_post.txt` the two links sit above the acknowledgment. In the
composer they are at the end, after the hashtags. This was deliberate: typing a
URL mid-post makes LinkedIn insert a preview card, and that re-render moved the
caret and corrupted the text three separate times, once dropping `@INSTANT`
inside the GitHub URL. Typing the URLs last avoids it entirely. The content is
otherwise identical.
