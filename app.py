"""FinTriage AI - Gradio application.

Run locally:   python app.py
Deployed on Hugging Face Spaces from this same file.

The app never invents numbers. Every metric shown in the Model Performance tab
is read from results/metrics/, and the dashboard reflects only complaints the
current session actually classified.
"""
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd

from src.app_theme import CSS, DISCLAIMER
from src.charts import (
    category_distribution,
    category_share_donut,
    confidence_histogram,
    confusion_heatmap,
    empty_figure,
    latency_vs_size,
    metric_comparison,
    probability_bars,
    routing_split,
    training_curves,
)
from src.config import display_name, load_config
from src.inference.batch import BatchError, classify_frame, read_csv_safely
from src.inference.examples import EXAMPLES, example_texts
from src.inference.predictor import DistilBertPredictor, resolve_model_dir
from src.ui_blocks import (
    batch_summary_card,
    comparison_table,
    disclaimer,
    empty_state,
    esc,
    hero,
    note,
    tiles,
    verdict_card,
)

CFG = load_config()
LABELS = list(CFG["data"]["labels"])
DISPLAY = [display_name(label, CFG) for label in LABELS]
METRICS_DIR = CFG.path("results", "metrics")
MIN_CHARS = 25


# --------------------------------------------------------------------------- #
# model + metrics loading
# --------------------------------------------------------------------------- #

def load_predictor() -> tuple[DistilBertPredictor | None, str | None]:
    try:
        return DistilBertPredictor(resolve_model_dir(CFG), cfg=CFG, device="cpu"), None
    except Exception as exc:  # noqa: BLE001
        return None, str(exc)


PREDICTOR, MODEL_ERROR = load_predictor()
EXPLAINER = None  # built lazily: captum costs memory we only pay if used


def get_explainer():
    global EXPLAINER
    if EXPLAINER is None and PREDICTOR is not None:
        from src.explainability.integrated_gradients import IntegratedGradientsExplainer

        EXPLAINER = IntegratedGradientsExplainer(
            PREDICTOR, steps=16, max_chars=int(CFG["app"]["explain_max_chars"])
        )
    return EXPLAINER


def read_json(name: str) -> dict | None:
    path = METRICS_DIR / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


COMPARISON = read_json("model_comparison.json")
SELECTION = read_json("model_selection.json")
THRESHOLD_INFO = read_json("review_threshold.json")
QUALITY = read_json("data_quality_report.json")
SPLITS = read_json("split_manifest.json")
PROVENANCE = read_json("dataset_provenance.json")

THRESHOLD = PREDICTOR.threshold if PREDICTOR else float(CFG["inference"]["review_threshold"])
CALIBRATED = False  # raw softmax; select_model.py reports the ECE


def comparison_rows() -> list[dict]:
    if not COMPARISON:
        return []
    rows = []
    for name, payload in COMPARISON["models"].items():
        rows.append(
            {
                "model": name,
                "accuracy": payload["accuracy"],
                "f1_macro": payload["f1_macro"],
                "f1_weighted": payload["f1_weighted"],
                "precision_macro": payload["precision_macro"],
                "recall_macro": payload["recall_macro"],
                "train_time_sec": payload.get("train_time_sec"),
                "cpu_latency_p50_ms": payload["cpu_latency"]["p50_ms"],
                "artifact_size_mb": payload["artifact_size_mb"],
                "trainable_parameters": payload.get("trainable_parameters"),
                "confusion_matrix": payload["confusion_matrix"],
                "per_class": payload["per_class"],
            }
        )
    return rows


ROWS = comparison_rows()
WINNER = SELECTION["selected_model"] if SELECTION else None


# --------------------------------------------------------------------------- #
# session state (per browser session; nothing is persisted to disk)
# --------------------------------------------------------------------------- #

@dataclass
class SessionState:
    categories: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def add(self, categories: list[str], confidences: list[float], source: str) -> None:
        self.categories.extend(categories)
        self.confidences.extend(confidences)
        self.sources.extend([source] * len(categories))

    @property
    def total(self) -> int:
        return len(self.categories)

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"category": self.categories, "confidence": self.confidences, "source": self.sources}
        )


# --------------------------------------------------------------------------- #
# Tab 2 - single complaint
# --------------------------------------------------------------------------- #

def classify(text: str, state: SessionState):
    blank_outputs = (
        gr.update(visible=False),
        empty_figure("Submit a complaint to see the probabilities."),
        gr.update(visible=False),
        state,
    )
    if PREDICTOR is None:
        return (
            gr.update(
                value=note(
                    "The model is not loaded, so no prediction can be made. "
                    f"<br><code>{esc(MODEL_ERROR)}</code>",
                    warn=True,
                ),
                visible=True,
            ),
            empty_figure("Model unavailable."),
            gr.update(visible=False),
            state,
        )

    text = (text or "").strip()
    if not text:
        return (
            gr.update(value=note("Please paste a complaint before classifying.", warn=True), visible=True),
            *blank_outputs[1:],
        )
    if len(text) < MIN_CHARS:
        return (
            gr.update(
                value=note(
                    f"That is only {len(text)} characters. Please provide at least "
                    f"{MIN_CHARS} so there is something to classify.",
                    warn=True,
                ),
                visible=True,
            ),
            *blank_outputs[1:],
        )

    prediction = PREDICTOR.predict(text)
    state.add([prediction.display], [prediction.confidence], "single")
    return (
        gr.update(value=verdict_card(prediction, CALIBRATED), visible=True),
        probability_bars(prediction.probabilities),
        gr.update(visible=True, value=prediction.display),
        state,
    )


def apply_override(chosen: str, state: SessionState):
    """A human correction replaces the model's answer in the session analytics."""
    if not chosen or not state.categories:
        return gr.update(), state
    state.categories[-1] = chosen
    state.sources[-1] = "human override"
    return (
        gr.update(
            value=note(
                f"Recorded <strong>{esc(chosen)}</strong> as the reviewed category. "
                "The dashboard now counts the corrected label.",
            ),
            visible=True,
        ),
        state,
    )


# --------------------------------------------------------------------------- #
# Tab 3 - batch CSV
# --------------------------------------------------------------------------- #

def process_batch(file_obj, column_choice: str, state: SessionState):
    def fail(message: str):
        return (
            gr.update(value=note(esc(message), warn=True), visible=True),
            gr.update(value=None, visible=False),
            gr.update(value=None, visible=False),
            state,
        )

    if PREDICTOR is None:
        return fail(f"The model is not loaded: {MODEL_ERROR}")
    if file_obj is None:
        return fail("Please choose a CSV file first.")

    try:
        frame, warnings = read_csv_safely(
            file_obj.name,
            max_mb=float(CFG["app"]["max_upload_mb"]),
            max_rows=int(CFG["app"]["max_rows"]),
        )
        result = classify_frame(
            frame,
            PREDICTOR,
            text_column=(column_choice or None) if column_choice != "auto-detect" else None,
            batch_size=16,
            warnings=warnings,
        )
    except BatchError as exc:
        return fail(str(exc))
    except Exception:  # noqa: BLE001
        return fail("Something went wrong reading that file. Please check it is a valid CSV.")

    state.add(
        result.frame["predicted_category"].tolist(),
        result.frame["confidence"].tolist(),
        "batch",
    )

    preview_cols = [
        result.text_column, "predicted_category", "confidence", "routing", "top_3",
    ]
    preview = result.frame[preview_cols].head(25).copy()
    preview[result.text_column] = preview[result.text_column].astype(str).str.slice(0, 160) + "…"

    tmp = Path(tempfile.gettempdir()) / "fintriage_predictions.csv"
    result.frame.to_csv(tmp, index=False)

    return (
        gr.update(value=batch_summary_card(result, THRESHOLD), visible=True),
        gr.update(value=preview, visible=True),
        gr.update(value=str(tmp), visible=True),
        state,
    )


def inspect_columns(file_obj):
    if file_obj is None:
        return gr.update(choices=["auto-detect"], value="auto-detect")
    try:
        frame, _ = read_csv_safely(
            file_obj.name, float(CFG["app"]["max_upload_mb"]), int(CFG["app"]["max_rows"])
        )
    except BatchError:
        return gr.update(choices=["auto-detect"], value="auto-detect")
    return gr.update(choices=["auto-detect", *frame.columns.tolist()], value="auto-detect")


# --------------------------------------------------------------------------- #
# Tab 4 - analytics dashboard
# --------------------------------------------------------------------------- #

def refresh_dashboard(state: SessionState):
    if state.total == 0:
        return (
            gr.update(
                value=empty_state(
                    "No complaints classified in this session yet. Classify one, or "
                    "upload a CSV, and these numbers will fill in."
                ),
                visible=True,
            ),
            empty_figure("Nothing to chart yet."),
            empty_figure("Nothing to chart yet."),
            empty_figure("Nothing to chart yet."),
            empty_figure("Nothing to chart yet."),
            gr.update(value=None, visible=False),
        )

    frame = state.frame()
    counts = frame["category"].value_counts().reindex(
        [d for d in DISPLAY if d in set(frame["category"])]
    )
    confidences = frame["confidence"].to_numpy(dtype=float)
    n_review = int((confidences < THRESHOLD).sum())
    n_auto = state.total - n_review
    n_batch = int((frame["source"] == "batch").sum())
    n_override = int((frame["source"] == "human override").sum())

    header = tiles(
        [
            ("Processed", f"{state.total:,}", "this session only"),
            ("Categories seen", f"{counts.size}", f"of {len(DISPLAY)}"),
            ("Mean confidence", f"{confidences.mean()*100:.1f}%", "raw softmax"),
            ("Manual review", f"{n_review:,}", f"{n_review / state.total:.1%} of volume"),
            ("From batch upload", f"{n_batch:,}", "CSV rows"),
            ("Human overrides", f"{n_override:,}", "corrected labels"),
        ]
    )
    table = (
        pd.DataFrame(
            {
                "Category": counts.index,
                "Complaints": counts.to_numpy(),
                "Share": [f"{v / state.total:.1%}" for v in counts.to_numpy()],
            }
        )
    )
    return (
        gr.update(value=header, visible=True),
        category_distribution(counts),
        category_share_donut(counts),
        confidence_histogram(confidences, THRESHOLD),
        routing_split(n_auto, n_review),
        gr.update(value=table, visible=True),
    )


def reset_session(state: SessionState):
    return (SessionState(), *refresh_dashboard(SessionState()))


# --------------------------------------------------------------------------- #
# Tab 6 - explainability
# --------------------------------------------------------------------------- #

def explain(text: str, target_label: str):
    if PREDICTOR is None:
        return (
            gr.update(value=note(f"The model is not loaded: {esc(MODEL_ERROR)}", warn=True), visible=True),
            gr.update(value=None, visible=False),
        )
    text = (text or "").strip()
    if len(text) < MIN_CHARS:
        return (
            gr.update(value=note(f"Please provide at least {MIN_CHARS} characters.", warn=True), visible=True),
            gr.update(value=None, visible=False),
        )
    explainer = get_explainer()
    target = None
    if target_label and target_label != "predicted category":
        target = DISPLAY.index(target_label)

    explanation = explainer.explain(text, target=target)
    top = ", ".join(f"<code>{esc(t.token)}</code>" for t in explanation.top_positive(8))
    summary = (
        f"<strong>{esc(display_name(explanation.label, CFG))}</strong> "
        f"&middot; confidence {explanation.confidence*100:.1f}%<br>"
        f"Most influential tokens: {top}<br>"
        f"<span style='font-size:.82rem'>{esc(explanation.method)}, "
        f"{explanation.steps} interpolation steps."
        + (" Text was truncated for the explanation." if explanation.truncated else "")
        + f"</span><br><span style='font-size:.82rem'>{esc(explanation.caveat)}</span>"
    )
    return (
        gr.update(value=note(summary), visible=True),
        gr.update(value=explanation.highlight_pairs(), visible=True),
    )


# --------------------------------------------------------------------------- #
# static tab content
# --------------------------------------------------------------------------- #

def overview_html() -> str:
    if PROVENANCE and QUALITY and SPLITS:
        data_tiles = tiles(
            [
                ("Source rows", f"{PROVENANCE['n_rows']:,}", "CFPB complaints"),
                ("After cleaning", f"{QUALITY['n_output']:,}", "deduplicated"),
                ("Held-out test", f"{SPLITS['sizes']['test']:,}", "never trained on"),
                ("Categories", "5", "product families"),
            ]
        )
    else:
        data_tiles = empty_state("Dataset reports not generated yet.")

    model_line = (
        f"The production model is <strong>{esc(WINNER)}</strong>, selected on validation "
        "macro F1 and deployment cost."
        if WINNER
        else "No production model selected yet."
    )
    return f"""
<div class="ft-card">
  <h3>What this does</h3>
  <p class="ft-hint">
    FinTriage AI reads a written financial complaint and predicts which product
    family it belongs to, so it can be routed to the team that handles that
    product. It returns a confidence score and, when that score is low, asks for
    a human instead of routing automatically.
  </p>
  {data_tiles}
</div>
<div style="height:14px"></div>
<div class="ft-card">
  <h3>How to use it</h3>
  <p class="ft-hint">Four things you can do here.</p>
  <table class="ft-table"><tbody>
    <tr><td><strong>Classify Complaint</strong></td><td>Paste one complaint. Get the category, the top three candidates, and a routing recommendation you can override.</td></tr>
    <tr><td><strong>Batch Classification</strong></td><td>Upload a CSV of complaints. Get predictions, a summary, and a downloadable results file.</td></tr>
    <tr><td><strong>Analytics Dashboard</strong></td><td>See the distribution of everything you have classified in this session.</td></tr>
    <tr><td><strong>Explainability</strong></td><td>See which tokens the prediction was most sensitive to.</td></tr>
  </tbody></table>
  <div style="height:14px"></div>
  {note(model_line)}
</div>
{disclaimer()}"""


def performance_html() -> str:
    if not ROWS:
        return empty_state(
            "No evaluation results are available yet. Train the models, then run "
            "scripts/select_model.py and scripts/evaluate.py."
        )
    basis = (
        f"<strong>{esc(WINNER)}</strong> is deployed. "
        + esc(SELECTION.get("decision_basis", ""))
        if SELECTION
        else ""
    )
    threshold_note = ""
    if THRESHOLD_INFO:
        threshold_note = note(
            f"Review threshold <strong>{THRESHOLD_INFO['threshold']:.2f}</strong>, chosen on the "
            f"validation split: {esc(THRESHOLD_INFO.get('selection_rule', ''))}. "
            f"It auto-routes {THRESHOLD_INFO.get('auto_coverage', 0):.1%} of complaints at "
            f"{THRESHOLD_INFO.get('achieved_auto_accuracy') or 0:.1%} accuracy. "
            f"Expected calibration error {THRESHOLD_INFO.get('expected_calibration_error')}. "
            + esc(THRESHOLD_INFO.get("calibration_note", ""))
        )
    return f"""
<div class="ft-card">
  <h3>Four architectures on the same held-out test set</h3>
  <p class="ft-hint">
    {COMPARISON['test_split_size']:,} complaints, never seen during training or
    model selection. Every number below is from our own run.
  </p>
  {comparison_table(ROWS, WINNER)}
  <div style="height:14px"></div>
  {note(basis) if basis else ""}
  {threshold_note}
</div>"""


def per_class_frame() -> pd.DataFrame:
    if not ROWS:
        return pd.DataFrame()
    records = []
    for row in ROWS:
        for label, scores in row["per_class"].items():
            records.append(
                {
                    "Model": row["model"],
                    "Category": display_name(label, CFG),
                    "Precision": round(scores["precision"], 4),
                    "Recall": round(scores["recall"], 4),
                    "F1": round(scores["f1"], 4),
                    "Support": scores["support"],
                }
            )
    return pd.DataFrame(records)


def about_html() -> str:
    rows_published = f"{PROVENANCE['n_rows']:,}" if PROVENANCE else "not recorded"
    source = (
        f"<a href='https://www.kaggle.com/datasets/{esc(CFG['data']['kaggle_slug'])}' "
        f"target='_blank' rel='noopener'>Kaggle: {esc(CFG['data']['kaggle_slug'])}</a>"
    )
    return f"""
<div class="ft-card">
  <h3>Data</h3>
  <p class="ft-hint">Where the training data comes from, and what we did to it.</p>
  <table class="ft-table"><tbody>
    <tr><td>Dataset</td><td>{source}, file <code>{esc(CFG['data']['csv_name'])}</code></td></tr>
    <tr><td>Underlying source</td><td>US Consumer Financial Protection Bureau complaint database (public domain). Narratives are scrubbed of personal information by the CFPB before publication.</td></tr>
    <tr><td>Rows as published</td><td>{rows_published}</td></tr>
    <tr><td>Cleaning</td><td>Dropped nulls and empties, removed exact duplicate narratives, dropped complaints under 3 tokens, truncated the longest.</td></tr>
    <tr><td>Leakage control</td><td>Near-duplicate narratives are grouped with MinHash/LSH and kept inside a single split, so a complaint cannot appear in both training and test in near-identical form.</td></tr>
    <tr><td>Split</td><td>Stratified 80/10/10, seed 42, group-aware.</td></tr>
  </tbody></table>
</div>
<div style="height:14px"></div>
<div class="ft-card">
  <h3>Attribution</h3>
  <p class="ft-hint">
    The problem framing, the five-category label set and the four-model comparison
    were inspired by
    <a href="https://github.com/Razan3ssam/Consumer-Complaint-Classifier" target="_blank"
       rel="noopener">Razan3ssam/Consumer-Complaint-Classifier</a>.
    That project's evaluation protocol is reused so the results are comparable.
    This implementation, its code, its near-duplicate handling, its calibrated
    routing threshold, its explanations and this interface are independent work,
    and every metric reported here is from our own training runs.
  </p>
</div>
<div style="height:14px"></div>
<div class="ft-card">
  <h3>Responsible use and privacy</h3>
  <p class="ft-hint">What this tool is and is not for.</p>
  <table class="ft-table"><tbody>
    <tr><td>Scope</td><td>Routing support for English-language complaints in five product families. Nothing else.</td></tr>
    <tr><td>Not advice</td><td>This is not a financial, legal or credit advisor and gives no recommendations about your situation.</td></tr>
    <tr><td>Confidence</td><td>Raw softmax scores, not calibrated probabilities. A high score is not a guarantee of correctness.</td></tr>
    <tr><td>Explanations</td><td>Gradient attributions show local sensitivity, not the model's reasoning.</td></tr>
    <tr><td>Uploads</td><td>Processed in memory for the request and offered back as a download. Nothing you paste or upload is stored or logged.</td></tr>
    <tr><td>Examples</td><td>The sample complaints in this app were written for it. No real consumer narrative is shown.</td></tr>
  </tbody></table>
</div>
{disclaimer()}"""


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #

def build() -> gr.Blocks:
    with gr.Blocks(css=CSS, title="FinTriage AI", theme=gr.themes.Soft()) as demo:
        session = gr.State(SessionState())
        gr.HTML(hero())

        if PREDICTOR is None:
            gr.HTML(
                note(
                    "<strong>The model is not loaded.</strong> Classification, batch "
                    "processing and explanations are unavailable in this instance. "
                    f"<br><code>{esc(MODEL_ERROR)}</code>",
                    warn=True,
                )
            )

        with gr.Tabs():
            with gr.Tab("Overview"):
                gr.HTML(overview_html())

            with gr.Tab("Classify Complaint"):
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.HTML('<div class="ft-card"><h3>Complaint text</h3>'
                                '<p class="ft-hint">Paste the complaint narrative. '
                                'English, at least 25 characters.</p></div>')
                        complaint = gr.Textbox(
                            lines=11,
                            placeholder="My credit card was charged an annual fee after I closed the account…",
                            label="",
                            show_label=False,
                        )
                        with gr.Row():
                            classify_btn = gr.Button("Classify complaint", variant="primary", scale=2)
                            clear_btn = gr.ClearButton([complaint], value="Clear", scale=1)
                        gr.Examples(
                            examples=example_texts(),
                            inputs=[complaint],
                            label="Example complaints (synthetic, written for this app)",
                            examples_per_page=6,
                        )
                    with gr.Column(scale=4):
                        verdict = gr.HTML(visible=False)
                        prob_plot = gr.Plot(
                            value=empty_figure("Submit a complaint to see the probabilities."),
                            label="",
                        )
                        with gr.Group(visible=False) as override_group:
                            gr.HTML('<p class="ft-hint" style="margin:8px 0 4px">'
                                    "Disagree? Record the correct category.</p>")
                            override = gr.Dropdown(
                                choices=DISPLAY, label="Reviewed category", interactive=True
                            )
                            override_btn = gr.Button("Record correction", size="sm")
                            override_msg = gr.HTML(visible=False)

                classify_btn.click(
                    classify,
                    inputs=[complaint, session],
                    outputs=[verdict, prob_plot, override, session],
                    api_name="classify",
                ).then(lambda: gr.update(visible=True), outputs=[override_group])
                override_btn.click(
                    apply_override, inputs=[override, session], outputs=[override_msg, session]
                )

            with gr.Tab("Batch Classification"):
                gr.HTML(
                    f'<div class="ft-card"><h3>Classify a CSV of complaints</h3>'
                    f'<p class="ft-hint">Up to {CFG["app"]["max_upload_mb"]} MB and '
                    f'{CFG["app"]["max_rows"]:,} rows. The text column is detected '
                    f'automatically, or pick it yourself. Nothing you upload is stored.</p></div>'
                )
                with gr.Row():
                    upload = gr.File(label="CSV file", file_types=[".csv"], file_count="single")
                    column_pick = gr.Dropdown(
                        choices=["auto-detect"], value="auto-detect", label="Text column"
                    )
                batch_btn = gr.Button("Classify file", variant="primary")
                batch_summary = gr.HTML(visible=False)
                batch_preview = gr.Dataframe(label="First 25 rows", visible=False, wrap=True)
                download = gr.File(label="Download full results", visible=False)

                upload.change(inspect_columns, inputs=[upload], outputs=[column_pick])
                batch_btn.click(
                    process_batch,
                    inputs=[upload, column_pick, session],
                    outputs=[batch_summary, batch_preview, download, session],
                )

            with gr.Tab("Analytics Dashboard"):
                gr.HTML(
                    '<div class="ft-card"><h3>This session</h3><p class="ft-hint">'
                    "Live counts from the complaints you have classified here. Not "
                    "demonstration data, and not shared with anyone else.</p></div>"
                )
                with gr.Row():
                    refresh_btn = gr.Button("Refresh", variant="primary", scale=1)
                    reset_btn = gr.Button("Clear session data", scale=1)
                dash_tiles = gr.HTML(visible=False)
                with gr.Row():
                    dist_plot = gr.Plot(value=empty_figure("Nothing to chart yet."), label="")
                    donut_plot = gr.Plot(value=empty_figure("Nothing to chart yet."), label="")
                with gr.Row():
                    conf_plot = gr.Plot(value=empty_figure("Nothing to chart yet."), label="")
                    routing_plot = gr.Plot(value=empty_figure("Nothing to chart yet."), label="")
                dash_table = gr.Dataframe(label="Category breakdown", visible=False)

                dash_outputs = [dash_tiles, dist_plot, donut_plot, conf_plot, routing_plot, dash_table]
                refresh_btn.click(refresh_dashboard, inputs=[session], outputs=dash_outputs)
                reset_btn.click(reset_session, inputs=[session], outputs=[session, *dash_outputs])

            with gr.Tab("Model Performance"):
                gr.HTML(performance_html())
                if ROWS:
                    with gr.Row():
                        gr.Plot(value=metric_comparison(ROWS, "accuracy", "Accuracy"), label="")
                        gr.Plot(value=metric_comparison(ROWS, "f1_macro", "Macro F1"), label="")
                    with gr.Row():
                        gr.Plot(value=latency_vs_size(ROWS), label="")
                    gr.HTML('<div class="ft-card"><h3>Per-class performance</h3>'
                            '<p class="ft-hint">Where each model struggles.</p></div>')
                    gr.Dataframe(value=per_class_frame(), label="", wrap=True)
                    gr.HTML('<div class="ft-card"><h3>Confusion matrices</h3>'
                            '<p class="ft-hint">Row-normalised, held-out test set.</p></div>')
                    with gr.Row():
                        for row in ROWS:
                            gr.Plot(
                                value=confusion_heatmap(
                                    row["confusion_matrix"], DISPLAY, row["model"]
                                ),
                                label="",
                            )
                    for name in ("SimpleRNN", "LSTM", "GRU", "DistilBERT"):
                        meta = read_json(f"train_{name}.json")
                        if meta and meta.get("history"):
                            gr.Plot(
                                value=training_curves(meta["history"], f"{name}: training curves"),
                                label="",
                            )

            with gr.Tab("Explainability"):
                gr.HTML(
                    '<div class="ft-card"><h3>Which tokens moved the prediction</h3>'
                    '<p class="ft-hint">Layer Integrated Gradients over the embedding '
                    "layer of the deployed model. The stronger a token is shaded, the "
                    "more the prediction moved with it. This shows sensitivity, not "
                    "reasoning.</p></div>"
                )
                explain_text = gr.Textbox(
                    lines=8,
                    label="Complaint text",
                    placeholder="Paste a complaint to explain…",
                    value=EXAMPLES[2][1],
                )
                explain_target = gr.Dropdown(
                    choices=["predicted category", *DISPLAY],
                    value="predicted category",
                    label="Explain which category",
                )
                explain_btn = gr.Button("Explain prediction", variant="primary")
                explain_summary = gr.HTML(visible=False)
                highlight = gr.HighlightedText(
                    label="Token attributions",
                    visible=False,
                    show_legend=False,
                    color_map=None,
                )
                explain_btn.click(
                    explain,
                    inputs=[explain_text, explain_target],
                    outputs=[explain_summary, highlight],
                )

            with gr.Tab("About the Project"):
                gr.HTML(about_html())

    return demo


if __name__ == "__main__":
    build().queue(max_size=16).launch(server_name="0.0.0.0", show_api=False)
