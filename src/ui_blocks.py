"""HTML fragments for the app. Kept out of app.py so the callbacks stay readable."""
from __future__ import annotations

import html
from typing import Iterable

from src.app_theme import CATEGORY_COLORS, DISCLAIMER


def esc(value) -> str:
    return html.escape(str(value))


def hero() -> str:
    chips = [
        "5 complaint categories",
        "Confidence-based routing",
        "Batch CSV",
        "Token-level explanations",
    ]
    chip_html = "".join(f'<span class="ft-chip">{esc(c)}</span>' for c in chips)
    return f"""
<div id="ft-hero">
  <h1>FinTriage AI</h1>
  <p class="ft-sub">Intelligent Financial Complaint Classification &amp; Routing</p>
  <div class="ft-chips">{chip_html}</div>
</div>"""


def empty_state(message: str) -> str:
    return f'<div class="ft-empty">{esc(message)}</div>'


def note(text: str, warn: bool = False) -> str:
    return f'<div class="ft-note{" warn" if warn else ""}">{text}</div>'


def disclaimer() -> str:
    return f'<div class="ft-disclaimer"><strong>Please note.</strong> {esc(DISCLAIMER)}</div>'


def tiles(items: Iterable[tuple[str, str, str]]) -> str:
    body = "".join(
        f'<div class="ft-tile"><div class="ft-k">{esc(k)}</div>'
        f'<div class="ft-v">{esc(v)}</div>'
        f'<div class="ft-n">{esc(n)}</div></div>'
        for k, v, n in items
    )
    return f'<div class="ft-tiles">{body}</div>'


def verdict_card(prediction, calibrated: bool) -> str:
    pct = prediction.confidence * 100
    colour = CATEGORY_COLORS.get(prediction.display, "#2563EB")
    if prediction.needs_review:
        badge = (
            '<div class="ft-badge review"><span class="dot"></span>'
            "Manual Review Recommended</div>"
        )
    else:
        badge = (
            '<div class="ft-badge auto"><span class="dot"></span>'
            "Automated Classification Available</div>"
        )
    rows = "".join(
        f"""<tr><td style="width:46%">{esc(name)}</td>
        <td style="width:38%"><div class="ft-bar"><span style="width:{value*100:.1f}%;
        background:{CATEGORY_COLORS.get(name, '#2563EB')}"></span></div></td>
        <td style="text-align:right;width:16%">{value*100:.1f}%</td></tr>"""
        for name, value in _top3_display(prediction)
    )
    cal = (
        "Confidence is a calibrated probability."
        if calibrated
        else "Confidence is a raw softmax score, not a calibrated probability — "
        "treat it as a routing signal, not a guarantee of correctness."
    )
    return f"""
<div class="ft-verdict">
  <div class="ft-cat" style="color:{colour}">{esc(prediction.display)}</div>
  <div class="ft-conf">Confidence {pct:.1f}% &middot; routing threshold {prediction.threshold*100:.0f}%</div>
  <div class="ft-bar"><span style="width:{pct:.1f}%;background:{colour}"></span></div>
  {badge}
  <h3 style="margin:22px 0 8px;font-size:.78rem;text-transform:uppercase;
      letter-spacing:.07em;color:#5B6B82">Top 3 categories</h3>
  <table class="ft-table"><tbody>{rows}</tbody></table>
  <p style="margin:14px 0 0;font-size:.79rem;color:#5B6B82">{esc(cal)}</p>
</div>"""


def _top3_display(prediction) -> list[tuple[str, float]]:
    from src.config import display_name

    return [(display_name(label), value) for label, value in prediction.top_k(3)]


def batch_summary_card(result, threshold: float) -> str:
    review_pct = result.n_review / max(result.n_classified, 1) * 100
    body = tiles(
        [
            ("Rows in file", f"{result.n_input_rows:,}", "as uploaded"),
            ("Classified", f"{result.n_classified:,}", "usable complaint text"),
            ("Skipped", f"{result.n_skipped:,}", "empty or too short"),
            ("Manual review", f"{result.n_review:,}", f"{review_pct:.1f}% below {threshold:.0%}"),
        ]
    )
    warnings = "".join(note(esc(w), warn=True) for w in result.warnings)
    column = note(f"Text read from column <strong>{esc(result.text_column)}</strong>.")
    return body + column + warnings


def comparison_table(rows: list[dict], winner: str | None) -> str:
    if not rows:
        return empty_state("No evaluation results yet. Train the models and run scripts/evaluate.py.")
    header = (
        "<tr><th>Model</th><th>Accuracy</th><th>Macro F1</th><th>Weighted F1</th>"
        "<th>Train time</th><th>CPU latency (p50)</th><th>Size</th><th>Params</th></tr>"
    )
    body = []
    for row in rows:
        css = ' class="ft-winner"' if row["model"] == winner else ""
        tag = " &nbsp;<span style='color:#2563EB;font-size:.72rem'>PRODUCTION</span>" if row["model"] == winner else ""
        params = f"{row['trainable_parameters']:,}" if row.get("trainable_parameters") else "—"
        train_time = f"{row['train_time_sec']:,.0f} s" if row.get("train_time_sec") else "—"
        body.append(
            f"<tr{css}><td>{esc(row['model'])}{tag}</td>"
            f"<td>{row['accuracy']:.4f}</td><td>{row['f1_macro']:.4f}</td>"
            f"<td>{row['f1_weighted']:.4f}</td><td>{train_time}</td>"
            f"<td>{row['cpu_latency_p50_ms']:.0f} ms</td>"
            f"<td>{row['artifact_size_mb']:.0f} MB</td><td>{params}</td></tr>"
        )
    return f'<table class="ft-table"><thead>{header}</thead><tbody>{"".join(body)}</tbody></table>'
