"""Plotly figures for the app. One builder per chart so the dashboard and the
model-performance tab cannot drift apart stylistically."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.app_theme import BLUE, CATEGORY_COLORS, INK, LINE, MUTED, NAVY

LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="-apple-system, Segoe UI, Inter, sans-serif", size=12, color=INK),
    margin=dict(l=10, r=14, t=42, b=10),
    hoverlabel=dict(bgcolor="white", bordercolor=LINE),
)
AXIS = dict(showgrid=True, gridcolor="#EEF2F7", zeroline=False, linecolor=LINE)


def _fig(title: str, height: int = 320) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **LAYOUT,
        title=dict(text=title, font=dict(size=14, color=NAVY), x=0, xanchor="left"),
        height=height,
    )
    return fig


def probability_bars(probabilities: dict[str, float]) -> go.Figure:
    from src.config import display_name

    items = sorted(probabilities.items(), key=lambda kv: kv[1])
    names = [display_name(k) for k, _ in items]
    values = [v * 100 for _, v in items]
    fig = _fig("Probability by category", height=280)
    fig.add_bar(
        x=values,
        y=names,
        orientation="h",
        marker=dict(color=[CATEGORY_COLORS.get(n, BLUE) for n in names]),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        hovertemplate="%{y}: %{x:.2f}%<extra></extra>",
    )
    fig.update_xaxes(**AXIS, range=[0, 112], ticksuffix="%")
    fig.update_yaxes(showgrid=False, linecolor=LINE)
    return fig


def category_distribution(counts: pd.Series) -> go.Figure:
    fig = _fig("Predicted category distribution")
    total = max(int(counts.sum()), 1)
    fig.add_bar(
        x=list(counts.index),
        y=counts.to_numpy(),
        marker=dict(color=[CATEGORY_COLORS.get(n, BLUE) for n in counts.index]),
        text=[f"{v:,}<br>{v / total:.1%}" for v in counts.to_numpy()],
        textposition="outside",
        hovertemplate="%{x}: %{y:,} complaints<extra></extra>",
    )
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(**AXIS, title="Complaints")
    return fig


def category_share_donut(counts: pd.Series) -> go.Figure:
    fig = _fig("Share of volume")
    fig.add_pie(
        labels=list(counts.index),
        values=counts.to_numpy(),
        hole=0.62,
        marker=dict(colors=[CATEGORY_COLORS.get(n, BLUE) for n in counts.index]),
        textinfo="percent",
        hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
    )
    fig.update_layout(legend=dict(orientation="h", y=-0.12, font=dict(size=11)))
    return fig


def confidence_histogram(confidences: np.ndarray, threshold: float) -> go.Figure:
    fig = _fig("Confidence distribution")
    fig.add_histogram(
        x=confidences * 100,
        nbinsx=20,
        marker=dict(color=BLUE, line=dict(color="white", width=1)),
        hovertemplate="%{x}: %{y} complaints<extra></extra>",
    )
    fig.add_vline(
        x=threshold * 100,
        line=dict(color="#B45309", dash="dash", width=2),
        annotation_text=f"review below {threshold:.0%}",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#B45309"),
    )
    fig.update_xaxes(**AXIS, title="Confidence", ticksuffix="%")
    fig.update_yaxes(**AXIS, title="Complaints")
    return fig


def routing_split(n_auto: int, n_review: int) -> go.Figure:
    fig = _fig("Routing outcome", height=250)
    fig.add_bar(
        x=[n_auto, n_review],
        y=["Automated", "Manual review"],
        orientation="h",
        marker=dict(color=["#047857", "#B45309"]),
        text=[f"{n_auto:,}", f"{n_review:,}"],
        textposition="outside",
        hovertemplate="%{y}: %{x:,}<extra></extra>",
    )
    fig.update_xaxes(**AXIS, title="Complaints")
    fig.update_yaxes(showgrid=False, linecolor=LINE)
    return fig


def metric_comparison(rows: list[dict], metric: str, title: str) -> go.Figure:
    fig = _fig(title)
    names = [r["model"] for r in rows]
    values = [r[metric] for r in rows]
    fig.add_bar(
        x=names,
        y=values,
        marker=dict(color=[BLUE if i < len(names) - 1 else NAVY for i in range(len(names))]),
        text=[f"{v:.4f}" for v in values],
        textposition="outside",
        hovertemplate="%{x}: %{y:.4f}<extra></extra>",
    )
    lo = min(values) - 0.04 if values else 0
    fig.update_xaxes(showgrid=False, linecolor=LINE)
    fig.update_yaxes(**AXIS, range=[max(lo, 0), 1.0])
    return fig


def latency_vs_size(rows: list[dict]) -> go.Figure:
    fig = _fig("Cost of accuracy: latency vs. model size")
    for row in rows:
        fig.add_scatter(
            x=[row["cpu_latency_p50_ms"]],
            y=[row["artifact_size_mb"]],
            mode="markers+text",
            marker=dict(size=14 + row["f1_macro"] * 22, color=BLUE, line=dict(color="white", width=2)),
            text=[row["model"]],
            textposition="top center",
            textfont=dict(size=11),
            name=row["model"],
            hovertemplate=(
                f"<b>{row['model']}</b><br>macro F1 {row['f1_macro']:.4f}"
                "<br>%{x:.0f} ms &middot; %{y:.0f} MB<extra></extra>"
            ),
        )
    fig.update_layout(showlegend=False)
    fig.update_xaxes(**AXIS, title="CPU latency p50 (ms, log)", type="log")
    fig.update_yaxes(**AXIS, title="Artifact size (MB, log)", type="log")
    return fig


def confusion_heatmap(matrix: list[list[int]], labels: list[str], title: str) -> go.Figure:
    arr = np.array(matrix, dtype=float)
    normalised = arr / np.clip(arr.sum(axis=1, keepdims=True), 1, None)
    fig = _fig(title, height=420)
    fig.add_heatmap(
        z=normalised,
        x=labels,
        y=labels,
        colorscale=[[0, "#F7FAFF"], [0.5, "#93B4F5"], [1, NAVY]],
        text=[[f"{int(arr[i][j]):,}<br>{normalised[i][j]:.0%}" for j in range(len(labels))]
              for i in range(len(labels))],
        texttemplate="%{text}",
        textfont=dict(size=10),
        hovertemplate="true %{y} → predicted %{x}<br>%{text}<extra></extra>",
        colorbar=dict(title="row share", thickness=12, tickformat=".0%"),
    )
    fig.update_xaxes(title="Predicted", side="bottom", linecolor=LINE)
    fig.update_yaxes(title="Actual", autorange="reversed", linecolor=LINE)
    return fig


def training_curves(history: list[dict], title: str) -> go.Figure:
    fig = _fig(title)
    epochs = [h["epoch"] for h in history]
    fig.add_scatter(x=epochs, y=[h["train_loss"] for h in history], name="train loss",
                    mode="lines+markers", line=dict(color=BLUE, width=2.5))
    fig.add_scatter(x=epochs, y=[h["val_loss"] for h in history], name="val loss",
                    mode="lines+markers", line=dict(color=MUTED, width=2.5, dash="dot"))
    fig.add_scatter(x=epochs, y=[h["val_f1_macro"] for h in history], name="val macro F1",
                    mode="lines+markers", line=dict(color="#047857", width=2.5), yaxis="y2")
    fig.update_layout(
        yaxis2=dict(overlaying="y", side="right", title="macro F1", showgrid=False,
                    range=[0, 1], linecolor=LINE),
        legend=dict(orientation="h", y=1.12, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(**AXIS, title="Epoch", dtick=1)
    fig.update_yaxes(**AXIS, title="Loss")
    return fig


def empty_figure(message: str) -> go.Figure:
    fig = _fig("", height=260)
    fig.add_annotation(text=message, showarrow=False, font=dict(size=13, color=MUTED),
                       x=0.5, y=0.5, xref="paper", yref="paper")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
