"""Visual identity for the Gradio app: deep navy, electric blue, clean white."""

NAVY = "#0B1A33"
NAVY_SOFT = "#13284D"
BLUE = "#2563EB"
BLUE_BRIGHT = "#3B82F6"
INK = "#0F172A"
MUTED = "#5B6B82"
LINE = "#E2E8F0"
SURFACE = "#FFFFFF"
CANVAS = "#F6F8FC"
OK = "#047857"
WARN = "#B45309"

# One colour per class, kept stable everywhere so a category reads the same in
# the bar chart, the dashboard and the batch table.
CATEGORY_COLORS = {
    "Credit Card": "#2563EB",
    "Credit Reporting": "#7C3AED",
    "Debt Collection": "#DB2777",
    "Mortgages and Loans": "#0891B2",
    "Retail Banking": "#059669",
}

CSS = f"""
:root {{
  --ft-navy: {NAVY}; --ft-navy-soft: {NAVY_SOFT}; --ft-blue: {BLUE};
  --ft-blue-bright: {BLUE_BRIGHT}; --ft-ink: {INK}; --ft-muted: {MUTED};
  --ft-line: {LINE}; --ft-surface: {SURFACE}; --ft-canvas: {CANVAS};
  --ft-ok: {OK}; --ft-warn: {WARN};
}}
.gradio-container {{
  max-width: 1180px !important;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto, sans-serif;
  background: var(--ft-canvas);
}}
footer {{ display: none !important; }}

/* ---------- masthead ---------- */
#ft-hero {{
  background: linear-gradient(135deg, var(--ft-navy) 0%, var(--ft-navy-soft) 58%, #1E3A8A 100%);
  border-radius: 18px; padding: 30px 32px; margin-bottom: 18px; color: #fff;
  box-shadow: 0 12px 32px rgba(11,26,51,.22);
}}
#ft-hero h1 {{
  margin: 0 0 6px; font-size: 1.95rem; font-weight: 700; letter-spacing: -.02em; color: #fff;
}}
#ft-hero .ft-sub {{ margin: 0; font-size: 1.02rem; color: #BFD3F2; font-weight: 450; }}
#ft-hero .ft-chips {{ margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }}
#ft-hero .ft-chip {{
  background: rgba(255,255,255,.12); border: 1px solid rgba(255,255,255,.2);
  border-radius: 999px; padding: 5px 13px; font-size: .8rem; color: #E4EDFB;
}}

/* ---------- cards ---------- */
.ft-card {{
  background: var(--ft-surface); border: 1px solid var(--ft-line);
  border-radius: 14px; padding: 20px 22px; box-shadow: 0 1px 3px rgba(15,23,42,.05);
}}
.ft-card h3 {{
  margin: 0 0 4px; font-size: 1.04rem; font-weight: 650; color: var(--ft-ink);
}}
.ft-card p.ft-hint {{ margin: 0 0 14px; font-size: .86rem; color: var(--ft-muted); }}

/* ---------- verdict ---------- */
.ft-verdict {{
  border-radius: 14px; padding: 20px 22px; border: 1px solid var(--ft-line);
  background: var(--ft-surface);
}}
.ft-verdict .ft-cat {{
  font-size: 1.55rem; font-weight: 700; color: var(--ft-navy); letter-spacing: -.01em;
}}
.ft-verdict .ft-conf {{ font-size: .9rem; color: var(--ft-muted); margin-top: 2px; }}
.ft-bar {{
  height: 8px; border-radius: 999px; background: #EDF1F7; overflow: hidden; margin: 14px 0 4px;
}}
.ft-bar > span {{ display: block; height: 100%; border-radius: 999px; background: var(--ft-blue); }}
.ft-badge {{
  display: inline-flex; align-items: center; gap: 7px; border-radius: 999px;
  padding: 6px 14px; font-size: .84rem; font-weight: 600; margin-top: 14px;
}}
.ft-badge.auto {{ background: #E7F6EF; color: var(--ft-ok); border: 1px solid #A7E0C6; }}
.ft-badge.review {{ background: #FEF4E6; color: var(--ft-warn); border: 1px solid #F3D19B; }}
.ft-badge .dot {{ width: 7px; height: 7px; border-radius: 50%; background: currentColor; }}

/* ---------- metric tiles ---------- */
.ft-tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; }}
.ft-tile {{
  background: var(--ft-surface); border: 1px solid var(--ft-line);
  border-radius: 12px; padding: 16px 18px;
}}
.ft-tile .ft-k {{
  font-size: .72rem; text-transform: uppercase; letter-spacing: .07em;
  color: var(--ft-muted); font-weight: 650;
}}
.ft-tile .ft-v {{ font-size: 1.6rem; font-weight: 700; color: var(--ft-navy); margin-top: 5px; }}
.ft-tile .ft-n {{ font-size: .78rem; color: var(--ft-muted); margin-top: 2px; }}

/* ---------- notes ---------- */
.ft-note {{
  border-left: 3px solid var(--ft-blue); background: #F1F6FE;
  padding: 12px 16px; border-radius: 0 10px 10px 0; font-size: .86rem; color: #26405F;
}}
.ft-note.warn {{ border-left-color: var(--ft-warn); background: #FEF8EE; color: #6B4A12; }}
.ft-disclaimer {{
  margin-top: 18px; padding: 14px 18px; border: 1px dashed #C9D4E4;
  border-radius: 12px; background: #FBFCFE; font-size: .82rem; color: var(--ft-muted);
}}
.ft-empty {{
  text-align: center; padding: 42px 20px; color: var(--ft-muted); font-size: .9rem;
  border: 1px dashed var(--ft-line); border-radius: 12px; background: var(--ft-surface);
}}
table.ft-table {{ width: 100%; border-collapse: collapse; font-size: .86rem; }}
table.ft-table th {{
  text-align: left; padding: 9px 12px; background: #F3F6FB; color: var(--ft-ink);
  font-weight: 650; border-bottom: 1px solid var(--ft-line); font-size: .78rem;
  text-transform: uppercase; letter-spacing: .04em;
}}
table.ft-table td {{ padding: 9px 12px; border-bottom: 1px solid #EFF2F7; color: #223349; }}
table.ft-table tr.ft-winner td {{ background: #F0F6FF; font-weight: 600; }}

/* ---------- tabs ---------- */
.tab-nav button {{ font-weight: 600 !important; font-size: .92rem !important; }}
.tab-nav button.selected {{ color: var(--ft-blue) !important; }}

@media (max-width: 640px) {{
  #ft-hero {{ padding: 22px 20px; }}
  #ft-hero h1 {{ font-size: 1.5rem; }}
  .ft-card {{ padding: 16px; }}
  .ft-verdict .ft-cat {{ font-size: 1.3rem; }}
  .ft-tiles {{ grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); }}
}}
"""

DISCLAIMER = (
    "This tool provides automated complaint categorization for informational and "
    "workflow-support purposes. Predictions may be incorrect and should be reviewed "
    "before consequential decisions."
)
