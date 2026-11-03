"""Capture authentic screenshots of the running FinTriage AI app.

Drives the real application with the real fine-tuned DistilBERT checkpoint. No
value shown in any image is edited: every prediction, probability and metric is
whatever the app produced at capture time.

    python scripts/capture_screenshots.py --url http://localhost:7860
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "linkedin_assets" / "raw"

COMPLAINT = (
    "A collection agency has been calling me five or six times a day, including "
    "at my workplace after I asked them in writing to stop. They are demanding "
    "payment for a debt I already settled last year, and they refuse to send me "
    "written validation of the amount they claim I still owe."
)
EXPLAIN_TEXT = (
    "My mortgage servicer changed without any notice to me and my escrow payment "
    "jumped by four hundred dollars. When I asked for a breakdown of the new "
    "escrow analysis they sent an unreadable form, and my modification "
    "application has been pending for six months with no decision."
)


def tab(page, name: str) -> None:
    page.get_by_role("tab", name=name, exact=True).click()
    page.wait_for_timeout(1200)


def shoot(page, path: Path, selector: str | None = None, full: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    page.wait_for_timeout(600)
    if selector:
        page.locator(selector).first.screenshot(path=str(path))
    else:
        page.screenshot(path=str(path), full_page=full)
    kb = path.stat().st_size / 1024
    print(f"  saved {path.name}  ({kb:.0f} KB)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:7860")
    args = ap.parse_args()

    csv_path = ROOT / "linkedin_assets" / "demo_complaints.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_text(
        "case_id,narrative\n"
        'DEMO-1,"I was charged an annual fee on my card after I had already called to close the account and the representative confirmed the closure."\n'
        'DEMO-2,"There is an account on my credit report that does not belong to me. I filed a dispute with the bureau two months ago and nothing has changed."\n'
        'DEMO-3,"A collection agency keeps calling me six times a day at work demanding payment for a debt I already settled in full."\n'
        'DEMO-4,"My mortgage servicer changed without notice and my escrow payment jumped by four hundred dollars with no breakdown provided."\n'
        'DEMO-5,"My checking account was overdrawn and the bank charged four separate overdraft fees in one day by reordering transactions."\n'
        'DEMO-6,"The bank placed a hold on my payroll deposit for three days without explaining why, and I could not pay rent on time."\n'
        'DEMO-7,"My credit card limit was reduced without warning which pushed my utilisation up and hurt my score."\n',
        encoding="utf-8",
    )
    print(f"synthetic demo CSV: {csv_path}")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        ctx = browser.new_context(
            viewport={"width": 1480, "height": 1020},
            device_scale_factor=2,          # retina-sharp text
        )
        page = ctx.new_page()
        print(f"opening {args.url}")
        # Gradio holds a long-lived connection open, so "networkidle" never
        # fires. Wait for the app's own markup instead.
        page.goto(args.url, wait_until="domcontentloaded", timeout=90_000)
        page.wait_for_selector("#ft-hero", timeout=90_000)
        page.get_by_role("tab", name="Overview", exact=True).wait_for(timeout=60_000)
        page.wait_for_timeout(3000)

        # Fail loudly rather than ship a screenshot with an error banner in it.
        if page.locator("text=The model is not loaded").count():
            print("ERROR: the app is showing the model-missing banner", file=sys.stderr)
            return 1

        # ---- 1. hero + overview --------------------------------------------
        print("1/5 overview")
        shoot(page, RAW / "01_overview.png", selector=".gradio-container")

        # ---- 2. live classification ----------------------------------------
        print("2/5 classification")
        tab(page, "Classify Complaint")
        box = page.locator("textarea:visible").first
        box.click()
        box.fill(COMPLAINT)
        page.get_by_role("button", name="Classify complaint").click()
        page.wait_for_selector(".ft-verdict", timeout=60_000)
        page.wait_for_timeout(2500)
        shoot(page, RAW / "02_classification.png", selector=".gradio-container")

        # ---- 3. model performance ------------------------------------------
        print("3/5 model performance")
        tab(page, "Model Performance")
        page.get_by_text("Four architectures on the same held-out test set").wait_for(timeout=60_000)
        page.wait_for_timeout(4000)
        shoot(page, RAW / "03_model_performance.png", selector=".gradio-container")

        # ---- 4. explainability ---------------------------------------------
        print("4/5 explainability")
        tab(page, "Explainability")
        ta = page.locator("textarea:visible").first
        ta.click()
        ta.fill(EXPLAIN_TEXT)
        page.get_by_role("button", name="Explain prediction").click()
        page.wait_for_timeout(9000)
        shoot(page, RAW / "04_explainability.png", selector=".gradio-container")

        # ---- 5. batch + analytics ------------------------------------------
        print("5/5 batch + analytics")
        tab(page, "Batch Classification")
        # Every tab keeps a file input in the DOM; pick the one in this panel.
        uploads = page.locator("input[type=file]")
        print(f"   file inputs in DOM: {uploads.count()}")
        uploads.first.set_input_files(str(csv_path))
        page.wait_for_timeout(3000)
        page.get_by_role("button", name="Classify file").click()
        # Wait for the summary tiles rather than guessing a duration.
        page.wait_for_selector("text=Rows in file", timeout=120_000)
        page.wait_for_selector("text=Text read from column", timeout=60_000)
        page.wait_for_timeout(3000)
        if page.locator("text=Drop File Here").count() and not page.locator("text=Rows in file").count():
            print("ERROR: batch did not process", file=sys.stderr)
            return 1
        shoot(page, RAW / "05a_batch.png", selector=".gradio-container")

        tab(page, "Analytics Dashboard")
        page.get_by_role("button", name="Refresh").click()
        page.wait_for_selector("text=From batch upload", timeout=60_000)
        page.wait_for_timeout(6000)
        shoot(page, RAW / "05b_analytics.png", selector=".gradio-container")

        browser.close()
    print(f"\nraw captures in {RAW}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
