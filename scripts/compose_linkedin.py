"""Compose the raw app captures into LinkedIn-ready 1080x1350 frames.

Only framing: crop, scale, and a navy header/footer band. No pixel inside the
screenshot is edited, so every prediction, probability and metric shown is what
the application actually produced.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "linkedin_assets" / "raw"
OUT = ROOT / "linkedin_launch" / "screenshots"
ALT = ROOT / "linkedin_assets"

W, H = 1080, 1350
HEADER, FOOTER = 168, 86
PAD = 40
NAVY, NAVY2 = (11, 26, 51), (19, 40, 77)
BLUE_T = (191, 211, 242)
CANVAS = (246, 248, 252)
MUTED = (120, 136, 158)

FONTS = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/System/Library/Fonts/SFNS.ttf",
]


def font(size: int, bold: bool = False):
    for path in FONTS if bold else reversed(FONTS):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


def rounded_shadow(img: Image.Image, radius: int = 14) -> Image.Image:
    """Round the corners and drop a soft shadow, so the capture reads as a card."""
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, *[d - 1 for d in img.size]],
                                           radius=radius, fill=255)
    card = Image.new("RGBA", img.size, (0, 0, 0, 0))
    card.paste(img.convert("RGB"), (0, 0), mask)

    pad = 18
    canvas = Image.new("RGBA", (img.width + pad * 2, img.height + pad * 2), (0, 0, 0, 0))
    shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [pad, pad + 4, pad + img.width, pad + img.height + 4],
        radius=radius, fill=(15, 23, 42, 46))
    shadow = shadow.filter(ImageFilter.GaussianBlur(10))
    canvas.alpha_composite(shadow)
    canvas.alpha_composite(card, (pad, pad))
    return canvas


def frame(panels: list[Image.Image], title: str, subtitle: str, footer: str) -> Image.Image:
    base = Image.new("RGB", (W, H), CANVAS)
    d = ImageDraw.Draw(base)

    # header gradient
    for y in range(HEADER):
        t = y / HEADER
        d.line([(0, y), (W, y)],
               fill=tuple(int(NAVY[i] + (NAVY2[i] - NAVY[i]) * t) for i in range(3)))
    d.text((PAD, 44), title, font=font(42, True), fill=(255, 255, 255))
    d.text((PAD, 103), subtitle, font=font(23), fill=BLUE_T)

    avail_w = W - PAD * 2
    avail_h = H - HEADER - FOOTER - PAD
    gap = 22 if len(panels) > 1 else 0

    # One scale for the whole stack, so panels keep their relative size and sit
    # flush together; the block is then centred. Splitting the height evenly
    # instead leaves a gap in the middle and shrinks whichever panel is taller.
    block_w = max(p.width for p in panels)
    block_h = sum(p.height for p in panels)
    scale = min(avail_w / block_w, (avail_h - gap * (len(panels) - 1)) / block_h)

    scaled = [p.resize((int(p.width * scale), int(p.height * scale)), Image.LANCZOS)
              for p in panels]
    total = sum(p.height for p in scaled) + gap * (len(scaled) - 1)
    y = HEADER + (avail_h - total) // 2

    for panel in scaled:
        card = rounded_shadow(panel)
        base.paste(card, ((W - card.width) // 2, y - 18), card)
        y += panel.height + gap

    d.text((PAD, H - FOOTER + 26), footer, font=font(19), fill=MUTED)
    return base


TARGET_RATIO = (W - PAD * 2) / (H - HEADER - FOOTER - PAD)   # width / height


def crop(name: str, top: int, bottom: int | None = None, fit: bool = True) -> Image.Image:
    """Crop from `top`. With fit=True the height is chosen so the panel fills the
    frame's content area instead of floating in it."""
    img = Image.open(RAW / name)
    if bottom is None:
        bottom = top + int(img.width / TARGET_RATIO) if fit else img.height
    return img.crop((0, top, img.width, min(bottom, img.height)))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SRC = "Captured from the running FinTriage AI application"

    # The result panel of the classify tab is its own column; crop it directly.
    classify = Image.open(RAW / "02_classification.png").crop((1316, 555, 2325, 2035))

    specs = [
        ("01_fintriage_overview.png",
         [crop("01_overview.png", 0, 2040)],
         "FinTriage AI",
         "Financial complaint classification & intelligent routing",
         SRC),
        ("02_ai_classification.png",
         [classify],
         "Real-time classification",
         "Fine-tuned DistilBERT · top-3 categories · confidence-based routing",
         SRC + " — synthetic complaint, live model output"),
        ("03_model_comparison.png",
         [crop("03_model_performance.png", 432)],
         "Four architectures, one test set",
         "12,435 held-out complaints · DistilBERT selected on validation",
         SRC),
        ("04_explainable_ai.png",
         [crop("04_explainability.png", 440, 800),
          crop("04_explainability.png", 1596, 2284)],
         "Explainable predictions",
         "Layer Integrated Gradients over the deployed model",
         SRC + " — synthetic complaint, live attributions"),
        ("05_batch_analytics.png",
         [crop("05a_batch.png", 1140, 2112),
          crop("05b_analytics.png", 920, 1880)],
         "Batch processing & analytics",
         "CSV classification with session-level category distribution",
         SRC + " — synthetic demonstration data"),
    ]

    for name, panels, title, subtitle, footer in specs:
        img = frame(panels, title, subtitle, footer)
        img.save(OUT / name, "PNG", optimize=True)
        img.save(ALT / name, "PNG", optimize=True)
        print(f"  {name}  {img.size[0]}x{img.size[1]}  "
              f"{(OUT / name).stat().st_size/1024:.0f} KB")
    print(f"\nfinal frames: {OUT}")
    print(f"originals preserved: {RAW}")


if __name__ == "__main__":
    main()
