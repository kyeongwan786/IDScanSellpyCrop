"""Calibration card for measuring how much the printer cuts off each edge.

The card is printed exactly like a normal photo (filling the 86 x 54 mm C
card area, no compensation).  Every 0.25 mm from each edge a numbered line is
drawn.  After printing, the user reads the number of the outermost line that
is still fully visible on each side; that number is how many millimetres the
printer removed there.  The app then places photos inside exactly that
visible window.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CARD_W_MM = 86.0
CARD_H_MM = 54.0
STEP_MM = 0.25
MAX_MM = 3.0
COLORS = [(220, 0, 0), (0, 70, 220), (0, 140, 60), (0, 0, 0)]


def _font(size_px: int) -> ImageFont.ImageFont:
    candidates = [
        "arialbd.ttf", "arial.ttf",  # Windows
        "DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    ]
    for name in candidates:
        try:
            return ImageFont.truetype(name, size_px)
        except OSError:
            continue
    try:
        return ImageFont.load_default(size_px)
    except TypeError:  # Pillow < 10.1
        return ImageFont.load_default()


def line_values() -> list[float]:
    count = int(round(MAX_MM / STEP_MM))
    return [round(STEP_MM * (i + 1), 2) for i in range(count)]


def render_calibration_card(width_px: int, height_px: int) -> Image.Image:
    """Render the card at the printer's pixel size for the full C card area."""
    img = Image.new("RGB", (width_px, height_px), "white")
    draw = ImageDraw.Draw(img)
    sx = width_px / CARD_W_MM
    sy = height_px / CARD_H_MM

    def X(mm: float) -> float:
        return mm * sx

    def Y(mm: float) -> float:
        return mm * sy

    line_w = max(1, round(0.08 * sx))
    font = _font(max(8, round(1.7 * sy)))
    values = line_values()

    # Nested rectangles: line for value d sits d mm inside every edge.
    for i, d in enumerate(values):
        color = COLORS[i % len(COLORS)]
        draw.rectangle(
            [X(d), Y(d), X(CARD_W_MM - d) - 1, Y(CARD_H_MM - d) - 1],
            outline=color, width=line_w,
        )

    def label(text: str, x: float, y: float, color, anchor: str) -> None:
        bbox = draw.textbbox((x, y), text, font=font, anchor=anchor)
        pad = max(1, round(0.15 * sx))
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill="white", outline=color, width=1,
        )
        draw.text((x, y), text, font=font, fill=color, anchor=anchor)

    # Labels staggered along each side so they do not overlap.  Each label
    # touches its own line from the inside.
    n = len(values)
    gap = 0.35
    for i, d in enumerate(values):
        color = COLORS[i % len(COLORS)]
        text = f"{d:g}"
        # left / right: spread along the height between 8 and 46 mm
        ly = 8.0 + (CARD_H_MM - 16.0) * (i + 0.5) / n
        label(text, X(d + gap), Y(ly), color, "lm")
        label(text, X(CARD_W_MM - d - gap), Y(CARD_H_MM - ly), color, "rm")
        # top / bottom: spread along the width between 12 and 74 mm
        lx = 12.0 + (CARD_W_MM - 24.0) * (i + 0.5) / n
        label(text, X(lx), Y(d + gap), color, "mt")
        label(text, X(CARD_W_MM - lx), Y(CARD_H_MM - d - gap), color, "mb")

    # Centre: side names and a 20 mm scale bar (checks driver scaling).
    big = _font(max(10, round(3.2 * sy)))
    cx, cy = CARD_W_MM / 2, CARD_H_MM / 2
    draw.text((X(cx), Y(cy - 9)), "TOP", font=big, fill="black", anchor="mm")
    draw.text((X(cx), Y(cy + 9)), "BOTTOM", font=big, fill="black", anchor="mm")
    draw.text((X(12), Y(cy)), "LEFT", font=big, fill="black", anchor="lm")
    draw.text((X(CARD_W_MM - 12), Y(cy)), "RIGHT", font=big, fill="black", anchor="rm")
    bar_y = Y(cy)
    draw.line([X(cx - 10), bar_y, X(cx + 10), bar_y], fill="black", width=line_w * 2)
    for t in (-10, 0, 10):
        draw.line([X(cx + t), bar_y - Y(1), X(cx + t), bar_y + Y(1)],
                  fill="black", width=line_w * 2)
    draw.text((X(cx), bar_y + Y(1.6)), "20 mm", font=font, fill="black", anchor="mt")
    return img


if __name__ == "__main__":  # quick preview: python -m idcard_cropper.calibration out.png
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "calibration_card.png")
    render_calibration_card(1016, 638).save(out)
    print(out)
