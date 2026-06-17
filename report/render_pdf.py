"""Render report.md to a simple image-based PDF.

This avoids LaTeX, LibreOffice, and PDF font-embedding issues on headless
servers. The output is not typographically fancy, but it is readable and keeps
Korean text intact by rasterizing each page before saving as PDF.
"""

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parent
REPORT_MD = ROOT / "report.md"
REPORT_PDF = ROOT / "report.pdf"
FONT_PATH = "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf"
MONO_FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"

PAGE_W = 1240
PAGE_H = 1754
MARGIN_X = 90
MARGIN_Y = 80
LINE_GAP = 9


def load_font(path, size):
    return ImageFont.truetype(path, size=size)


def iter_render_lines(text):
    in_code = False
    for raw in text.splitlines():
        line = raw.rstrip()
        if line.startswith("```"):
            in_code = not in_code
            yield "", "body"
            continue
        if in_code:
            for wrapped in textwrap.wrap(line, width=92, replace_whitespace=False) or [""]:
                yield wrapped, "code"
            continue
        if not line:
            yield "", "body"
            continue
        if line.startswith("# "):
            yield line[2:], "h1"
            continue
        if line.startswith("## "):
            yield line[3:], "h2"
            continue
        if line.startswith("### "):
            yield line[4:], "h3"
            continue
        if line.startswith("|"):
            for wrapped in textwrap.wrap(line, width=108, replace_whitespace=False) or [""]:
                yield wrapped, "code"
            continue
        for wrapped in textwrap.wrap(line, width=70, replace_whitespace=False) or [""]:
            yield wrapped, "body"


def style(kind):
    if kind == "h1":
        return "body_bold", 34, 22
    if kind == "h2":
        return "body_bold", 26, 18
    if kind == "h3":
        return "body_bold", 22, 14
    if kind == "code":
        return "mono", 15, 6
    return "body", 19, LINE_GAP


def main():
    fonts = {
        "body": load_font(FONT_PATH, 19),
        "body_bold": load_font(FONT_PATH, 24),
        "mono": load_font(MONO_FONT_PATH, 15),
    }
    pages = []
    page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
    draw = ImageDraw.Draw(page)
    y = MARGIN_Y

    def new_page():
        nonlocal page, draw, y
        pages.append(page)
        page = Image.new("RGB", (PAGE_W, PAGE_H), "white")
        draw = ImageDraw.Draw(page)
        y = MARGIN_Y

    for line, kind in iter_render_lines(REPORT_MD.read_text(encoding="utf-8")):
        font_key, font_size, extra_gap = style(kind)
        font = fonts[font_key]
        line_h = font_size + extra_gap
        if y + line_h > PAGE_H - MARGIN_Y:
            new_page()
        if line:
            draw.text((MARGIN_X, y), line, fill="black", font=font)
        y += line_h

    pages.append(page)
    first, rest = pages[0], pages[1:]
    first.save(REPORT_PDF, "PDF", resolution=150.0, save_all=True, append_images=rest)
    print(f"Wrote {REPORT_PDF}")


if __name__ == "__main__":
    main()

