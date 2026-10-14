import math
import os

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor

# --- Font discovery (relative to CRT-Nodes/Fonts) ---
NODE_FILE_PATH = os.path.abspath(__file__)
CRT_NODE_DIR = os.path.dirname(NODE_FILE_PATH)
FONTS_DIR = os.path.join(os.path.dirname(CRT_NODE_DIR), "Fonts")

FONT_LIST = ["Default"]
if not os.path.exists(FONTS_DIR):
    try:
        os.makedirs(FONTS_DIR)
    except Exception:
        pass
try:
    font_files = sorted(
        f for f in os.listdir(FONTS_DIR)
        if f.lower().endswith((".ttf", ".otf", ".ttc"))
    )
    if font_files:
        FONT_LIST.extend(font_files)
except Exception as e:
    print(f"[CRT Image Label] Error scanning font directory: {e}")

# Aspect ratios: Resolution (CRT) list + extra extreme panoramic / tower ratios.
ASPECT_RATIOS = [
    "1:1 (Square)",
    "1:2 (Portrait)",
    "1:3 (Portrait Tower)",
    "1:4 (Portrait Strip)",
    "2:3 (Portrait)",
    "3:4 (Portrait)",
    "4:5 (Portrait)",
    "5:7 (Portrait)",
    "5:8 (Portrait)",
    "7:9 (Portrait)",
    "9:16 (Portrait)",
    "9:19 (Portrait)",
    "9:21 (Portrait)",
    "9:32 (Portrait Ultrawide)",
    "2:1 (Landscape)",
    "3:1 (Landscape Panorama)",
    "3:2 (Landscape)",
    "4:1 (Landscape Strip)",
    "4:3 (Landscape)",
    "5:3 (Landscape)",
    "5:4 (Landscape)",
    "7:5 (Landscape)",
    "8:5 (Landscape)",
    "9:7 (Landscape)",
    "16:9 (Landscape)",
    "19:9 (Landscape)",
    "21:9 (Landscape)",
    "32:9 (Landscape Super Ultrawide)",
]

MIN_FONT_PX = 12


def _parse_color(value, fallback):
    """Accept '#RRGGBB', '#RGB', named colors; return (R, G, B)."""
    if isinstance(value, (tuple, list)) and len(value) >= 3:
        return (int(value[0]), int(value[1]), int(value[2]))
    try:
        return ImageColor.getrgb(str(value).strip())
    except Exception:
        return fallback


def _load_font(font_name, size):
    size = max(8, int(size))
    if font_name != "Default":
        try:
            return ImageFont.truetype(os.path.join(FONTS_DIR, font_name), size)
        except Exception as e:
            print(f"[CRT Image Label] Failed to load font {font_name}: {e}. Falling back.")
    # Preferred fallbacks for a label look
    for fallback in ("arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(fallback, size)
        except Exception:
            continue
    return ImageFont.load_default()


def _wrap_paragraph(words, font, draw, max_width):
    """Greedy word wrap for one paragraph. Words are never split across lines;
    an overlong word keeps its own line whole (the fitter then shrinks the
    font until it fits)."""
    lines = []
    current = ""
    for word in words:
        trial = word if not current else current + " " + word
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _wrap_text(text, font, draw, max_width):
    lines = []
    for para in str(text).split("\n"):
        if not para.strip():
            lines.append("")
            continue
        lines.extend(_wrap_paragraph(para.split(), font, draw, max_width) or [""])
    return lines


def _measure(lines, font, draw, spacing):
    """Return (block_w, block_h, line_heights, ascents)."""
    widths = []
    heights = []
    ascents = []
    for line in lines:
        if not line:
            # Blank line still advances
            bbox = font.getbbox("A") if hasattr(font, "getbbox") else (0, 0, 0, 10)
            h = max(1, bbox[3] - bbox[1])
            widths.append(0)
            heights.append(h)
            ascents.append(0)
            continue
        bbox = draw.textbbox((0, 0), line, font=font, stroke_width=0)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
        ascents.append(bbox[1])  # top offset (usually negative)
    block_w = max(widths) if widths else 0
    block_h = sum(heights) + spacing * max(0, len(lines) - 1)
    return block_w, block_h, heights, ascents


def _fit_font_size(text, font_name, safe_w, safe_h):
    """Binary-search the largest font size whose wrapped text fits the safe area.
    Words are never split, so the search shrinks until whole words fit."""
    probe = Image.new("RGB", (8, 8))
    draw = ImageDraw.Draw(probe)
    safe_w = max(8, int(safe_w))
    safe_h = max(8, int(safe_h))

    hi = int(min(safe_h, safe_w * 2))
    hi = max(MIN_FONT_PX + 1, min(hi, 1024))
    lo = MIN_FONT_PX
    best = None

    def fits(size):
        font = _load_font(font_name, size)
        spacing = max(2, int(size * 0.25))
        lines = _wrap_text(text, font, draw, safe_w)
        bw, bh, _, _ = _measure(lines, font, draw, spacing)
        return (bw <= safe_w and bh <= safe_h), lines, spacing

    # Quick check: does the minimum fit at all?
    ok_min, lines_min, sp_min = fits(lo)
    if not ok_min:
        return lo, lines_min, sp_min, False

    while lo <= hi:
        mid = (lo + hi) // 2
        ok, lines, sp = fits(mid)
        if ok:
            best = (mid, lines, sp)
            lo = mid + 1
        else:
            hi = mid - 1
    if best is None:
        return lo, lines_min, sp_min, False
    size, lines, sp = best
    return size, lines, sp, True


def _shrink_line_to_width(line, font, draw, safe_w):
    """Shorten one overwide line with an ellipsis, dropping whole words first.
    Words are never split, except a single word wider than the whole canvas,
    which is hard-cut with an ellipsis as a last resort against clipping."""
    if draw.textlength(line, font=font) <= safe_w:
        return line
    words = line.split()
    while words and draw.textlength(" ".join(words) + " …", font=font) > safe_w:
        words.pop()
    if words:
        return " ".join(words) + " …"
    word = line
    while word and draw.textlength(word + "…", font=font) > safe_w:
        word = word[:-1]
    return (word + "…") if word else "…"


def _truncate_to_fit(lines, font, draw, spacing, safe_w, safe_h):
    """Fit lines into the safe area without clipping: overwide lines lose
    whole words (ellipsis), then vertically overflowing lines are dropped."""
    fitted = [_shrink_line_to_width(l, font, draw, safe_w) if l else l for l in lines]
    kept = []
    used = 0
    dropped = False
    for line in fitted:
        bbox = font.getbbox("A") if not line else draw.textbbox((0, 0), line, font=font)
        lh = max(1, bbox[3] - bbox[1])
        need = lh + (spacing if kept else 0)
        if used + need <= safe_h:
            kept.append(line)
            used += need
        else:
            dropped = True
            break
    # If lines were dropped with no visible mark of the cut, mark it.
    if dropped and kept and not kept[-1].endswith("…"):
        kept[-1] = _shrink_line_to_width(kept[-1] + " …", font, draw, safe_w)
    return kept if kept else (["…"] if lines else [])


class CRT_ImageLabel:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "aspect_ratio": (ASPECT_RATIOS, {"default": "1:1 (Square)"}),
                "megapixels": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.1, "max": 32.0, "step": 0.1},
                ),
                "text_color": ("COLOR", {"default": "#FFFFFF"}),
                "background_color": ("COLOR", {"default": "#000000"}),
                "font": (FONT_LIST,),
                "relative_size": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 3.0,
                        "step": 0.05,
                        "tooltip": "Multiplier on the auto-fitted font size. 1.0 = auto fit.",
                    },
                ),
                "enable_glow": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "forceInput": True,
                        "default": "",
                        "tooltip": "Text to render as a centered label.",
                    },
                ),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "generate"
    CATEGORY = "CRT/Text"
    DESCRIPTION = (
        "Renders the input string as a centered label on a solid canvas. "
        "Canvas size comes from aspect ratio + megapixels; the font size and "
        "word-wrapping are auto-optimized so text never clips and never sits "
        "tiny on the canvas. Words are never split across lines. "
        "relative_size scales the auto-fitted size (1.0 = auto fit). "
        "Fonts load from CRT-Nodes/Fonts."
    )

    def generate(self, aspect_ratio, megapixels, text_color, background_color,
                 font, enable_glow, text="", relative_size=1.0):
        text = "" if text is None else str(text)
        if not text.strip():
            text = "Label"

        # --- Canvas size from aspect ratio + megapixels ---
        ratio_str = str(aspect_ratio).split(" ")[0]
        try:
            wr, hr = map(int, ratio_str.split(":"))
        except ValueError:
            wr, hr = 1, 1
        ratio = wr / max(hr, 1)
        total_pixels = max(1, int(float(megapixels) * 1_000_000))
        w = math.sqrt(total_pixels * ratio)
        h = math.sqrt(total_pixels / ratio)
        # Snap to 8px grid for VAE-friendly sizes
        width = max(64, int(round(w / 8) * 8))
        height = max(64, int(round(h / 8) * 8))

        bg = _parse_color(background_color, (0, 0, 0))
        fg = _parse_color(text_color, (255, 255, 255))

        # Safe area: 6% padding each side keeps text off the edges
        # on both balanced and extreme canvases.
        pad_x = max(8, int(width * 0.06))
        pad_y = max(8, int(height * 0.06))
        safe_w = max(16, width - 2 * pad_x)
        safe_h = max(16, height - 2 * pad_y)

        auto_size, _, _, _ = _fit_font_size(text, font, safe_w, safe_h)
        try:
            scale = float(relative_size)
        except (TypeError, ValueError):
            scale = 1.0
        size = max(8, int(round(auto_size * scale)))
        pil_font = _load_font(font, size)
        spacing = max(2, int(size * 0.25))

        probe = Image.new("RGB", (8, 8))
        measure_draw = ImageDraw.Draw(probe)
        # Re-wrap at the final size (wrapping depends on font size), then
        # guarantee the no-clip contract even when scaled above the auto fit.
        lines = _wrap_text(text, pil_font, measure_draw, safe_w)
        block_w, block_h, _, _ = _measure(lines, pil_font, measure_draw, spacing)
        if block_w > safe_w or block_h > safe_h:
            lines = _truncate_to_fit(lines, pil_font, measure_draw, spacing, safe_w, safe_h)
        block_w, block_h, line_heights, _ = _measure(lines, pil_font, measure_draw, spacing)

        base = Image.new("RGB", (width, height), bg)
        # Center the text block on the canvas.
        x0 = (width - block_w) // 2
        y0 = (height - block_h) // 2

        def draw_lines(target, fill, stroke_width=0, stroke_fill=None):
            d = ImageDraw.Draw(target)
            y = y0
            for i, line in enumerate(lines):
                if not line:
                    y += line_heights[i] + spacing
                    continue
                bbox = d.textbbox((0, 0), line, font=pil_font, stroke_width=stroke_width)
                lw = bbox[2] - bbox[0]
                # Compensate the bbox left/top bearing so glyphs are truly centered.
                x = x0 + (block_w - lw) // 2 - bbox[0]
                ty = y - bbox[1]
                d.text((x, ty), line, font=pil_font, fill=fill,
                       stroke_width=stroke_width, stroke_fill=stroke_fill)
                y += line_heights[i] + spacing

        if enable_glow:
            # Soft glow: blurred copy of the text in the text color under the crisp text.
            glow_layer = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            draw_lines(glow_layer, fg + (255,))
            radius = max(2, size // 12)
            glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius))
            base = Image.alpha_composite(base.convert("RGBA"), glow_layer).convert("RGB")

        draw_lines(base, fg)

        arr = np.array(base).astype(np.float32) / 255.0
        return (torch.from_numpy(arr).unsqueeze(0),)


NODE_CLASS_MAPPINGS = {
    "CRT_ImageLabel": CRT_ImageLabel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_ImageLabel": "Image Label (CRT)",
}
