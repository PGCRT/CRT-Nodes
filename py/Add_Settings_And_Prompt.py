import torch
import numpy as np
import os
from PIL import Image, ImageDraw, ImageFont

# --- Helper Function for Font Discovery ---
# This part of the code runs once when the node is loaded.

# Get the directory of the current script
NODE_FILE_PATH = os.path.abspath(__file__)
CRT_NODE_DIR = os.path.dirname(NODE_FILE_PATH)

# Define the path to the custom fonts directory, relative to this script
FONTS_DIR = os.path.join(os.path.dirname(CRT_NODE_DIR), "Fonts")
FONT_LIST = ["Default"]

# Ensure the Fonts directory exists
if not os.path.exists(FONTS_DIR):
    print(f"[CRT Node] Font directory not found at {FONTS_DIR}, creating it.")
    os.makedirs(FONTS_DIR)

# Scan the directory for valid font files (.ttf, .otf)
try:
    font_files = [f for f in os.listdir(FONTS_DIR) if f.lower().endswith(('.tt', '.otf'))]
    if font_files:
        FONT_LIST.extend(font_files)
    else:
        print(f"[CRT Node] No fonts found in {FONTS_DIR}. Only the default font will be available.")
except Exception as e:
    print(f"[CRT Node] Error scanning font directory: {e}")

# --- Main Node Class ---


class CRT_AddSettingsAndPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE",),
                "settings": ("STRING", {"multiline": True, "default": "Settings..."}),
                "prompt": ("STRING", {"multiline": True, "default": "A beautiful landscape..."}),
                "orientation": (["horizontal", "vertical"],),
                "font": (FONT_LIST,),
                "font_size": ("INT", {"default": 24, "min": 8, "max": 256, "step": 1}),
                "font_color": ("STRING", {"default": "white"}),
                "background_color": ("STRING", {"default": "black"}),
                "margin": ("INT", {"default": 15, "min": 0, "max": 256, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "add_text_block"
    CATEGORY = "CRT/Text"
    DESCRIPTION = """
Adds a text block with settings and a prompt to an image,
either horizontally at the bottom or vertically on the right.
The block's size is dynamically adjusted to fit the text.
Fonts are loaded from: ComfyUI/custom_nodes/CRT-Nodes/Fonts
"""

    def _get_font(self, font_name, font_size):
        """Loads a font, falling back to a default if necessary."""
        if font_name == "Default":
            try:
                return ImageFont.truetype("arial.ttf", font_size)
            except IOError:
                return ImageFont.load_default()

        font_path = os.path.join(FONTS_DIR, font_name)
        try:
            return ImageFont.truetype(font_path, font_size)
        except Exception as e:
            print(f"[CRT Node] Failed to load font {font_name}: {e}. Falling back to default.")
            return ImageFont.load_default()

    def _wrap_text(self, text, font, max_width):
        """Wraps text to fit within a specified width."""
        lines = []
        paragraphs = text.split('\n')

        for para in paragraphs:
            if not para:
                lines.append('')
                continue

            words = para.split(' ')
            current_line = ''
            for word in words:
                if font.getbbox(current_line + word)[2] <= max_width:
                    current_line += word + ' '
                else:
                    lines.append(current_line.rstrip())
                    current_line = word + ' '
            lines.append(current_line.rstrip())

        return lines

    def add_text_block(
        self, image, settings, prompt, orientation, font, font_size, font_color, background_color, margin
    ):
        # Prepare inputs and dimensions
        batch_size, original_height, original_width, _ = image.shape
        full_text = f"{settings.strip()}\n\nPrompt: {prompt.strip()}"

        # Load font and calculate line height
        pil_font = self._get_font(font, font_size)
        line_height = pil_font.getbbox("A")[3] - pil_font.getbbox("A")[1]
        line_spacing_factor = 1.4
        y_step = int(line_height * line_spacing_factor)

        # --- Logic for Horizontal Orientation ---
        if orientation == "horizontal":
            text_area_width = original_width - (2 * margin)
            wrapped_lines = self._wrap_text(full_text, pil_font, text_area_width)

            total_text_height = len(wrapped_lines) * y_step
            label_height = total_text_height + (2 * margin)

            label_pil = Image.new("RGB", (original_width, label_height), background_color)
            draw = ImageDraw.Draw(label_pil)

            y_pos = margin
            for line in wrapped_lines:
                draw.text((margin, y_pos), line, font=pil_font, fill=font_color)
                y_pos += y_step

            label_np = np.array(label_pil).astype(np.float32) / 255.0
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)
            label_tensor_batch = label_tensor.expand(batch_size, -1, -1, -1).to(image.device)

            combined_image = torch.cat((image, label_tensor_batch), dim=1)

        # --- Logic for Vertical Orientation ---
        else:  # "vertical"
            max_text_height = original_height - (2 * margin)

            # Iteratively find the necessary width to fit the text vertically
            text_area_width = 256  # Start with a minimum width
            while True:
                wrapped_lines = self._wrap_text(full_text, pil_font, text_area_width)
                total_text_height = len(wrapped_lines) * y_step
                if total_text_height <= max_text_height:
                    break
                text_area_width += 64  # Increase width and try again

            # Calculate the final block width based on the longest line of text
            max_line_width = 0
            for line in wrapped_lines:
                line_width = pil_font.getbbox(line)[2]
                if line_width > max_line_width:
                    max_line_width = line_width

            label_width = int(max_line_width) + (2 * margin)

            label_pil = Image.new("RGB", (label_width, original_height), background_color)
            draw = ImageDraw.Draw(label_pil)

            y_pos = margin
            for line in wrapped_lines:
                draw.text((margin, y_pos), line, font=pil_font, fill=font_color)
                y_pos += y_step

            label_np = np.array(label_pil).astype(np.float32) / 255.0
            label_tensor = torch.from_numpy(label_np).unsqueeze(0)
            label_tensor_batch = label_tensor.expand(batch_size, -1, -1, -1).to(image.device)

            # Concatenate along the width dimension (dim=2)
            combined_image = torch.cat((image, label_tensor_batch), dim=2)

        return (combined_image,)
