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
    font_files = [f for f in os.listdir(FONTS_DIR) if f.lower().endswith(('.ttf', '.otf'))]
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
                "font": (FONT_LIST, ),
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
Adds a text block with settings and a prompt to the
bottom of an image. The block's height is dynamically
adjusted to fit the text content without clipping.
Fonts are loaded from: ComfyUI/custom_nodes/CRT-Nodes/Fonts
"""

    def _get_font(self, font_name, font_size):
        """Loads a font, falling back to a default if necessary."""
        if font_name == "Default":
            # In a portable install, fonts might not be in system paths.
            # We bundle a failsafe font or use Pillow's default.
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
        # Split text into paragraphs to preserve user-made newlines
        paragraphs = text.split('\n')
        
        for para in paragraphs:
            if not para:  # Handle empty lines
                lines.append('')
                continue
                
            words = para.split(' ')
            current_line = ''
            for word in words:
                # Check width of the current line with the new word
                if font.getbbox(current_line + word)[2] <= max_width:
                    current_line += word + ' '
                else:
                    # If the line is full, push it and start a new one
                    lines.append(current_line.rstrip())
                    current_line = word + ' '
            lines.append(current_line.rstrip()) # Add the last line
            
        return lines

    def add_text_block(self, image, settings, prompt, font, font_size, font_color, background_color, margin):
        # Prepare inputs and dimensions
        batch_size, original_height, original_width, _ = image.shape
        full_text = f"{settings.strip()}\n\nPrompt: {prompt.strip()}"
        
        # Load font
        pil_font = self._get_font(font, font_size)
        
        # --- Dynamic Height Calculation ---
        # Calculate the maximum width available for text
        text_area_width = original_width - (2 * margin)
        
        # Wrap the text and get all lines
        wrapped_lines = self._wrap_text(full_text, pil_font, text_area_width)
        
        # Calculate the height needed for the text block
        line_height = pil_font.getbbox("A")[3] - pil_font.getbbox("A")[1] # Height of a capital letter
        line_spacing_factor = 1.4 # Multiplier for spacing between lines
        total_text_height = int(len(wrapped_lines) * line_height * line_spacing_factor)
        label_height = total_text_height + (2 * margin)

        # --- Image Creation and Drawing ---
        # Create the label image using PIL
        label_pil = Image.new("RGB", (original_width, label_height), background_color)
        draw = ImageDraw.Draw(label_pil)

        # Draw each line of text
        y_pos = margin
        for line in wrapped_lines:
            draw.text((margin, y_pos), line, font=pil_font, fill=font_color)
            y_pos += int(line_height * line_spacing_factor)

        # Convert the PIL image to a PyTorch tensor
        label_np = np.array(label_pil).astype(np.float32) / 255.0
        label_tensor = torch.from_numpy(label_np).unsqueeze(0)

        # Expand the single label tensor to match the batch size of the input images
        # This is more efficient than creating a label for each image individually
        label_tensor_batch = label_tensor.expand(batch_size, -1, -1, -1)
        
        # Move tensor to the same device as the input image tensor
        label_tensor_batch = label_tensor_batch.to(image.device)
        
        # Concatenate the original images with the new label block at the bottom
        # dim=1 corresponds to the height dimension (B, H, W, C)
        combined_image = torch.cat((image, label_tensor_batch), dim=1)
        
        return (combined_image,)