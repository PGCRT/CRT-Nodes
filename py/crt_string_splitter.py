import re
import os

class CRT_StringSplitter:
    """
    A node to split a single string into multiple strings based on either empty lines or a custom symbol.
    The number of outputs is dynamically controlled by the 'split_count' widget.
    """
    
    # Define the maximum number of outputs. This MUST match the "max" value in the split_count widget.
    MAX_SPLITS = 64

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": "Paragraph 1...\n\nParagraph 2..."}),
                "split_count": ("INT", {"default": 2, "min": 1, "max": cls.MAX_SPLITS, "step": 1}),
                "split_symbol": ("STRING", {"multiline": False, "default": "", "tooltip": "A custom character or string to split by. If empty, splits by empty lines."}),
                "strip_whitespace": ("BOOLEAN", {"default": True, "label_on": "Yes", "label_off": "No", "tooltip": "Remove leading/trailing whitespace from each split part"}),
            }
        }

    RETURN_TYPES = ("STRING",) * MAX_SPLITS
    RETURN_NAMES = tuple(f"string_{i+1}" for i in range(MAX_SPLITS))
    
    FUNCTION = "split_string"
    CATEGORY = "CRT/Text"

    def split_string(self, text, split_count, strip_whitespace, split_symbol):
        node_name = "\033[96m[CRT String Splitter]\033[0m"
        print(f"{node_name} Splitting text into {split_count} active parts.")

        parts = []
        if text and text.strip():
            cleaned_symbol = split_symbol.strip()
            if cleaned_symbol:
                print(f"{node_name} Using custom symbol for splitting: '{cleaned_symbol}'")
                parts = text.split(cleaned_symbol)
            else:
                print(f"{node_name} Using empty lines for splitting.")
                parts = re.split(r'\n\s*\n', text)
            
            if strip_whitespace:
                parts = [p.strip() for p in parts]
            
            parts = [p for p in parts if p]
            print(f"{node_name} Found {len(parts)} non-empty parts.")
        else:
            print(f"\033[93m{node_name} ⚠ Warning: Input text is empty.\033[0m")

        active_outputs = []
        for i in range(split_count):
            if i < len(parts):
                active_outputs.append(parts[i])
            else:
                active_outputs.append("")
        
        padded_outputs = active_outputs + [""] * (self.MAX_SPLITS - len(active_outputs))
        
        print(f"\032[92m{node_name} ✓ Complete - Returning tuple of size {len(padded_outputs)} ({split_count} active).\033[0m")
        
        return tuple(padded_outputs)