from comfy import ui
class FancyNoteNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "ui_font_size": ("INT", {"default": 16, "min": 10, "max": 30, "step": 1}),
                "ui_text_color": ("STRING", {"default": "#D8BFD8"}),
                "ui_glow_color": ("STRING", {"default": "#D8BFD8"}),
                "ui_accent_color": ("STRING", {"default": "#E6E6FA"}),
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT"
            }
        }
    
    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "CRT"
    
    def execute(self, text, ui_font_size, ui_text_color, ui_glow_color, ui_accent_color, **kwargs):
        return {}