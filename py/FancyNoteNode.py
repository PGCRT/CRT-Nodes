class FancyNoteNode:
    def __init__(self):
        self.properties = {}

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "hidden": {
                "ui_font_size": ("INT", {"default": 80, "min": 8, "max": 250, "step": 1}),
                "ui_text_color": ("STRING", {"default": "#7300ff"}),
                "ui_glow_color": ("STRING", {"default": "#7300ff"}),
                "ui_accent_color": ("STRING", {"default": "#7300ff"}),
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    OUTPUT_NODE = True
    CATEGORY = "CRT/Utils/UI"

    def execute(self, text, **kwargs):
        return {}
