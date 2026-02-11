class CRT_LineSpot:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
            "optional": {
                "passthrough": ("STRING", {"forceInput": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "execute"
    CATEGORY = "CRT/Text"

    def execute(self, text="", passthrough=""):
        # Use passthrough if connected, otherwise use the internal text
        display_text = passthrough if passthrough != "" else text
        return {"ui": {"text": [display_text]}, "result": (display_text,)}
