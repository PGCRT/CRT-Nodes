class CRT_Textbox:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": False,
                        "print_to_screen": True,
                    },
                ),
            },
            "optional": {
                "passthrough": (
                    "STRING",
                    {"default": "", "multiline": True, "forceInput": True},
                )
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "textbox"
    CATEGORY = "CRT/Text"

    def textbox(self, text="", passthrough=""):
        if passthrough != "":
            text = passthrough
            return {"ui": {"text": text}, "result": (text,)}
        return (text,)


NODE_CLASS_MAPPINGS = {"CRT_Textbox": CRT_Textbox}

NODE_DISPLAY_NAME_MAPPINGS = {"CRT_Textbox": "Textbox (CRT)"}
