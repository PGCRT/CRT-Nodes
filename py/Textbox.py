class CRT_Textbox:
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
                        "tooltip": "Displays and passes through the connected string. When passthrough is not connected, this editable text is returned.",
                    },
                ),
            },
            "optional": {
                "passthrough": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "forceInput": True,
                        "tooltip": "Optional connected string to display and return unchanged.",
                    },
                )
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    OUTPUT_NODE = True
    FUNCTION = "textbox"
    CATEGORY = "CRT/Text"

    def textbox(self, text="", **kwargs):
        # Optional forceInput values are omitted when disconnected, so checking
        # the key preserves a deliberately connected empty string.
        value = kwargs["passthrough"] if "passthrough" in kwargs else text
        value = "" if value is None else str(value)
        return {"ui": {"text": [value]}, "result": (value,)}


NODE_CLASS_MAPPINGS = {"CRT_Textbox": CRT_Textbox}

NODE_DISPLAY_NAME_MAPPINGS = {"CRT_Textbox": "Textbox (CRT)"}
