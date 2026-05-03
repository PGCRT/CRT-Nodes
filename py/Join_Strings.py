class CRT_JoinStrings:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "string_a": ("STRING", {"default": "", "multiline": True}),
                "string_b": ("STRING", {"default": "", "multiline": True}),
                "separator": ("STRING", {"default": " "}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "join_strings"
    CATEGORY = "CRT/Text"

    def join_strings(self, string_a, string_b, separator):
        return (f"{string_a}{separator}{string_b}",)
