class CRT_RemoveLines:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": False, "default": ""}),
                "early_lines": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "last_lines": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1}),
                "remove_empty_lines": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "remove_lines"
    CATEGORY = "CRT/Text"

    def remove_lines(self, text, early_lines, last_lines, remove_empty_lines):
        if not text:
            return ("",)

        lines = text.splitlines()

        if remove_empty_lines:
            lines = [line for line in lines if line.strip()]

        if early_lines > 0:
            lines = lines[early_lines:]

        if last_lines > 0 and len(lines) > 0:
            lines = lines[:-last_lines] if last_lines < len(lines) else []

        result = "\n".join(lines)
        return (result,)
