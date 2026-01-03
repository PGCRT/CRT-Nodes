class CRT_StringLineCounter:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
            },
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "count_lines"
    CATEGORY = "CRT/utils"

    def count_lines(self, text):
        if not text:
            return (0,)
        
        # splitlines() handles \n, \r, and \r\n automatically
        lines = text.splitlines()
        
        # If the last character is a newline, splitlines() usually ignores it. 
        # If you want to count a trailing empty line as a line, use text.split('\n')
        line_count = len(lines)
        
        return (line_count,)

NODE_CLASS_MAPPINGS = {
    "CRT_StringLineCounter": CRT_StringLineCounter
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_StringLineCounter": "String Line Counter (CRT)"
}