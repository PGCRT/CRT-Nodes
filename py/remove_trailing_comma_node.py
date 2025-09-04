class RemoveTrailingCommaNode:
    @staticmethod
    def INPUT_TYPES():
        return {
            "required": {
                "input_string": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "remove_trailing_comma"
    CATEGORY = "CRT/Text"

    def remove_trailing_comma(self, input_string):
        if input_string.endswith(","):
            trimmed_string = input_string[:-1]
        else:
            trimmed_string = input_string
        
        return (trimmed_string,)
