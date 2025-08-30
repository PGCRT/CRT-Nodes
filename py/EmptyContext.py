class EmptyContext:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {}}

    RETURN_TYPES = ("WANVIDCONTEXT",)
    RETURN_NAMES = ("context_options",)
    FUNCTION = "get_empty_context"
    CATEGORY = "CRT/Logic"

    def get_empty_context(self):
        return (None,)