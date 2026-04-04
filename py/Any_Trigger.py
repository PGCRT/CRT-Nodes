class AnyTrigger:
    """
    A node that outputs a boolean based on whether an input is connected.
    Returns True if any input is connected, False otherwise.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "any_input": ("*",),
            },
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("boolean",)
    FUNCTION = "check_connection"
    CATEGORY = "CRT/Logic"

    def check_connection(self, any_input=None):
        """
        Check if the optional input is connected.
        Returns True if connected, False if not.
        """
        is_connected = any_input is not None
        return (is_connected,)


NODE_CLASS_MAPPINGS = {
    "AnyTrigger": AnyTrigger,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnyTrigger": "Any Trigger (CRT)",
}
