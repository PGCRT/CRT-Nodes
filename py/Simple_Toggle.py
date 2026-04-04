class SimpleToggleNode:
    """
    A simple boolean toggle node that outputs True/False values.
    The frontend will replace the default widget with a custom toggle UI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = (" ",)
    FUNCTION = "get_value"
    CATEGORY = "CRT/Utils/UI"

    def get_value(self, value):
        return (value,)


# Export the node
NODE_CLASS_MAPPINGS = {"SimpleToggleNode": SimpleToggleNode}

NODE_DISPLAY_NAME_MAPPINGS = {"SimpleToggleNode": "Simple Toggle"}
