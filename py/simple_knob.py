class SimpleKnobNode:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {
                    "default": 0,
                    "min": -10000.0,
                    "max": 10000.0,
                    "step": 0.01,
                    "widget": "simple_knob",
                }),
            }
        }

    RETURN_TYPES = ("FLOAT",)
    RETURN_NAMES = (" ",)
    FUNCTION = "get_value"
    CATEGORY = "CRT/Utils/UI"

    def get_value(self, value):
        return (value,)

NODE_CLASS_MAPPINGS = {
    "SimpleKnobNode": SimpleKnobNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SimpleKnobNode": "K"
}