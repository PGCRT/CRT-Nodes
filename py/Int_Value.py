class CRT_IntValue:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("INT", {"default": 0, "min": -2147483648, "max": 2147483647, "step": 1}),
            }
        }

    RETURN_TYPES = ("INT",)
    FUNCTION = "get_value"
    CATEGORY = "CRT/Utils/LogicValues"

    def get_value(self, value):
        return (int(value),)


NODE_CLASS_MAPPINGS = {"CRT_IntValue": CRT_IntValue}

NODE_DISPLAY_NAME_MAPPINGS = {"CRT_IntValue": "Int Value (CRT)"}
