class BooleanInvert:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_boolean": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "CRT/Logic"
    RETURN_TYPES = ("BOOLEAN",)
    RETURN_NAMES = ("output_boolean",)
    FUNCTION = "invert_boolean"

    def invert_boolean(self, input_boolean):
        print(f"📌 BooleanInvert: Starting execution, input_boolean={input_boolean}")
        output = not input_boolean
        print(f"📌 Output: {output}")
        return (output,)

    @classmethod
    def IS_CHANGED(cls, input_boolean, **kwargs):
        # Return the input boolean to trigger re-execution if it changes
        return input_boolean

    @classmethod
    def VALIDATE_INPUTS(cls, input_boolean, **kwargs):
        # Boolean input is always valid
        return True


NODE_CLASS_MAPPINGS = {"BooleanInvert": BooleanInvert}

NODE_DISPLAY_NAME_MAPPINGS = {"BooleanInvert": "Boolean Invert (CRT)"}
