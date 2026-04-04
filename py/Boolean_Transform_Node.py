class BooleanTransformNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"input_string": ("STRING", {"multiline": False, "default": "0.0"})}}

    RETURN_TYPES = ("INT", "BOOLEAN")
    FUNCTION = "transform_to_boolean"
    CATEGORY = "CRT/Utils/Logic & Values"

    def transform_to_boolean(self, input_string):
        if isinstance(input_string, list) and input_string:
            input_string = input_string[0]

        if not isinstance(input_string, str):
            input_string = str(input_string)

        try:
            num = float(input_string)
        except ValueError:
            num = 0.0

        boolean_value = num != 0
        int_value = 1 if boolean_value else 0

        return (int_value, boolean_value)
