class StrengthToStepsNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "desired_steps": ("INT", {"default": 6, "min": 1}),
                "strength_percent": ("FLOAT", {"default": 40.0, "min": 1.0, "max": 100.0, "step": 1.0}),
            }
        }

    RETURN_TYPES = ("INT", "INT")
    RETURN_NAMES = ("total_steps", "start_step")
    FUNCTION = "compute_steps"
    CATEGORY = "CRT/Logic"

    def compute_steps(self, desired_steps, strength_percent):
        # Ensure strength is not 0 to avoid division by zero
        strength_fraction = max(strength_percent, 1.0) / 100.0

        # Calculate total_steps
        total_steps = round(desired_steps / strength_fraction)

        # Calculate start_step
        start_step = total_steps - desired_steps

        return (total_steps, start_step)
