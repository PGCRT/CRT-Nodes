import torch
import math

class BatchBrightnessCurve:
    """
    Adjusts the brightness of an image batch using a continuous gradient curve.
    
    It interpolates between:
    - Start Level (Frame 0)
    - Mid Level (Center of batch)
    - End Level (Last Frame)
    
    The Q-Factor controls how "fat" or "sharp" the curve is.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "start_level": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 5.0, 
                    "step": 0.01,
                    "tooltip": "Brightness at the very first frame."
                }),
                "mid_level": ("FLOAT", {
                    "default": 1.0, 
                    "min": 0.0, 
                    "max": 5.0, 
                    "step": 0.01,
                    "tooltip": "Brightness at the exact center of the video."
                }),
                "end_level": ("FLOAT", {
                    "default": 0.0, 
                    "min": 0.0, 
                    "max": 5.0, 
                    "step": 0.01,
                    "tooltip": "Brightness at the very last frame."
                }),
                "q_factor": ("FLOAT", {
                    "default": 2.0, 
                    "min": 0.1, 
                    "max": 10.0, 
                    "step": 0.1,
                    "tooltip": "Curve Shape. 1.0=Linear, 2.0=Parabolic (Smooth). Higher values = Stays closer to Mid Level longer."
                }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_curve"
    CATEGORY = "CRT/Image"

    def apply_curve(self, images, start_level, mid_level, end_level, q_factor):
        batch_size = images.shape[0]
        
        if batch_size == 0:
            return (images,)

        # Prepare a list of multipliers
        multipliers = []
        
        for i in range(batch_size):
            # Calculate normalized position t (0.0 to 1.0)
            if batch_size <= 1:
                t = 0.5 # Single image is considered center
            else:
                t = i / (batch_size - 1)

            # Determine which "Edge" we are closer to (Start or End)
            if t <= 0.5:
                # Left side: Interpolate between start_level and mid_level
                edge_val = start_level
            else:
                # Right side: Interpolate between end_level and mid_level
                edge_val = end_level

            # Calculate Distance from Center (0.0 = Center, 1.0 = Edge)
            # t=0.0 -> dist=1.0 | t=0.5 -> dist=0.0 | t=1.0 -> dist=1.0
            dist_from_center = abs(t - 0.5) * 2.0

            # Apply Q-Factor (The Curve)
            # If Q=2.0 (Standard), the influence of the edge drops quadratically
            weight = math.pow(dist_from_center, q_factor)

            # Interpolate
            # Weight 1.0 (Edge) -> uses edge_val
            # Weight 0.0 (Center) -> uses mid_level
            val = edge_val * weight + mid_level * (1.0 - weight)

            multipliers.append(val)

        # Convert list to tensor [Batch, 1, 1, 1] for broadcasting
        mult_tensor = torch.tensor(multipliers, device=images.device, dtype=images.dtype)
        mult_tensor = mult_tensor.view(-1, 1, 1, 1)

        # Apply brightness
        output_images = torch.clamp(images * mult_tensor, min=0.0)

        return (output_images,)

NODE_CLASS_MAPPINGS = {
    "BatchBrightnessCurve": BatchBrightnessCurve
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BatchBrightnessCurve": "Batch Brightness Curve (CRT)"
}