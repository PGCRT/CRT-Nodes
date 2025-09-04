import torch

class Technicolor2FX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "brightness": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.5, "step": 0.01, "display": "slider"}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.5, "step": 0.01, "display": "slider"}),
                "color_strength_r": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "color_strength_g": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "color_strength_b": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_technicolor"
    CATEGORY = "CRT/FX"

    def apply_technicolor(self, image: torch.Tensor, strength, brightness, saturation, color_strength_r, color_strength_g, color_strength_b):
        
        device = image.device
        batch_size, _, _, _ = image.shape
        result_images = []
        
        color_strength_tensor = torch.tensor([color_strength_r, color_strength_g, color_strength_b], device=device).view(1, 1, 1, 3)

        for i in range(batch_size):
            color = image[i:i+1].clamp(0.0, 1.0)

            inv_color = 1.0 - color
            
            target1_a = torch.cat((inv_color[..., 1:2], inv_color[..., 0:1], inv_color[..., 1:2]), dim=-1)
            target2_a = torch.cat((inv_color[..., 2:3], inv_color[..., 2:3], inv_color[..., 0:1]), dim=-1)
            
            processed_color = color * target1_a
            processed_color *= target2_a

            temp_cs = processed_color * color_strength_tensor
            temp_b = processed_color * brightness

            target1_b = torch.cat((temp_cs[..., 1:2], temp_cs[..., 0:1], temp_cs[..., 1:2]), dim=-1)
            target2_b = torch.cat((temp_cs[..., 2:3], temp_cs[..., 2:3], temp_cs[..., 0:1]), dim=-1)
            
            final_color = color - target1_b
            final_color += temp_b
            final_color = final_color - target2_b

            color = torch.lerp(color, final_color, strength)

            luma = torch.sum(color * 0.33333, dim=-1, keepdim=True)
            color = torch.lerp(luma, color, saturation)
            
            result_images.append(color)

        return (torch.cat(result_images, dim=0),)