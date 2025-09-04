import torch
import torch.nn.functional as F

class SmartDeNoiseFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "sigma": ("FLOAT", {"default": 1.25, "min": 0.001, "max": 8.0, "step": 0.001, "display": "slider"}),
                "threshold": ("FLOAT", {"default": 0.05, "min": 0.001, "max": 0.25, "step": 0.001, "display": "slider"}),
                "radius_multiplier": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 3.0, "step": 0.001, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_smart_denoise"
    CATEGORY = "CRT/FX"

    def apply_smart_denoise(self, image: torch.Tensor, sigma, threshold, radius_multiplier):
        device = image.device
        batch_size, height, width, _ = image.shape
        result_images = []

        radius = int(round(radius_multiplier * sigma))
        
        inv_sigma_sqx2 = 0.5 / (sigma * sigma + 1e-6)
        inv_threshold_sqx2 = 0.5 / (threshold * threshold + 1e-6)

        for i in range(batch_size):
            img_tensor_bhwc = image[i:i+1]
            
            padded_img = F.pad(img_tensor_bhwc.permute(0, 3, 1, 2), (radius, radius, radius, radius), mode='replicate').permute(0, 2, 3, 1)

            a_buff = torch.zeros_like(img_tensor_bhwc)
            z_buff = torch.zeros_like(img_tensor_bhwc)

            for y_offset in range(-radius, radius + 1):
                for x_offset in range(-radius, radius + 1):
                    
                    dist_sq = float(x_offset**2 + y_offset**2)
                    if dist_sq > radius**2:
                        continue

                    y_slice = slice(radius + y_offset, radius + y_offset + height)
                    x_slice = slice(radius + x_offset, radius + x_offset + width)
                    walk_px = padded_img[:, y_slice, x_slice, :]
                    
                    dc_sq = torch.sum((walk_px - img_tensor_bhwc)**2, dim=-1, keepdim=True)
                    
                    spatial_weight_val = -dist_sq * inv_sigma_sqx2
                    spatial_weight = torch.exp(torch.tensor(spatial_weight_val, device=device))

                    color_weight = torch.exp(-dc_sq * inv_threshold_sqx2)
                    
                    weight = spatial_weight * color_weight
                    
                    a_buff += walk_px * weight
                    z_buff += weight

            final_image = a_buff / (z_buff + 1e-6)
            result_images.append(final_image.clamp(0.0, 1.0))
            
        return (torch.cat(result_images, dim=0),)