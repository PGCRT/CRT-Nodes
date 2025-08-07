import torch

class PrismFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "k1": ("FLOAT", {"default": 0.016, "min": -0.1, "max": 0.1, "step": 0.001, "display": "slider"}),
                "k2": ("FLOAT", {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001, "display": "slider"}),
                "k3": ("FLOAT", {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001, "display": "slider"}),
                "k4": ("FLOAT", {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001, "display": "slider"}),
                "achromat_amount": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "samples_limit": ("INT", {"default": 32, "min": 6, "max": 128, "step": 2, "display": "slider"}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_prism_effect"
    CATEGORY = "CRT/FX"

    def _spectrum(self, hue):
        hue = hue * 4.0
        r = torch.clamp(1.5 - torch.abs(hue - 1.0), 0.0, 1.0)
        g = torch.clamp(1.5 - torch.abs(hue - 2.0), 0.0, 1.0)
        r += torch.clamp(hue - 3.5, 0.0, 1.0)
        b = 1.0 - r
        return torch.stack([r, g, b], dim=-1)

    def apply_prism_effect(self, image: torch.Tensor, k1, k2, k3, k4, achromat_amount, samples_limit, strength):
        device = image.device
        batch_size, height, width, _ = image.shape
        result_images = []

        k_coeffs = torch.tensor([k1, k2, k3, k4], device=device)

        for i in range(batch_size):
            img_tensor_bchw = image[i:i+1].permute(0, 3, 1, 2)
            
            y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device), torch.linspace(-1, 1, width, device=device), indexing='ij')
            
            view_coord = torch.stack([x, y], dim=-1)
            
            aspect_ratio = width / height
            if aspect_ratio > 1.0:
                view_coord[..., 0] *= aspect_ratio
            else:
                view_coord[..., 1] /= aspect_ratio

            radius_sq = torch.sum(view_coord**2, dim=-1)
            pow_radius = torch.stack([radius_sq, radius_sq**2, radius_sq**3, radius_sq**4], dim=-1)
            
            radial_distortion = torch.sum(k_coeffs * pow_radius, dim=-1, keepdim=True)
            distortion_vec = view_coord * radial_distortion
            
            base_grid = torch.stack((x,y), dim=-1).unsqueeze(0)
            
            accumulated_color = torch.zeros_like(img_tensor_bchw)
            
            samples = max(6, samples_limit - (samples_limit % 2))

            for j in range(samples):
                hue = j / float(samples - 1)
                progress = hue - 0.5
                progress = torch.lerp(torch.tensor(progress), 0.5 - abs(progress), achromat_amount)
                
                offset = progress * distortion_vec * 2.0
                current_grid = base_grid + offset.unsqueeze(0)
                
                sampled_color = torch.nn.functional.grid_sample(img_tensor_bchw, current_grid, mode='bilinear', padding_mode='border', align_corners=False)
                
                spectrum_color = self._spectrum(hue).view(1, 3, 1, 1)
                
                accumulated_color += sampled_color * spectrum_color

            final_color_bchw = accumulated_color * (2.0 / samples)
            
            final_color_bhwc = final_color_bchw.permute(0, 2, 3, 1)
            
            original_image_bhwc = image[i:i+1]
            
            final_image = torch.lerp(original_image_bhwc, final_color_bhwc, strength)
            
            result_images.append(final_image.clamp(0.0, 1.0))
            
        return (torch.cat(result_images, dim=0),)