import torch

class LensDistortFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "radial_distortion_1": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.2, "step": 0.001, "display": "slider"}),
                "radial_distortion_2": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.2, "step": 0.001, "display": "slider"}),
                "radial_distortion_3": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.2, "step": 0.001, "display": "slider"}),
                "radial_distortion_4": ("FLOAT", {"default": 0.0, "min": -0.2, "max": 0.2, "step": 0.001, "display": "slider"}),
                "anamorphic_squeeze": ("FLOAT", {"default": 1.0, "min": 1.0, "max": 2.0, "step": 0.05, "display": "slider"}),
                "chromatic_radius": ("FLOAT", {"default": -0.2, "min": -0.2, "max": 0.2, "step": 0.001, "display": "slider"}),
                "decentering_x": ("FLOAT", {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001, "display": "slider"}),
                "decentering_y": ("FLOAT", {"default": 0.0, "min": -0.1, "max": 0.1, "step": 0.001, "display": "slider"}),
                "center_offset_x": ("FLOAT", {"default": 0.0, "min": -0.05, "max": 0.05, "step": 0.001, "display": "slider"}),
                "center_offset_y": ("FLOAT", {"default": 0.0, "min": -0.05, "max": 0.05, "step": 0.001, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_lens_distortion"
    CATEGORY = "CRT/FX"
    
    def _spectrum(self, hue_tensor):
        hue_tensor = hue_tensor * 4.0
        r = torch.clamp(1.5 - torch.abs(hue_tensor - 1.0), 0.0, 1.0)
        g = torch.clamp(1.5 - torch.abs(hue_tensor - 2.0), 0.0, 1.0)
        r += torch.clamp(hue_tensor - 3.5, 0.0, 1.0)
        b = 1.0 - r
        return torch.stack([r, g, b], dim=-1)

    def apply_lens_distortion(self, image: torch.Tensor, radial_distortion_1, radial_distortion_2, radial_distortion_3, radial_distortion_4, anamorphic_squeeze, chromatic_radius, decentering_x, decentering_y, center_offset_x, center_offset_y):
        device = image.device
        batch_size, height, width, _ = image.shape
        result_images = []

        k = torch.tensor([radial_distortion_1, radial_distortion_2, radial_distortion_3, radial_distortion_4], device=device)
        p = torch.tensor([decentering_x, decentering_y], device=device)
        c = torch.tensor([center_offset_x, center_offset_y], device=device)

        for i in range(batch_size):
            img_tensor_bchw = image[i:i+1].permute(0, 3, 1, 2)
            
            y, x = torch.meshgrid(torch.linspace(-1, 1, height, device=device), torch.linspace(-1, 1, width, device=device), indexing='ij')
            
            aspect_ratio = width / height
            x_norm, y_norm = x.clone(), y.clone()
            if aspect_ratio > 1.0:
                x_norm *= aspect_ratio
            else:
                y_norm /= aspect_ratio

            view_coord = torch.stack([x_norm, y_norm], dim=-1) - c
            
            anam_view_coord = view_coord.clone()
            anam_view_coord[..., 1] /= anamorphic_squeeze
            
            anamorph_r_sq = torch.sum(anam_view_coord**2, dim=-1)
            pow_radius = torch.stack([anamorph_r_sq, anamorph_r_sq**2, anamorph_r_sq**3, anamorph_r_sq**4], dim=-1)
            
            radial_dist = 1.0 / (1.0 + torch.sum(k * pow_radius, dim=-1))
            decentering_dist = torch.sum(view_coord * p, dim=-1)
            
            distorted_coord = view_coord * (radial_dist + decentering_dist).unsqueeze(-1) + c

            if aspect_ratio > 1.0:
                distorted_coord[..., 0] /= aspect_ratio
            else:
                distorted_coord[..., 1] *= aspect_ratio

            distorted_grid = distorted_coord.unsqueeze(0)
            
            if chromatic_radius != 0.0:
                original_coord = torch.stack([x, y], dim=-1)
                distortion_vec = distorted_coord - original_coord
                
                accumulated_color = torch.zeros_like(img_tensor_bchw)
                samples = 32
                
                hue_tensor = torch.arange(samples, device=device) / float(samples - 1)
                
                offset_factors = (chromatic_radius * (hue_tensor - 0.5) + 1.0).view(-1, 1, 1, 1)

                sample_coords = original_coord.unsqueeze(0) + distortion_vec.unsqueeze(0) * offset_factors
                sample_coords = sample_coords.view(-1, height, width, 2)
                
                img_tiled = img_tensor_bchw.repeat(samples, 1, 1, 1)
                
                sampled_colors = torch.nn.functional.grid_sample(img_tiled, sample_coords, mode='bilinear', padding_mode='border', align_corners=False)
                
                spectrum_colors = self._spectrum(hue_tensor).view(samples, 3, 1, 1)
                
                accumulated_color = torch.sum(sampled_colors * spectrum_colors, dim=0, keepdim=True)
                
                final_color_bchw = accumulated_color * (2.0 / samples)
            else:
                final_color_bchw = torch.nn.functional.grid_sample(img_tensor_bchw, distorted_grid, mode='bilinear', padding_mode='border', align_corners=False)

            final_image_bhwc = final_color_bchw.permute(0, 2, 3, 1)
            result_images.append(final_image_bhwc.clamp(0.0, 1.0))
            
        return (torch.cat(result_images, dim=0),)