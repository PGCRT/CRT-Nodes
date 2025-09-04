import torch

class ColorIsolationFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target_color": ("COLOR", {"default": "#FF0000"}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -0.5, "max": 0.5, "step": 0.01, "display": "slider"}),
                "window_function": (["Gauss", "Triangle"],),
                "hue_width": ("FLOAT", {"default": 0.3, "min": 0.001, "max": 2.0, "step": 0.001, "display": "slider"}),
                "curve_steepness": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01, "display": "slider"}),
                "mode": (["Isolate", "Reject"],),
                "output_mode": (["Final Image", "Show Mask"],),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_color_isolation"
    CATEGORY = "CRT/FX"

    def _rgb_to_hsv(self, rgb):
        cmax, _ = torch.max(rgb, dim=-1)
        cmin, _ = torch.min(rgb, dim=-1)
        delta = cmax - cmin
        
        h = torch.zeros_like(cmax)
        s = torch.zeros_like(cmax)
        v = cmax
        
        s[cmax != 0] = delta[cmax != 0] / (cmax[cmax != 0] + 1e-7)
        
        rc, gc, bc = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        
        idx = (cmax == rc) & (delta != 0)
        h[idx] = torch.fmod(((gc[idx] - bc[idx]) / delta[idx]), 6)
        
        idx = (cmax == gc) & (delta != 0)
        h[idx] = ((bc[idx] - rc[idx]) / delta[idx]) + 2
        
        idx = (cmax == bc) & (delta != 0)
        h[idx] = ((rc[idx] - gc[idx]) / delta[idx]) + 4
        
        h = h / 6.0
        h[h < 0] += 1.0
        
        return torch.stack([h, s, v], dim=-1)

    def _hsv_to_rgb(self, hsv):
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = torch.floor(h * 6)
        f = h * 6 - i
        p = v * (1 - s)
        q = v * (1 - f * s)
        t = v * (1 - (1 - f) * s)
        i = i.long() % 6
        
        rgb = torch.zeros_like(hsv)
        
        rgb[i == 0] = torch.stack([v, t, p], dim=-1)[i == 0]
        rgb[i == 1] = torch.stack([q, v, p], dim=-1)[i == 1]
        rgb[i == 2] = torch.stack([p, v, t], dim=-1)[i == 2]
        rgb[i == 3] = torch.stack([p, q, v], dim=-1)[i == 3]
        rgb[i == 4] = torch.stack([t, p, v], dim=-1)[i == 4]
        rgb[i == 5] = torch.stack([v, p, q], dim=-1)[i == 5]
        
        return rgb
        
    def _calculate_mask(self, x, height, offset, overlap, func_type):
        if func_type == "Gauss":
            overlap_scaled = overlap / 5.0
            val = height * torch.exp(-torch.pow(x - offset, 2) / (2 * overlap_scaled**2 + 1e-7))
        else:
            val = height * (1.0 - torch.abs(x - offset) / (overlap * 0.5 + 1e-7))

        return val.clamp(0.0, 1.0)

    def apply_color_isolation(self, image: torch.Tensor, target_color, hue_shift, window_function, hue_width, curve_steepness, mode, output_mode):
        device = image.device
        batch_size, height, width, channels = image.shape
        result_images = []

        luma_weights = torch.tensor([0.2126, 0.7151, 0.0721], device=device).view(1, 1, 1, 3)
        
        r = int(target_color[1:3], 16) / 255.0
        g = int(target_color[3:5], 16) / 255.0
        b = int(target_color[5:7], 16) / 255.0
        target_rgb_tensor = torch.tensor([r, g, b], device=device).view(1, 1, 1, 3)
        target_hue = self._rgb_to_hsv(target_rgb_tensor)[0, 0, 0, 0]

        for i in range(batch_size):
            img_tensor = image[i:i+1]
            hsv = self._rgb_to_hsv(img_tensor)
            hue = hsv[..., 0]
            
            offset = target_hue
            
            mask1 = self._calculate_mask(hue, curve_steepness, offset, hue_width, window_function)
            mask2 = self._calculate_mask(hue - 1.0, curve_steepness, offset, hue_width, window_function)
            mask3 = self._calculate_mask(hue + 1.0, curve_steepness, offset, hue_width, window_function)
            
            final_mask = (mask1 + mask2 + mask3).clamp(0.0, 1.0)

            if mode == "Reject":
                final_mask = 1.0 - final_mask
            
            final_mask_expanded = final_mask.unsqueeze(-1)
                
            if output_mode == "Show Mask":
                final_image = final_mask_expanded.repeat(1, 1, 1, channels)
            else:
                luma = torch.sum(img_tensor * luma_weights, dim=-1, keepdim=True)
                
                isolated_color = img_tensor
                if abs(hue_shift) > 1e-6:
                    shifted_hsv = hsv.clone()
                    shifted_hsv[..., 0] = torch.fmod(shifted_hsv[..., 0] + hue_shift + 1.0, 1.0)
                    isolated_color = self._hsv_to_rgb(shifted_hsv)
                
                final_image = torch.lerp(luma, isolated_color, final_mask_expanded)
            
            result_images.append(final_image.clamp(0.0, 1.0))
            
        return (torch.cat(result_images, dim=0),)