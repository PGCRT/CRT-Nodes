import torch

class FilmGrainFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "grain_amount": ("FLOAT", {"default": 0.333, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "grain_size": ("FLOAT", {"default": 1.5, "min": 0.5, "max": 8.0, "step": 0.1, "display": "slider"}),
                "grain_intensity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "grain_density": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 11.0, "step": 0.1, "display": "slider"}),
                "grain_color": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "intensity_shadows": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "intensity_highlights": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "negative_noise_in_highlights": ("BOOLEAN", {"default": False}),
                "preserve_original_color": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_film_grain"
    CATEGORY = "CRT/FX"

    def _fade(self, t):
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def apply_film_grain(self, image: torch.Tensor, grain_amount, grain_size, grain_intensity, grain_density, grain_color, intensity_shadows, intensity_highlights, negative_noise_in_highlights, preserve_original_color, seed):
        
        device = image.device
        batch_size, height, width, _ = image.shape
        result_images = []

        luma_coeff = torch.tensor([0.299, 0.587, 0.114], device=device).view(1, 3, 1, 1)

        for i in range(batch_size):
            original_image = image[i:i+1].permute(0, 3, 1, 2)
            luma = torch.sum(original_image * luma_coeff, dim=1, keepdim=True)

            generator = torch.Generator(device=device)
            generator.manual_seed(seed + i)
            
            low_res_h = int(height / grain_size) if grain_size > 0 else height
            low_res_w = int(width / grain_size) if grain_size > 0 else width

            noise = torch.rand(1, 3, low_res_h, low_res_w, generator=generator, device=device)
            noise = torch.nn.functional.interpolate(noise, size=(height, width), mode='bicubic', align_corners=False)
            noise = (noise * 2.0) - 1.0

            noise *= grain_intensity
            
            density_factor = max(11.1 - grain_density, 0.1)
            noise = torch.pow(noise.abs(), density_factor) * noise.sign()
            
            luma_faded = self._fade(luma)
            noise *= torch.lerp(torch.tensor(intensity_shadows, device=device), 
                                torch.tensor(intensity_highlights, device=device), 
                                luma_faded)
            
            if negative_noise_in_highlights:
                neg_noise = -noise.abs()
                neg_noise = neg_noise[:, [2, 0, 1], :, :] 
                noise = torch.lerp(noise, neg_noise, luma * luma)

            mono_noise = torch.mean(noise, dim=1, keepdim=True)
            noise = torch.lerp(mono_noise, noise, grain_color)
            
            noisy_image = original_image + noise * grain_amount
            
            if preserve_original_color:
                noisy_luma = torch.sum(noisy_image.clamp(0.0, 1.0) * luma_coeff, dim=1, keepdim=True)
                luma_diff = noisy_luma - luma
                final_image_bchw = (original_image + luma_diff).clamp(0.0, 1.0)
            else:
                final_image_bchw = noisy_image.clamp(0.0, 1.0)
                
            final_image_bhwc = final_image_bchw.permute(0, 2, 3, 1)
            result_images.append(final_image_bhwc)

        return (torch.cat(result_images, dim=0),)