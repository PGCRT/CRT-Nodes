import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class ArcaneBloomFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "intensity": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 4.0, "step": 0.01, "display": "slider"}),
                "threshold": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "smoothing": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "radius_multiplier": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1, "display": "slider"}),
                "saturation": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"}),
                "exposure": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 4.0, "step": 0.01, "display": "slider"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_arcane_bloom"
    CATEGORY = "CRT/FX"

    def _gaussian_blur(self, tensor, radius):
        if radius < 0.1: return tensor
        level_h, level_w = tensor.shape[2], tensor.shape[3]
        kernel_size = int(radius * 3.0) | 1 
        max_dim = min(level_h, level_w)
        if kernel_size > max_dim:
            kernel_size = max_dim if max_dim % 2 != 0 else max_dim - 1
        if kernel_size < 3: return tensor
        return TF.gaussian_blur(tensor, kernel_size=(kernel_size, kernel_size), sigma=radius)

    def apply_arcane_bloom(self, image: torch.Tensor, intensity, threshold, smoothing, radius_multiplier, saturation, exposure):
        device = image.device
        batch_size, height, width, _ = image.shape
        result_images = []

        luma_weights = torch.tensor([0.2126, 0.7152, 0.0722], device=device).view(1, 3, 1, 1)

        for i in range(batch_size):
            img_tensor_bchw = image[i:i+1].permute(0, 3, 1, 2)
            
            # --- Threshold and Soft-Knee Method ---
            luma = torch.sum(img_tensor_bchw * luma_weights, dim=1, keepdim=True)
            
            soft_knee = threshold * smoothing + 1e-5
            
            soft_threshold = torch.clamp(luma - threshold + soft_knee, 0.0, soft_knee * 2.0)
            soft_threshold = (soft_threshold * soft_threshold) / (soft_knee * 4.0 + 1e-7)
            
            hard_threshold = torch.relu(luma - threshold)
            
            bloom_mask = (soft_threshold + hard_threshold)
            
            bloom_source = img_tensor_bchw * bloom_mask

            # --- Pyramid Generation ---
            pyramid = []
            current_level = bloom_source
            for _ in range(6):
                if current_level.shape[2] < 4 or current_level.shape[3] < 4: break
                downsampled = F.interpolate(current_level, scale_factor=0.5, mode='bilinear', align_corners=False)
                pyramid.append(downsampled)
                current_level = downsampled

            if not pyramid:
                result_images.append(image[i:i+1])
                continue

            # --- Correct Additive Upscale ---
            last_level = self._gaussian_blur(pyramid[-1], radius_multiplier)

            for j in range(len(pyramid) - 2, -1, -1):
                upscaled = F.interpolate(last_level, size=(pyramid[j].shape[2], pyramid[j].shape[3]), mode='bilinear', align_corners=False)
                combined = upscaled + pyramid[j]
                last_level = self._gaussian_blur(combined, radius_multiplier)

            bloom_texture = F.interpolate(last_level, size=(height, width), mode='bilinear', align_corners=False)
            
            # --- Apply modifications only to the bloom overlay ---
            if saturation != 1.0:
                bloom_luma = torch.sum(bloom_texture * luma_weights, dim=1, keepdim=True)
                bloom_texture = torch.lerp(bloom_luma, bloom_texture, saturation)
            
            bloom_texture *= intensity
            bloom_texture *= exposure
            
            # --- Final Composite ---
            # Screen blend mode: 1 - (1 - base) * (1 - overlay)
            final_image = 1.0 - (1.0 - img_tensor_bchw) * (1.0 - bloom_texture)
            
            result_images.append(final_image.permute(0, 2, 3, 1))

        return (torch.cat(result_images, dim=0).clamp(0.0, 1.0),)