import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF


class AdvancedBloomFX:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "bloom_intensity": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.01, "display": "slider"},
                ),
                "bloom_threshold": (
                    "FLOAT",
                    {"default": 0.8, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"},
                ),
                "bloom_smoothing": (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"},
                ),
                "bloom_radius": (
                    "FLOAT",
                    {"default": 15.0, "min": 1.0, "max": 150.0, "step": 1.0, "display": "slider"},
                ),
                "exposure": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 5.0, "step": 0.01, "display": "slider"}),
                "lightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "saturation": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "hue_shift": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "contrast": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "color_filter": ("COLOR", {"default": "#FFFFFF"}),
                "temperature": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "tint": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"}),
                "split_shadows": ("COLOR", {"default": "#808080"}),
                "split_highlights": ("COLOR", {"default": "#808080"}),
                "split_balance": (
                    "FLOAT",
                    {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01, "display": "slider"},
                ),
                "smh_shadow_color": ("COLOR", {"default": "#FFFFFF"}),
                "smh_midtone_color": ("COLOR", {"default": "#FFFFFF"}),
                "smh_highlight_color": ("COLOR", {"default": "#FFFFFF"}),
                "smh_shadow_start": (
                    "FLOAT",
                    {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"},
                ),
                "smh_shadow_end": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"},
                ),
                "smh_highlight_start": (
                    "FLOAT",
                    {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"},
                ),
                "smh_highlight_end": (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "display": "slider"},
                ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "apply_effect"
    CATEGORY = "CRT/FX"

    def _smoothstep(self, edge0, edge1, x):
        t = torch.clamp((x - edge0) / (edge1 - edge0 + 1e-6), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def _rgb_to_hsv(self, rgb):
        cmax, cmax_idx = torch.max(rgb, dim=-1)
        cmin = torch.min(rgb, dim=-1)[0]
        delta = cmax - cmin
        h = torch.zeros_like(cmax)
        s = torch.zeros_like(cmax)
        s[cmax != 0] = delta[cmax != 0] / cmax[cmax != 0]

        idx = (cmax_idx == 0) & (delta != 0)
        h[idx] = torch.fmod(((rgb[..., 1][idx] - rgb[..., 2][idx]) / delta[idx]), 6)
        idx = (cmax_idx == 1) & (delta != 0)
        h[idx] = ((rgb[..., 2][idx] - rgb[..., 0][idx]) / delta[idx]) + 2
        idx = (cmax_idx == 2) & (delta != 0)
        h[idx] = ((rgb[..., 0][idx] - rgb[..., 1][idx]) / delta[idx]) + 4

        h = h / 6.0
        h[h < 0] += 1.0
        return torch.stack([h, s, cmax], dim=-1)

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

    def _aces_film(self, x):
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return torch.clamp((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)

    def apply_effect(
        self,
        image: torch.Tensor,
        bloom_intensity,
        bloom_threshold,
        bloom_smoothing,
        bloom_radius,
        exposure,
        lightness,
        saturation,
        hue_shift,
        contrast,
        color_filter,
        temperature,
        tint,
        split_shadows,
        split_highlights,
        split_balance,
        smh_shadow_color,
        smh_midtone_color,
        smh_highlight_color,
        smh_shadow_start,
        smh_shadow_end,
        smh_highlight_start,
        smh_highlight_end,
    ):

        device = image.device
        batch_size, _, _, _ = image.shape
        result_images = []

        color_filter = torch.tensor(
            [
                int(color_filter[1:3], 16) / 255.0,
                int(color_filter[3:5], 16) / 255.0,
                int(color_filter[5:7], 16) / 255.0,
            ],
            device=device,
        )
        split_shadows = torch.tensor(
            [
                int(split_shadows[1:3], 16) / 255.0,
                int(split_shadows[3:5], 16) / 255.0,
                int(split_shadows[5:7], 16) / 255.0,
            ],
            device=device,
        )
        split_highlights = torch.tensor(
            [
                int(split_highlights[1:3], 16) / 255.0,
                int(split_highlights[3:5], 16) / 255.0,
                int(split_highlights[5:7], 16) / 255.0,
            ],
            device=device,
        )
        smh_shadow_color = torch.tensor(
            [
                int(smh_shadow_color[1:3], 16) / 255.0,
                int(smh_shadow_color[3:5], 16) / 255.0,
                int(smh_shadow_color[5:7], 16) / 255.0,
            ],
            device=device,
        )
        smh_midtone_color = torch.tensor(
            [
                int(smh_midtone_color[1:3], 16) / 255.0,
                int(smh_midtone_color[3:5], 16) / 255.0,
                int(smh_midtone_color[5:7], 16) / 255.0,
            ],
            device=device,
        )
        smh_highlight_color = torch.tensor(
            [
                int(smh_highlight_color[1:3], 16) / 255.0,
                int(smh_highlight_color[3:5], 16) / 255.0,
                int(smh_highlight_color[5:7], 16) / 255.0,
            ],
            device=device,
        )

        for i in range(batch_size):
            img_tensor = image[i].clone()

            img_tensor *= exposure

            if bloom_intensity > 0:
                knee = bloom_threshold * bloom_smoothing + 1e-5
                curve = torch.tensor([bloom_threshold - knee, knee * 2.0, 0.25 / knee], device=device)

                luma = torch.sum(img_tensor * torch.tensor([0.299, 0.587, 0.114], device=device), dim=-1, keepdim=True)

                response_curve = torch.clamp(luma - curve[0], 0.0, curve[1])
                response_curve = curve[2] * response_curve * response_curve

                bloom_mask = torch.max(response_curve, luma - bloom_threshold) / torch.max(
                    luma, torch.tensor(1e-10, device=device)
                )
                brights = img_tensor * bloom_mask

                radius_val = int(bloom_radius)
                if radius_val % 2 == 0:
                    radius_val += 1

                blurred_brights = TF.gaussian_blur(
                    brights.permute(2, 0, 1), kernel_size=(radius_val, radius_val), sigma=bloom_radius / 3.0
                ).permute(1, 2, 0)

                img_tensor += blurred_brights * bloom_intensity

            img_tensor = torch.clamp(img_tensor + lightness, 0.0, 1.0)

            if saturation != 0.0 or hue_shift != 0.0:
                hsv = self._rgb_to_hsv(img_tensor)
                hsv[..., 1] *= 1.0 + saturation
                hsv[..., 0] += hue_shift
                hsv[..., 0] = torch.fmod(hsv[..., 0] + 1.0, 1.0)
                img_tensor = self._hsv_to_rgb(hsv.clamp(0.0, 1.0))

            if contrast != 0.0:
                img_tensor = torch.clamp(0.5 + (img_tensor - 0.5) * (1.0 + contrast), 0.0, 1.0)

            img_tensor *= color_filter

            if temperature != 0.0 or tint != 0.0:
                temp_scale = torch.exp(torch.tensor(temperature * 0.1, device=device))
                tint_matrix = torch.tensor(
                    [[1.0, 0.0, 0.0], [0.0, 1.0 - tint * 0.5, 0.0], [0.0, 0.0, 1.0 / (1.0 + tint * 0.5)]], device=device
                )
                img_tensor *= torch.tensor([temp_scale, 1.0, 1.0 / temp_scale], device=device)
                img_tensor = torch.matmul(img_tensor.view(-1, 3), tint_matrix.T).view_as(img_tensor)

            if not torch.all(split_shadows == 0.5) or not torch.all(split_highlights == 0.5):
                luma = torch.sum(img_tensor * torch.tensor([0.299, 0.587, 0.114], device=device), dim=-1, keepdim=True)
                balance_point = 0.5 + split_balance * 0.25
                shadow_mask = torch.clamp(1.0 - luma / balance_point, 0.0, 1.0)
                highlight_mask = torch.clamp((luma - balance_point) / (1.0 - balance_point), 0.0, 1.0)
                img_tensor = img_tensor + shadow_mask * (split_shadows - 0.5)
                img_tensor = img_tensor + highlight_mask * (split_highlights - 0.5)

            luma = torch.sum(img_tensor * torch.tensor([0.299, 0.587, 0.114], device=device), dim=-1, keepdim=True)
            shadow_mask = self._smoothstep(smh_shadow_start, smh_shadow_end, luma)
            highlight_mask = self._smoothstep(smh_highlight_start, smh_highlight_end, luma)

            mid_mask = (1.0 - shadow_mask) * (
                1.0 - (1.0 - highlight_mask)
            )  # This is incorrect logic, it should be a bandpass

            smh_shadow_range = smh_shadow_end - smh_shadow_start
            smh_highlight_range = smh_highlight_end - smh_highlight_start

            shadow_w = 1.0 - self._smoothstep(smh_shadow_start, smh_shadow_end, luma)
            highlight_w = self._smoothstep(smh_highlight_start, smh_highlight_end, luma)
            mid_w = 1.0 - shadow_w - highlight_w

            img_tensor = (
                img_tensor * shadow_w * smh_shadow_color
                + img_tensor * mid_w * smh_midtone_color
                + img_tensor * highlight_w * smh_highlight_color
            )

            img_tensor = self._aces_film(img_tensor)
            result_images.append(img_tensor.clamp(0.0, 1.0))

        return (torch.stack(result_images, dim=0),)
