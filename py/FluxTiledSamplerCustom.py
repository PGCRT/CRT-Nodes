import logging
import torch
import comfy.utils
import comfy.model_management
import comfy.samplers
from nodes import VAEEncode, VAEDecode
import numpy as np
import typing

def prepare_inputs(required: list, optional: list = None):
    inputs = {}
    if required:
        inputs["required"] = {}
        for name, type_info in required:
            inputs["required"][name] = type_info
    if optional:
        inputs["optional"] = {}
        for name, type_info in optional:
            inputs["optional"][name] = type_info
    return inputs

class FluxTiledSamplerCustomAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        required_advanced = [
            ("input_type", (["latent", "image"], {"default": "latent"})),
            ("noise", ("NOISE",)),
            ("guider", ("GUIDER",)),
            ("sampler", ("SAMPLER",)),
        ]
        required_tiling = [
            ("vae", ("VAE",)),
            ("columns", ("INT", {"default": 2, "min": 1, "max": 32, "step": 1})),
            ("rows", ("INT", {"default": 2, "min": 1, "max": 32, "step": 1})),
            ("tile_padding", ("INT", {"default": 16, "min": 0, "max": 256, "step": 8})),
            ("mask_blur", ("INT", {"default": 32, "min": 0, "max": 64, "step": 1})),
            ("tiled_vae_decode", ("BOOLEAN", {"default": False})),
        ]
        optional_sigma_calc = [
            ("steps", ("INT", {"default": 20, "min": 1, "max": 10000})),
            ("denoise", ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01})),
            ("scheduler", (comfy.samplers.SCHEDULER_NAMES, {"default": "normal"})),
        ]
        optional_data = [
            ("image_input", ("IMAGE",)),
            ("latent_input", ("LATENT",)),
        ]
        inputs = {}
        inputs["required"] = {**dict(required_advanced), **dict(required_tiling)}
        inputs["optional"] = {**dict(optional_sigma_calc), **dict(optional_data)}
        return inputs

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("output_latent", "output_image")
    FUNCTION = "process_tiled_advanced"
    CATEGORY = "CRT/Sampling"

    def _calculate_sigmas_internal(self, model_patcher, scheduler_name, steps, denoise_strength):
        if denoise_strength <= 0.0:
            return torch.FloatTensor([])
        total_steps = steps
        if denoise_strength < 1.0:
            total_steps = int(steps / denoise_strength)
            if total_steps <= 0:
                total_steps = 1

        model_sampling_obj = None
        if hasattr(model_patcher, 'model') and hasattr(model_patcher.model, 'model_sampling'):
            model_sampling_obj = model_patcher.model.model_sampling
        elif hasattr(model_patcher, 'model_sampling'):
            model_sampling_obj = model_patcher.model_sampling
        else:
            raise AttributeError("Could not find 'model_sampling' attribute on the provided model object needed for sigma calculation.")

        calculated_sigmas = comfy.samplers.calculate_sigmas(model_sampling_obj, scheduler_name, total_steps).cpu()

        if denoise_strength < 1.0:
            if len(calculated_sigmas) > steps + 1:
                calculated_sigmas = calculated_sigmas[-(steps + 1):]
            elif len(calculated_sigmas) <= steps and len(calculated_sigmas) > 0:
                logging.warning(f"Calculated sigmas ({len(calculated_sigmas)}) less than or equal to target steps ({steps}). Using all calculated sigmas.")
            elif len(calculated_sigmas) == 0:
                logging.error(f"Sigma calculation resulted in empty tensor for scheduler {scheduler_name}, steps {total_steps}")
                return torch.FloatTensor([])
        return calculated_sigmas

    def _create_tile_mask(self, tile_latent_w, tile_latent_h, padding_latent, blur_radius_latent, device, r_idx, c_idx, rows, columns):
        mask = torch.ones((1, 1, tile_latent_h, tile_latent_w), device=device, dtype=torch.float32)
        if padding_latent > 0 and blur_radius_latent > 0:
            feather = max(1, blur_radius_latent)
            if tile_latent_h > 0:
                v_ramp = torch.linspace(0.0, 1.0, steps=feather, device=device, dtype=torch.float32)
                top_len = min(feather, tile_latent_h) if r_idx > 0 else 0
                bottom_len = min(feather, tile_latent_h - top_len) if r_idx < rows - 1 else 0
                if top_len > 0:
                    mask[:, :, :top_len, :] *= v_ramp[:top_len].view(1, 1, top_len, 1)
                if bottom_len > 0:
                    mask[:, :, tile_latent_h - bottom_len:, :] *= torch.flip(v_ramp[:bottom_len], dims=[0]).view(1, 1, bottom_len, 1)
            if tile_latent_w > 0:
                h_ramp = torch.linspace(0.0, 1.0, steps=feather, device=device, dtype=torch.float32)
                left_len = min(feather, tile_latent_w) if c_idx > 0 else 0
                right_len = min(feather, tile_latent_w - left_len) if c_idx < columns - 1 else 0
                if left_len > 0:
                    mask[:, :, :, :left_len] *= h_ramp[:left_len].view(1, 1, 1, left_len)
                if right_len > 0:
                    mask[:, :, :, tile_latent_w - right_len:] *= torch.flip(h_ramp[:right_len], dims=[0]).view(1, 1, 1, right_len)
        return mask.clamp(0.0, 1.0)

    def process_tiled_advanced(self, input_type: str, noise: typing.Any,
                              guider: typing.Any, sampler: typing.Any,
                              vae, columns, rows, tile_padding, mask_blur,
                              tiled_vae_decode, steps=20, denoise=1.0, scheduler="normal",
                              image_input=None, latent_input=None):

        pbar = comfy.utils.ProgressBar(rows * columns)

        if not hasattr(guider, 'model_patcher'):
            raise ValueError("The provided 'guider' object does not have a 'model_patcher' attribute (MODEL).")
        device = guider.model_patcher.load_device

        logging.info(f"Calculating sigmas internally: scheduler={scheduler}, steps={steps}, denoise={denoise}")
        sigmas = self._calculate_sigmas_internal(guider.model_patcher, scheduler, steps, denoise)
        if sigmas.numel() == 0:
            raise ValueError("Sigma calculation failed or resulted in empty sigmas. Check scheduler/steps/denoise inputs.")
        sigmas = sigmas.to(device)

        if input_type == "image":
            if image_input is None:
                raise ValueError("Image input is required for input_type 'image'")
            initial_latent_dict = VAEEncode().encode(vae, image_input)[0]
        elif input_type == "latent":
            if latent_input is None:
                raise ValueError("Latent input is required for input_type 'latent'")
            initial_latent_dict = latent_input
        else:
            raise ValueError(f"Unsupported input_type: {input_type}.")

        latent_samples = initial_latent_dict["samples"].clone().to(device)
        b_original, c_original, h_latent, w_latent = latent_samples.shape

        if b_original > 1:
            logging.warning(f"Input batch size is {b_original}. Tiling will process only the first image in the batch.")
            latent_samples = latent_samples[0:1]

        b, c, h_latent, w_latent = latent_samples.shape

        vae_factor = 8
        padding_latent = tile_padding // vae_factor
        blur_radius_latent = mask_blur // vae_factor

        base_tile_w_latent = w_latent // columns
        base_tile_h_latent = h_latent // rows

        output_latent_full = torch.zeros_like(latent_samples, device=device, dtype=latent_samples.dtype)
        blend_weights_full = torch.zeros_like(latent_samples, device=device, dtype=torch.float32)

        original_noise_obj_seed_attr = None
        noise_class_name = type(noise).__name__
        is_custom_random_noise = hasattr(noise, 'seed') and isinstance(noise.seed, int) and noise_class_name == 'Noise_RandomNoise'
        if is_custom_random_noise:
            original_noise_obj_seed_attr = noise.seed

        for r_idx in range(rows):
            for c_idx in range(columns):
                tile_seed_offset = r_idx * columns + c_idx
                x_start = c_idx * base_tile_w_latent
                y_start = r_idx * base_tile_h_latent
                current_tile_w = base_tile_w_latent if c_idx < columns - 1 else (w_latent - x_start)
                current_tile_h = base_tile_h_latent if r_idx < rows - 1 else (h_latent - y_start)
                padded_x_start = max(0, x_start - padding_latent)
                padded_y_start = max(0, y_start - padding_latent)
                padded_x_end = min(w_latent, x_start + current_tile_w + padding_latent)
                padded_y_end = min(h_latent, y_start + current_tile_h + padding_latent)
                tile_process_w = padded_x_end - padded_x_start
                tile_process_h = padded_y_end - padded_y_start

                if tile_process_w <= 0 or tile_process_h <= 0:
                    pbar.update(1)
                    continue

                current_tile_latent_slice = latent_samples[:, :, padded_y_start:padded_y_end, padded_x_start:padded_x_end].clone()
                tile_latent_dict_for_noise = {"samples": current_tile_latent_slice, "batch_index": None}

                current_tile_noise_seed_for_sampler = 0

                if is_custom_random_noise:
                    current_tile_noise_seed_for_sampler = original_noise_obj_seed_attr + tile_seed_offset
                    noise.seed = current_tile_noise_seed_for_sampler
                    tile_noise_tensor = noise.generate_noise(tile_latent_dict_for_noise)
                elif noise_class_name == 'Noise_EmptyNoise':
                    tile_noise_tensor = noise.generate_noise(tile_latent_dict_for_noise)
                else:
                    tile_noise_tensor = noise.generate_noise(tile_latent_dict_for_noise)
                    current_tile_noise_seed_for_sampler = getattr(noise, 'seed', 0) + tile_seed_offset

                tile_noise_tensor = tile_noise_tensor.to(device)

                x0_output_tile = {}
                total_sigma_steps = sigmas.shape[-1] - 1
                if total_sigma_steps < 0:
                    total_sigma_steps = 0
                disable_pbar_for_tile = True

                processed_tile_latent = guider.sample(
                    noise=tile_noise_tensor,
                    latent_image=current_tile_latent_slice,
                    sampler=sampler,
                    sigmas=sigmas,
                    denoise_mask=None,
                    callback=None,  # Removed callback to disable backend preview
                    disable_pbar=disable_pbar_for_tile,
                    seed=current_tile_noise_seed_for_sampler
                )

                tile_blend_mask = self._create_tile_mask(tile_process_w, tile_process_h, padding_latent, blur_radius_latent, device, r_idx, c_idx, rows, columns)
                output_latent_full[:, :, padded_y_start:padded_y_end, padded_x_start:padded_x_end] += processed_tile_latent.to(output_latent_full.dtype) * tile_blend_mask.to(output_latent_full.dtype)
                blend_weights_full[:, :, padded_y_start:padded_y_end, padded_x_start:padded_x_end] += tile_blend_mask

                pbar.update(1)

        if is_custom_random_noise and original_noise_obj_seed_attr is not None:
            noise.seed = original_noise_obj_seed_attr

        output_latent_full /= blend_weights_full.clamp(min=1e-6)
        output_latent_full = torch.nan_to_num(output_latent_full)

        final_image_tensor = None
        output_latent_for_decode = {"samples": output_latent_full}

        if tiled_vae_decode:
            try:
                vae_decode_tile_size_pixels = 512
                vae_decode_tile_latent = vae_decode_tile_size_pixels // 8
                if hasattr(vae, 'decode_tiled') and callable(vae.decode_tiled):
                    final_image_tensor = vae.decode_tiled(output_latent_full, tile_x=vae_decode_tile_latent, tile_y=vae_decode_tile_latent)
                else:
                    logging.warning("vae.decode_tiled not found or incompatible, using full decode via VAEDecode node.")
                    final_image_tensor = VAEDecode().decode(vae, output_latent_for_decode)[0]
            except Exception as e:
                logging.error(f"Tiled VAE Decode failed: {e}. Falling back to full decode via VAEDecode node.")
                final_image_tensor = VAEDecode().decode(vae, output_latent_for_decode)[0]
        else:
            final_image_tensor = VAEDecode().decode(vae, output_latent_for_decode)[0]

        return (output_latent_for_decode, final_image_tensor)

NODE_CLASS_MAPPINGS = {
    "FluxTiledSamplerCustomAdvanced": FluxTiledSamplerCustomAdvanced
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxTiledSamplerCustomAdvanced": "Flux Tiled Sampler (Advanced)"
}
NODE_STYLING = {
    "FluxTiledSamplerCustomAdvanced": {
        "background_color": "#000000",
        "header_color": "#5600BE"
    }
}