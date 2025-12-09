import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import nodes

class CRT_KSamplerBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                "positive": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (["Batch (Parallel)", "Sequential"], {"default": "Batch (Parallel)"}),
                "use_same_seed": ("BOOLEAN", {"default": False}),
            },
            "optional": {"negative": ("CONDITIONING",),}
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample_batch"
    CATEGORY = "CRT/Sampling"

    def _fix_conditioning(self, conditioning, target_batch_size):
        """Force every conditioning entry to have a valid pooled_output tensor."""
        if not conditioning:
            return conditioning

        fixed = []
        for cond_tensor, cond_dict in conditioning:
            new_dict = cond_dict.copy()

            # Ensure pooled_output exists and has correct batch size
            if "pooled_output" not in new_dict or new_dict["pooled_output"] is None:
                hidden = cond_tensor.shape[-1]
                # Use same device/dtype as the conditioning tensor
                pooled = torch.zeros(target_batch_size, hidden, device=cond_tensor.device, dtype=cond_tensor.dtype)
            else:
                pooled = new_dict["pooled_output"]
                if pooled.shape[0] != target_batch_size:
                    if pooled.shape[0] == 1:
                        pooled = pooled.repeat(target_batch_size, 1)
                    else:
                        pooled = pooled[:target_batch_size]
            new_dict["pooled_output"] = pooled
            fixed.append([cond_tensor, new_dict])
        return fixed

    def sample_batch(self, model, seed, steps, cfg, sampler_name, scheduler, positive,
                     latent_image, denoise=1.0, mode="Batch (Parallel)", use_same_seed=False, negative=None):

        target_batch = positive[0][0].shape[0] if positive else 1

        # Fix positive
        positive = self._fix_conditioning(positive, target_batch)

        # Fix negative
        if negative:
            negative = self._fix_conditioning(negative, target_batch)
        else:
            # Create empty negative with zeros (including pooled)
            ref = positive[0][0]
            zero_cond = torch.zeros_like(ref)
            zero_pooled = torch.zeros(target_batch, ref.shape[-1], device=ref.device, dtype=ref.dtype)
            negative = [[zero_cond, {"pooled_output": zero_pooled}]]

        # Fix latent batch size
        samples = latent_image["samples"]
        samples = comfy.sample.fix_empty_latent_channels(model, samples)
        if samples.shape[0] < target_batch:
            samples = samples.repeat(target_batch // samples.shape[0] + 1, 1, 1, 1)[:target_batch]
        elif samples.shape[0] > target_batch:
            samples = samples[:target_batch]

        # Noise
        if use_same_seed and mode == "Batch (Parallel)":
            noise = comfy.sample.prepare_noise(samples[0:1], seed, None).repeat(target_batch, 1, 1, 1)
        else:
            noise = comfy.sample.prepare_noise(samples, seed if use_same_seed else seed, None)

        # --- RESTORE PREVIEW CALLBACK ---
        callback = nodes.latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        # Sampling
        if mode == "Sequential":
            latents = []
            for i in range(target_batch):
                pos = [[positive[j][0][i:i+1], positive[j][1]] for j in range(len(positive))]
                neg = [[negative[j][0][i:i+1], negative[j][1]] for j in range(len(negative))]
                latent = samples[i:i+1]
                cur_seed = seed if use_same_seed else seed + i
                noise_i = comfy.sample.prepare_noise(latent, cur_seed, None)
                
                # Pass callback and disable_pbar
                out = comfy.sample.sample(model, noise_i, steps, cfg, sampler_name, scheduler,
                                          pos, neg, latent, denoise=denoise, seed=cur_seed,
                                          callback=callback, disable_pbar=disable_pbar)
                latents.append(out)
            samples = torch.cat(latents, dim=0)
        else:
            # Pass callback and disable_pbar
            samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler,
                                          positive, negative, samples, denoise=denoise, seed=seed,
                                          callback=callback, disable_pbar=disable_pbar)

        result = latent_image.copy()
        result["samples"] = samples
        return (result,)