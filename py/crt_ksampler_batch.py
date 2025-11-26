import torch
import comfy.samplers
import comfy.sample
import comfy.utils
import nodes

class CRT_KSamplerBatch:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The model used for denoising the input latent."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "step":0.1, "round": 0.01}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, ),
                "positive": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "mode": (["Batch (Parallel)", "Sequential"], {"default": "Batch (Parallel)"}),
                "use_same_seed": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "negative": ("CONDITIONING", ),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample_batch"
    CATEGORY = "CRT/Sampling"

    def get_batch_size(self, conditioning):
        if conditioning and len(conditioning) > 0:
            return conditioning[0][0].shape[0]
        return 1

    def prepare_negative(self, neg, target_count, ref_pos):
        if neg is None:
            new_cond = []
            for t, d in ref_pos:
                ref_tensor = t
                zero_tensor = torch.zeros_like(ref_tensor)
                
                if zero_tensor.shape[0] == 1:
                    zero_tensor = zero_tensor.repeat(target_count, 1, 1)
                elif zero_tensor.shape[0] != target_count:
                    zero_tensor = zero_tensor[:target_count]

                new_dict = d.copy()
                if "pooled_output" in d and d["pooled_output"] is not None:
                     p_out = torch.zeros_like(d["pooled_output"])
                     if p_out.shape[0] == 1:
                         p_out = p_out.repeat(target_count, 1)
                     new_dict["pooled_output"] = p_out
                
                new_cond.append([zero_tensor, new_dict])
            return new_cond

        current_batch = self.get_batch_size(neg)
        if current_batch == target_count:
            return neg
        
        if current_batch == 1:
            new_cond = []
            for t, d in neg:
                new_tensor = t.repeat(target_count, 1, 1)
                new_dict = d.copy()
                if "pooled_output" in d:
                    if d["pooled_output"].shape[0] == 1:
                        new_dict["pooled_output"] = d["pooled_output"].repeat(target_count, 1)
                new_cond.append([new_tensor, new_dict])
            return new_cond
        
        return neg

    def slice_cond(self, cond, idx):
        new_cond = []
        for t, d in cond:
            if t.shape[0] > 1:
                sliced_tensor = t[idx:idx+1]
                new_d = d.copy()
                if "pooled_output" in d and d["pooled_output"].shape[0] > 1:
                    new_d["pooled_output"] = d["pooled_output"][idx:idx+1]
                new_cond.append([sliced_tensor, new_d])
            else:
                new_cond.append([t, d])
        return new_cond

    def sample_batch(self, model, seed, steps, cfg, sampler_name, scheduler, positive, latent_image, denoise=1.0, mode="Batch (Parallel)", use_same_seed=False, negative=None):
        target_batch_size = self.get_batch_size(positive)
        processed_negative = self.prepare_negative(negative, target_batch_size, positive)

        latent_samples = latent_image["samples"]
        latent_samples = comfy.sample.fix_empty_latent_channels(model, latent_samples)
        
        current_latent_batch = latent_samples.shape[0]
        
        # Expand latents to match batch size
        if current_latent_batch < target_batch_size:
            repeat_count = target_batch_size // current_latent_batch
            if current_latent_batch == 1:
                work_samples = latent_samples.repeat(target_batch_size, 1, 1, 1)
            else:
                work_samples = torch.cat([latent_samples] * repeat_count, dim=0)[:target_batch_size]
        else:
            work_samples = latent_samples[:target_batch_size]

        callback = nodes.latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        if mode == "Batch (Parallel)":
            if use_same_seed:
                single_slice = work_samples[0:1]
                noise_single = comfy.sample.prepare_noise(single_slice, seed, None)
                noise = noise_single.repeat(target_batch_size, 1, 1, 1)
            else:
                noise = comfy.sample.prepare_noise(work_samples, seed, None)

            samples = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, processed_negative, work_samples,
                                          denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                          force_full_denoise=False, noise_mask=latent_image.get("noise_mask"), callback=callback, disable_pbar=disable_pbar, seed=seed)
        
        else:
            output_samples = []
            for i in range(target_batch_size):
                pos_slice = self.slice_cond(positive, i)
                neg_slice = self.slice_cond(processed_negative, i)
                single_latent = work_samples[i:i+1]
                current_seed = seed if use_same_seed else seed + i
                
                noise = comfy.sample.prepare_noise(single_latent, current_seed, None)
                
                s = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, pos_slice, neg_slice, single_latent,
                                        denoise=denoise, disable_noise=False, start_step=None, last_step=None,
                                        force_full_denoise=False, noise_mask=latent_image.get("noise_mask"), callback=callback, disable_pbar=disable_pbar, seed=current_seed)
                output_samples.append(s)
            
            samples = torch.cat(output_samples, dim=0)

        out = latent_image.copy()
        out["samples"] = samples
        return (out, )