import torch
import comfy.samplers
import comfy.sample
import comfy.model_management
from comfy.utils import ProgressBar
from nodes import common_ksampler


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def colored_print(message, color=Colors.ENDC):
    print(f"{color}{message}{Colors.ENDC}")


class LatentNoiseInjectionSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "vae": (
                    "VAE",
                    {"tooltip": "The VAE is required to encode the input image and decode the final latent."},
                ),
                "positive": ("CONDITIONING",),
                "negative": (
                    "CONDITIONING",
                    {
                        "tooltip": "Optional negative conditioning. If not connected, empty negative conditioning will be used."
                    },
                ),
                "cfg": (
                    "FLOAT",
                    {
                        "default": 7.0,
                        "min": 0.0,
                        "max": 100.0,
                        "step": 0.1,
                        "tooltip": "Classifier Free Guidance scale. Higher values follow the prompt more closely.",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "forceInput": True,
                        "tooltip": "Amount of denoising to apply. 1.0 = full denoising (txt2img), 0.5-0.8 typical for img2img.",
                    },
                ),
                "seed": ("INT", {"default": 1, "min": 0, "max": 0xFFFFFFFFFFFFFFFF, "forceInput": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "tooltip": "Total number of sampling steps."}),
                "sampler_name": ("STRING", {"default": "deis", "forceInput": True}),
                "scheduler": ("STRING", {"default": "beta", "forceInput": True}),
                "enable_noise_injection": (
                    ["disable", "enable"],
                    {"default": "disable", "tooltip": "Enable or disable noise injection during sampling"},
                ),
                "injection_point": (
                    "FLOAT",
                    {
                        "default": 0.75,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": "Percentage of steps after which to inject noise (e.g., 0.75 for 75%).",
                    },
                ),
                "injection_seed_offset": (
                    "INT",
                    {
                        "default": 1,
                        "min": -100,
                        "max": 100,
                        "step": 1,
                        "tooltip": "An offset added to the main seed to generate the injected noise pattern.",
                    },
                ),
                "injection_strength": (
                    "FLOAT",
                    {
                        "default": 0.25,
                        "min": -20.0,
                        "max": 20.0,
                        "step": 0.01,
                        "tooltip": "The strength of the injected noise.",
                    },
                ),
                "normalize_injected_noise": (
                    ["enable", "disable"],
                    {
                        "default": "enable",
                        "tooltip": "If enabled, normalizes the injected noise to match the latent's mean and std.",
                    },
                ),
            },
            "optional": {
                "latent_image": ("LATENT",),
                "image": ("IMAGE",),
            },
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("latent", "image")
    FUNCTION = "sample"
    CATEGORY = "CRT/Sampling"

    def sample(
        self,
        model,
        vae,
        positive,
        negative,
        cfg,
        seed,
        steps,
        denoise,
        sampler_name,
        scheduler,
        enable_noise_injection,
        injection_point,
        injection_seed_offset,
        injection_strength,
        normalize_injected_noise,
        latent_image=None,
        image=None,
    ):

        colored_print("\n🧪 Starting Latent Noise Injection Sampler...", Colors.HEADER)
        if negative is None:
            colored_print("🔄 Creating empty negative conditioning (not connected)...", Colors.CYAN)
            null_negative = []
            for t in positive:
                null_negative.append([torch.zeros_like(t[0]), t[1].copy()])
            colored_print(f"   ✅ Created {len(null_negative)} empty negative conditioning(s)", Colors.GREEN)
        else:
            colored_print("🔄 Using provided negative conditioning...", Colors.CYAN)
            null_negative = negative
            colored_print(f"   ✅ Using {len(null_negative)} negative conditioning(s)", Colors.GREEN)
        if image is not None:
            colored_print("🖼️  Converting input image to latent space...", Colors.CYAN)
            comfy.model_management.load_model_gpu(vae.patcher)
            latent = vae.encode(image[:, :, :, :3])
            latent_image = {"samples": latent}
            h, w = image.shape[1], image.shape[2]
            colored_print(f"   📐 Image dimensions: {w}x{h}", Colors.BLUE)
        elif latent_image is None:
            colored_print(
                "❌ ERROR: LatentNoiseInjectionSampler requires either an 'image' or a 'latent_image' input.",
                Colors.RED,
            )
            raise ValueError("LatentNoiseInjectionSampler requires either an 'image' or a 'latent_image' input.")
        else:
            colored_print("🎯 Using provided latent input...", Colors.CYAN)
            b, c, h_latent, w_latent = latent_image["samples"].shape
            colored_print(
                f"   📐 Latent dimensions: {w_latent*8}x{h_latent*8} (latent: {w_latent}x{h_latent})", Colors.BLUE
            )
        colored_print("⚙️  Sampling Configuration:", Colors.HEADER)
        colored_print(f"   🎲 Seed: {seed}", Colors.BLUE)
        colored_print(f"   📊 Sampler: {sampler_name} | Scheduler: {scheduler}", Colors.BLUE)
        colored_print(f"   🔄 Total Steps: {steps} | Denoise: {denoise:.2f} | CFG: {cfg:.1f}", Colors.BLUE)
        colored_print(
            f"   💉 Noise Injection: {enable_noise_injection}",
            Colors.CYAN if enable_noise_injection == "enable" else Colors.YELLOW,
        )
        if enable_noise_injection == "disable":
            colored_print("\n🚫 Noise injection disabled - running standard sampling...", Colors.YELLOW)
            final_latent_tuple = common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler, positive, null_negative, latent_image, denoise
            )
            colored_print("🎨 Decoding final latent to image...", Colors.GREEN)
            decoded_image = vae.decode(final_latent_tuple[0]["samples"])
            colored_print("✅ Standard sampling completed successfully!", Colors.GREEN)
            return (
                final_latent_tuple[0],
                decoded_image,
            )
        actual_steps = int(steps * denoise)
        if actual_steps == 0:
            actual_steps = 1
            colored_print("⚠️  WARNING: Denoise value too low, using minimum 1 step.", Colors.YELLOW)

        first_stage_steps = int(actual_steps * injection_point)
        if first_stage_steps == 0:
            first_stage_steps = 1
            colored_print(
                "⚠️  WARNING: Injection point too early, using minimum 1 step for first stage.", Colors.YELLOW
            )
        colored_print(
            f"   💉 Injection Point: {injection_point:.2f} ({first_stage_steps}/{actual_steps} steps)", Colors.CYAN
        )
        colored_print(
            f"   🎲 Injection Seed: {seed + injection_seed_offset} (offset: {injection_seed_offset:+d})", Colors.CYAN
        )
        colored_print(f"   💪 Injection Strength: {injection_strength:.3f}", Colors.CYAN)
        colored_print(f"   📏 Normalize Noise: {normalize_injected_noise}", Colors.CYAN)
        if first_stage_steps >= actual_steps:
            colored_print("🚫 Injection point at or beyond total steps - running standard sampling...", Colors.YELLOW)
            final_latent_tuple = common_ksampler(
                model, seed, steps, cfg, sampler_name, scheduler, positive, null_negative, latent_image, denoise
            )
            colored_print("🎨 Decoding final latent to image...", Colors.GREEN)
            decoded_image = vae.decode(final_latent_tuple[0]["samples"])
            colored_print("✅ Standard sampling completed successfully!", Colors.GREEN)
            return (
                final_latent_tuple[0],
                decoded_image,
            )
        colored_print(f"\n🔥 Stage 1: Initial sampling ({first_stage_steps} steps)...", Colors.GREEN)
        latent_after_stage1 = common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            null_negative,
            latent_image,
            denoise=denoise,
            start_step=0,
            last_step=first_stage_steps,
            force_full_denoise=False,
        )[0]
        colored_print("✅ Stage 1 completed!", Colors.GREEN)
        colored_print(f"\n💉 Injecting noise at step {first_stage_steps}...", Colors.HEADER)
        actual_injection_seed = seed + injection_seed_offset

        injected_latent_samples = latent_after_stage1["samples"].clone()
        original_mean = injected_latent_samples.mean().item()
        original_std = injected_latent_samples.std().item()
        colored_print(f"   📊 Original latent - Mean: {original_mean:.4f}, Std: {original_std:.4f}", Colors.BLUE)

        torch.manual_seed(actual_injection_seed)
        new_noise = torch.randn_like(injected_latent_samples)
        noise_mean = new_noise.mean().item()
        noise_std = new_noise.std().item()
        colored_print(f"   🎲 Generated noise - Mean: {noise_mean:.4f}, Std: {noise_std:.4f}", Colors.BLUE)

        if normalize_injected_noise == "enable":
            colored_print("   📏 Normalizing injected noise to match latent statistics...", Colors.CYAN)
            if original_std > 1e-6:
                new_noise = new_noise * original_std + original_mean
                colored_print(
                    f"   ✅ Noise normalized - Mean: {new_noise.mean().item():.4f}, Std: {new_noise.std().item():.4f}",
                    Colors.CYAN,
                )
            else:
                colored_print("   ⚠️  WARNING: Original std too small, skipping normalization", Colors.YELLOW)
        injected_latent_samples += new_noise * injection_strength
        final_mean = injected_latent_samples.mean().item()
        final_std = injected_latent_samples.std().item()
        colored_print(f"   📊 After injection - Mean: {final_mean:.4f}, Std: {final_std:.4f}", Colors.GREEN)
        change_magnitude = (new_noise * injection_strength).abs().mean().item()
        colored_print(f"   💥 Injection magnitude: {change_magnitude:.4f}", Colors.GREEN)

        injected_latent = latent_after_stage1.copy()
        injected_latent["samples"] = injected_latent_samples
        remaining_steps = actual_steps - first_stage_steps
        colored_print(f"\n🔥 Stage 2: Final sampling ({remaining_steps} steps)...", Colors.GREEN)
        colored_print("   🚫 Noise disabled for consistency", Colors.BLUE)

        final_latent_tuple = common_ksampler(
            model,
            seed,
            steps,
            cfg,
            sampler_name,
            scheduler,
            positive,
            null_negative,
            injected_latent,
            denoise=denoise,
            disable_noise=True,
            start_step=first_stage_steps,
            last_step=actual_steps,
            force_full_denoise=True,
        )

        final_latent = final_latent_tuple[0]
        colored_print("✅ Stage 2 completed!", Colors.GREEN)
        colored_print("\n🎨 Decoding final latent to image...", Colors.GREEN)
        decoded_image = vae.decode(final_latent["samples"])
        colored_print("\n✅ Latent Noise Injection completed successfully!", Colors.HEADER)
        colored_print(
            f"   🎯 Total process: {first_stage_steps} + {remaining_steps} = {actual_steps} steps", Colors.GREEN
        )
        colored_print(f"   💉 Injection applied at {injection_point*100:.1f}% progress", Colors.GREEN)
        colored_print(f"   🎲 Seeds used: {seed} → {actual_injection_seed}", Colors.GREEN)

        return (
            final_latent,
            decoded_image,
        )


NODE_CLASS_MAPPINGS = {"LatentNoiseInjectionSampler": LatentNoiseInjectionSampler}

NODE_DISPLAY_NAME_MAPPINGS = {"LatentNoiseInjectionSampler": "Latent Noise Injection Sampler (CRT)"}
