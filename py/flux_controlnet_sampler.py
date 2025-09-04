import torch
import torch.nn.functional as F
import numpy as np
import comfy.utils
import comfy.sd
import comfy.controlnet
import folder_paths
from nodes import common_ksampler

from PIL import Image

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

class ImageResize:
    def execute(self, image, width, height, method="stretch", interpolation="lanczos"):
        _, oh, ow, _ = image.shape
        colored_print(f"🔄 Resizing image from {ow}x{oh}...", Colors.CYAN)
        
        if method == 'keep proportion':
            if ow > oh:
                ratio = width / ow
            else:
                ratio = height / oh
            width, height = round(ow * ratio), round(oh * ratio)
            colored_print(f"   📐 Keeping proportions - Target: {width}x{height} (ratio: {ratio:.3f})", Colors.BLUE)
        else:
            width = width if width > 0 else ow
            height = height if height > 0 else oh
            colored_print(f"   📐 Direct resize to: {width}x{height}", Colors.BLUE)
            
        outputs = image.permute(0, 3, 1, 2)
        if interpolation == "lanczos":
            colored_print(f"   🎯 Using Lanczos interpolation", Colors.BLUE)
            outputs = comfy.utils.lanczos(outputs, width, height)
        else:
            colored_print(f"   🎯 Using {interpolation} interpolation", Colors.BLUE)
            outputs = F.interpolate(outputs, size=(height, width), mode=interpolation)
        
        colored_print(f"✅ Image resize completed!", Colors.GREEN)
        return torch.clamp(outputs.permute(0, 2, 3, 1), 0, 1)

class ColorMatch:
    def colormatch(self, image_ref, image_target, method, strength=1.0):
        colored_print(f"🎨 Applying color matching (method: {method}, strength: {strength:.2f})...", Colors.CYAN)
        
        try:
            from color_matcher import ColorMatcher
        except ImportError:
            colored_print("❌ ERROR: 'color-matcher' library not found.", Colors.RED)
            print("################################################################################")
            print("## ComfyUI-CRT: 'color-matcher' library not found.")
            print("## Please install it by running: pip install color-matcher")
            print("################################################################################")
            raise ImportError("ColorMatch requires 'color-matcher'. Please install it.")
        
        cm = ColorMatcher()
        out = []
        image_ref, image_target = image_ref.cpu(), image_target.cpu()
        
        batch_size = image_target.size(0)
        colored_print(f"   🔄 Processing {batch_size} image(s)...", Colors.BLUE)
        
        for i in range(batch_size):
            target_np = image_target[i].numpy()
            ref_np = image_ref[i if image_ref.size(0) == image_target.size(0) else 0].numpy()
            target_mean = np.mean(target_np, axis=(0,1))
            ref_mean = np.mean(ref_np, axis=(0,1))
            
            result_np = cm.transfer(src=target_np, ref=ref_np, method=method)
            blended_np = target_np + strength * (result_np - target_np)
            final_mean = np.mean(blended_np, axis=(0,1))
            
            colored_print(f"   📊 Image {i+1}: Target RGB({target_mean[0]:.3f},{target_mean[1]:.3f},{target_mean[2]:.3f}) → Final RGB({final_mean[0]:.3f},{final_mean[1]:.3f},{final_mean[2]:.3f})", Colors.BLUE)
            
            out.append(torch.from_numpy(blended_np))
        
        colored_print("✅ Color matching completed!", Colors.GREEN)
        return (torch.stack(out, dim=0).to(torch.float32).clamp_(0, 1),)

class FluxControlnetSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "vae": ("VAE",),
                "control_net": ("CONTROL_NET",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                
                "seed_shift": ("INT", {"default": 0, "min": -100000, "max": 100000, "step": 1, "tooltip": "Offset added to the main seed for variation"}),
                
                "sampler_name": ("STRING", {"default": "dpmpp_2m_sde", "forceInput": True}),
                "scheduler": ("STRING", {"default": "karras", "forceInput": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                "upscale_by": ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.01}),
                "controlnet_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 10.0, "step": 0.01}),
                "control_end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "color_match_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
            },
            "optional": {
                "image": ("IMAGE",),
                "latent": ("LATENT",),
            }
        }

    RETURN_TYPES = ("IMAGE", "LATENT")
    RETURN_NAMES = ("image", "latent")
    FUNCTION = "execute"
    CATEGORY = "CRT/Sampling"

    def execute(self, model, positive, vae, control_net, seed, seed_shift, steps, sampler_name, scheduler,
                upscale_by, controlnet_strength, control_end, color_match_strength, image=None, latent=None):

        colored_print("\n🎮 Starting Flux ControlNet Sampling...", Colors.HEADER)
        
        cfg = 1.0
        null_negative = []
        denoise = 1.0

        actual_seed = seed + seed_shift
        colored_print(f"🎲 Seed Configuration:", Colors.HEADER)
        colored_print(f"   Base Seed: {seed}", Colors.BLUE)
        colored_print(f"   Seed Shift: {seed_shift:+d}", Colors.BLUE)
        colored_print(f"   Final Seed: {actual_seed}", Colors.GREEN)
        if image is None and latent is None:
            colored_print("❌ ERROR: FluxControlnetSampler requires either an 'image' or a 'latent' input.", Colors.RED)
            raise ValueError("FluxControlnetSampler requires either an 'image' or a 'latent' input.")
        if image is not None:
            colored_print("🖼️  Using provided image input...", Colors.CYAN)
            source_image = image
            h, w = image.shape[1], image.shape[2]
            colored_print(f"   📐 Source image dimensions: {w}x{h}", Colors.BLUE)
        else:
            colored_print("🎯 Decoding latent to image...", Colors.CYAN)
            source_image = vae.decode(latent["samples"])
            h, w = source_image.shape[1], source_image.shape[2]
            colored_print(f"   📐 Decoded image dimensions: {w}x{h}", Colors.BLUE)

        if latent is not None:
            colored_print("🎯 Using provided latent input...", Colors.CYAN)
            initial_latent_samples = latent["samples"]
            b, c, h_latent, w_latent = initial_latent_samples.shape
            colored_print(f"   📐 Latent dimensions: {w_latent}x{h_latent} (channels: {c})", Colors.BLUE)
        else:
            colored_print("🔄 Encoding image to latent space...", Colors.CYAN)
            initial_latent_samples = vae.encode(source_image)
            b, c, h_latent, w_latent = initial_latent_samples.shape
            colored_print(f"   📐 Encoded latent dimensions: {w_latent}x{h_latent} (channels: {c})", Colors.BLUE)
        new_h, new_w = int(h_latent * upscale_by), int(w_latent * upscale_by)
        target_pixel_w, target_pixel_h = new_w * 8, new_h * 8
        
        colored_print(f"📈 Upscaling Configuration:", Colors.HEADER)
        colored_print(f"   Upscale Factor: {upscale_by:.2f}x", Colors.BLUE)
        colored_print(f"   Latent: {w_latent}x{h_latent} → {new_w}x{new_h}", Colors.BLUE)
        colored_print(f"   Pixel: {w_latent*8}x{h_latent*8} → {target_pixel_w}x{target_pixel_h}", Colors.GREEN)
        colored_print("🔄 Upscaling latent samples...", Colors.CYAN)
        upscaled_latent_samples = F.interpolate(initial_latent_samples, size=(new_h, new_w), mode='bicubic', align_corners=False)
        upscaled_latent = {"samples": upscaled_latent_samples}
        colored_print("✅ Latent upscaling completed!", Colors.GREEN)
        colored_print("🎮 Preparing ControlNet input...", Colors.CYAN)
        resizer = ImageResize()
        control_image = resizer.execute(source_image, target_pixel_w, target_pixel_h, method="keep proportion", interpolation="lanczos")
        control_start = 0.0
        cnet_positive = positive
        cnet_negative = null_negative
        
        colored_print(f"🎮 ControlNet Configuration:", Colors.HEADER)
        colored_print(f"   Strength: {controlnet_strength:.3f}", Colors.BLUE)
        colored_print(f"   Control Range: {control_start:.3f} → {control_end:.3f}", Colors.BLUE)
        colored_print(f"   Color Match Strength: {color_match_strength:.3f}", Colors.BLUE)
        
        if controlnet_strength > 0:
            colored_print("🔧 Applying ControlNet to conditioning...", Colors.CYAN)
            cnet_positive, cnet_negative = [], []
            
            conditioning_count = 0
            for conditioning_list, out_list in [(positive, cnet_positive), (null_negative, cnet_negative)]:
                for c in conditioning_list:
                    conditioning_count += 1
                    new_c = [c[0], c[1].copy()]
                    c_net = control_net.copy()
                    c_net.set_cond_hint(control_image.movedim(-1, 1), controlnet_strength, (control_start, control_end), vae=vae)
                    
                    if 'control' in new_c[1]:
                        colored_print(f"   🔗 Chaining with existing ControlNet", Colors.BLUE)
                        c_net.set_previous_controlnet(new_c[1]['control'])
                    
                    new_c[1]['control'] = c_net
                    new_c[1]['control_start'] = control_start
                    new_c[1]['control_end'] = control_end
                    out_list.append(new_c)
            
            colored_print(f"✅ ControlNet applied to {conditioning_count} conditioning(s)!", Colors.GREEN)
        else:
            colored_print("🚫 ControlNet strength is 0 - skipping ControlNet application", Colors.YELLOW)
        colored_print(f"⚙️  Sampling Configuration:", Colors.HEADER)
        colored_print(f"   🎲 Seed: {actual_seed}", Colors.BLUE)
        colored_print(f"   📊 Sampler: {sampler_name} | Scheduler: {scheduler}", Colors.BLUE)
        colored_print(f"   🔄 Steps: {steps} | CFG: {cfg} | Denoise: {denoise}", Colors.BLUE)
        colored_print("\n🔥 Starting ControlNet-guided sampling...", Colors.GREEN)
        final_latent_tuple = common_ksampler(
            model, actual_seed, steps, cfg, sampler_name, scheduler,
            cnet_positive, cnet_negative, upscaled_latent, denoise=denoise
        )
        final_latent = final_latent_tuple[0]
        colored_print("✅ Sampling completed!", Colors.GREEN)
        colored_print("🎨 Decoding final latent to image...", Colors.CYAN)
        sampled_image = vae.decode(final_latent["samples"])
        final_h, final_w = sampled_image.shape[1], sampled_image.shape[2]
        colored_print(f"   📐 Final image dimensions: {final_w}x{final_h}", Colors.BLUE)
        if color_match_strength > 0:
            colored_print("🎨 Applying color matching...", Colors.HEADER)
            matcher = ColorMatch()
            final_image = matcher.colormatch(control_image, sampled_image, method='mkl', strength=color_match_strength)[0]
        else:
            colored_print("🚫 Color matching disabled (strength = 0)", Colors.YELLOW)
            final_image = sampled_image
        colored_print(f"\n✅ Flux ControlNet Sampling completed successfully!", Colors.HEADER)
        colored_print(f"   🎯 Process: {w_latent*8}x{h_latent*8} → {final_w}x{final_h} ({upscale_by:.2f}x)", Colors.GREEN)
        colored_print(f"   🎮 ControlNet: {'Applied' if controlnet_strength > 0 else 'Skipped'} (strength: {controlnet_strength:.3f})", Colors.GREEN)
        colored_print(f"   🎨 Color Match: {'Applied' if color_match_strength > 0 else 'Skipped'} (strength: {color_match_strength:.3f})", Colors.GREEN)
        colored_print(f"   🎲 Seed used: {actual_seed}", Colors.GREEN)
        
        return (final_image, final_latent)

NODE_CLASS_MAPPINGS = {
    "FluxControlnetSampler": FluxControlnetSampler
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "FluxControlnetSampler": "Flux Controlnet Sampler (CRT)"
}