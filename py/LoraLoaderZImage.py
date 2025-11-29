import torch
import comfy.utils
import comfy.sd
import folder_paths
from nodes import LoraLoader

class LoraLoaderZImage(LoraLoader):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "The Z-Image diffusion model."}),
                "clip": ("CLIP", {"tooltip": "The CLIP model."}),
                "lora_name": (folder_paths.get_filename_list("loras"), {"tooltip": "The name of the Z-Image LoRA."}),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP")
    FUNCTION = "load_zimage_lora"
    CATEGORY = "CRT-Nodes/Loaders"
    DESCRIPTION = "Loads Z-Image LoRAs with auto-fusion for QKV mismatch. Preserves prefixes."

    def load_zimage_lora(self, model, clip, lora_name, strength_model, strength_clip):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip)

        lora_path = folder_paths.get_full_path_or_raise("loras", lora_name)
        lora = None
        
        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                self.loaded_lora = None

        if lora is None:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            
            new_lora = {}
            # Temporary storage to group QKV keys
            # Structure: { "diffusion_model.layers.0.attention": { "q": {A, B}, "k": {A, B}, "v": {A, B} } }
            qkv_groups = {}
            
            for k, v in lora.items():
                new_k = k
                
                # IMPORTANT: Do NOT strip 'diffusion_model.' prefix.
                # ComfyUI's internal state dict expects it.
                
                # 1. Output Projection Fix (to_out.0 -> out)
                # Matches Base Key: diffusion_model.layers.X.attention.out.weight
                if ".attention.to_out.0." in new_k:
                    new_k = new_k.replace(".attention.to_out.0.", ".attention.out.")
                    new_lora[new_k] = v
                    continue

                # 2. Handle QKV Separation
                # LoRA Key: diffusion_model.layers.X.attention.to_q.lora_A.weight
                if ".attention.to_" in new_k:
                    # Extract layer base: diffusion_model.layers.0.attention
                    # Extract type: q, k, or v
                    # Extract param: lora_A.weight or lora_B.weight
                    parts = new_k.split(".attention.to_")
                    base_prefix = parts[0] + ".attention" 
                    remainder = parts[1] # e.g. q.lora_A.weight
                    
                    qkv_type = remainder[0] # 'q', 'k', or 'v'
                    suffix = remainder[2:] # 'lora_A.weight'
                    
                    if base_prefix not in qkv_groups:
                        qkv_groups[base_prefix] = {'q': {}, 'k': {}, 'v': {}}
                    
                    qkv_groups[base_prefix][qkv_type][suffix] = v
                    continue

                # 3. Pass through everything else 
                # (mlp, adaLN_modulation should match if we keep prefix)
                new_lora[new_k] = v

            # --- FUSE QKV ---
            for base_key, group in qkv_groups.items():
                # Base Key Target: diffusion_model.layers.X.attention.qkv.weight
                
                # Check A weights (Down)
                ak_a = "lora_A.weight"
                if ak_a in group['q'] and ak_a in group['k'] and ak_a in group['v']:
                    q_a = group['q'][ak_a]
                    k_a = group['k'][ak_a]
                    v_a = group['v'][ak_a]
                    
                    # Stack A vertically: (3*rank, dim_in)
                    fused_A = torch.cat([q_a, k_a, v_a], dim=0)
                    new_lora[fused_A_key := f"{base_key}.qkv.lora_A.weight"] = fused_A

                # Check B weights (Up)
                ak_b = "lora_B.weight"
                if ak_b in group['q'] and ak_b in group['k'] and ak_b in group['v']:
                    q_b = group['q'][ak_b]
                    k_b = group['k'][ak_b]
                    v_b = group['v'][ak_b]
                    
                    # Block Diagonal B: (3*dim_out, 3*rank)
                    out_dim, rank = q_b.shape
                    
                    fused_B = torch.zeros((out_dim * 3, rank * 3), dtype=q_b.dtype, device=q_b.device)
                    fused_B[0:out_dim, 0:rank] = q_b
                    fused_B[out_dim:2*out_dim, rank:2*rank] = k_b
                    fused_B[2*out_dim:3*out_dim, 2*rank:3*rank] = v_b
                    
                    new_lora[fused_B_key := f"{base_key}.qkv.lora_B.weight"] = fused_B

                # Handle Alphas if necessary (Generic copy)
                ak_alpha = "lora_alpha"
                if ak_alpha in group['q']:
                    # Use Q's alpha for the fused block
                    new_lora[f"{base_key}.qkv.lora_alpha"] = group['q'][ak_alpha]

            lora = new_lora
            self.loaded_lora = (lora_path, lora)

        model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)
        return (model_lora, clip_lora)