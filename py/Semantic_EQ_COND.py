import torch
import comfy.utils
import re
import random 
import os
import torch.nn.functional as F
import comfy.model_management
import folder_paths
from safetensors.torch import load_file
try:
    from transformers import T5Tokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("\n[FluxSemanticEncoder] WARNING: `transformers` library not found. Debug logging will be disabled.")
    TRANSFORMERS_AVAILABLE = False
VOCAB_MODIFIER_CACHE = {}
TOKENIZER_CACHE = {}

def calculate_eq_gain_for_positional_eq(positional_frequency, eq_bands): 
    total_gain_factor = 1.0 
    if not eq_bands: 
        return total_gain_factor
    
    for band_center, band_gain, band_q in eq_bands: 
        if band_q <= 0: 
            band_q = 0.01 
        sigma = 0.5 / band_q  
        distance = abs(positional_frequency - band_center) 
        exponent_val = -(distance**2) / (2 * sigma**2) 
        influence = torch.exp(torch.tensor(exponent_val, dtype=torch.float32)) 
        effective_band_gain_component = 1.0 + (band_gain - 1.0) * influence 
        total_gain_factor *= effective_band_gain_component.item() 
    return total_gain_factor

class VocabularyModifier:
    def __init__(self, map_path, device, tokenizer_name="t5-large"):
        self.map, self.tokenizer, self.custom_token_ids = None, None, None
        
        if map_path in VOCAB_MODIFIER_CACHE:
            cached_data = VOCAB_MODIFIER_CACHE[map_path]
            self.map, self.custom_token_ids = cached_data['map'], cached_data['custom_token_ids']
        else:
            try:
                state_dict = load_file(map_path, device="cpu")
                self.map = state_dict["neighbors"].to(device)
                non_zero_rows = torch.any(self.map != 0, dim=1)
                self.custom_token_ids = torch.where(non_zero_rows)[0].tolist()
                VOCAB_MODIFIER_CACHE[map_path] = {'map': self.map, 'custom_token_ids': self.custom_token_ids}
                print(f"[VocabModifier] Loaded map and extracted {len(self.custom_token_ids)} custom tokens.")
            except Exception as e: 
                print(f"[VocabModifier] FATAL: Failed to load map file: {e}")
                return
                
        if TRANSFORMERS_AVAILABLE:
            if tokenizer_name in TOKENIZER_CACHE: 
                self.tokenizer = TOKENIZER_CACHE[tokenizer_name]
            else:
                try:
                    from transformers import AutoTokenizer
                    self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
                    TOKENIZER_CACHE[tokenizer_name] = self.tokenizer
                except Exception as e: 
                    print(f"[VocabModifier] Warning: Could not load tokenizer '{tokenizer_name}': {e}")

    def _log_change(self, original_id, new_id, mode):
        if self.tokenizer:
            try:
                new_word = self.tokenizer.decode([new_id], skip_special_tokens=True).strip()
                if original_id is not None:
                    original_word = self.tokenizer.decode([original_id], skip_special_tokens=True).strip()
                    if original_word: 
                        print(f"  [{mode}] Replaced '{original_word}' -> '{new_word}'")
                elif new_word: 
                    print(f"  [{mode}] Injected '{new_word}'")
            except Exception as e:
                print(f"  [{mode}] Token decode error: {e}")

    def apply_semantic_shift(self, token_ids, strength, distance, randomness, max_neighbors, seed):
        if self.map is None: 
            return token_ids
            
        shifted_ids = list(token_ids)
        available_neighbors = self.map.shape[1]
        if max_neighbors > available_neighbors: 
            max_neighbors = available_neighbors
            
        for i, token_id in enumerate(token_ids):
            if not isinstance(token_id, int) or token_id < 0 or token_id >= len(self.map): 
                continue
                
            torch.manual_seed(seed + i)
            random.seed(seed + i)
            
            if torch.rand(1).item() >= strength: 
                continue
                
            neighbors = [n for n in self.map[token_id][:max_neighbors].tolist() if n != 0 and n < self.map.shape[0]]
            if not neighbors: 
                continue
                
            base_index = int(distance * (len(neighbors) - 1))
            if randomness > 0.05:
                offset = random.randint(-max(1, int(randomness*len(neighbors)*0.4)), max(1, int(randomness*len(neighbors)*0.4)))
                base_index = max(0, min(len(neighbors) - 1, base_index + offset))
                
            new_token_id = neighbors[base_index]
            if new_token_id != token_id: 
                self._log_change(token_id, new_token_id, "Shift")
                shifted_ids[i] = new_token_id
                
        return shifted_ids

    def apply_token_injection(self, token_ids, mode, num_to_inject, seed):
        if not self.custom_token_ids or num_to_inject == 0: 
            return token_ids
            
        random.seed(seed)
        
        if mode == 'REPLACE':
            valid_indices = [i for i, tid in enumerate(token_ids) if tid > 2]
            num_to_inject = min(num_to_inject, len(valid_indices))
            if num_to_inject == 0: 
                return token_ids
                
            indices_to_replace = random.sample(valid_indices, num_to_inject)
            new_ids = list(token_ids)
            for index in indices_to_replace:
                new_token_id = random.choice(self.custom_token_ids)
                self._log_change(new_ids[index], new_token_id, "Replace")
                new_ids[index] = new_token_id
            return new_ids
            
        elif mode == 'ADD':
            new_ids = list(token_ids)
            for _ in range(num_to_inject):
                insertion_point = random.randint(0, len(new_ids))
                new_token_id = random.choice(self.custom_token_ids)
                new_ids.insert(insertion_point, new_token_id)
                self._log_change(None, new_token_id, "Add")
            return new_ids
            
        elif mode == 'MIX':
            num_add, num_replace = num_to_inject // 2, num_to_inject - num_to_inject // 2
            temp_ids = self.apply_token_injection(token_ids, 'REPLACE', num_replace, seed)
            return self.apply_token_injection(temp_ids, 'ADD', num_add, seed + 1)
            
        return token_ids
class FluxSemanticEncoder:
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("positive", "negative",)
    FUNCTION = "encode_and_eq"
    CATEGORY = "CRT/Conditioning" 
    
    MAX_CLIP_SEQUENCE_LENGTH = 77
    MAX_T5XXL_SEQUENCE_LENGTH = 512
    MAX_EQ_BANDS = 4

    @classmethod
    def INPUT_TYPES(s):
        try:
            node_path = os.path.dirname(__file__)
            map_dir = os.path.join(node_path, "SemanticShift")
            if os.path.isdir(map_dir):
                map_files = [f for f in os.listdir(map_dir) if f.endswith(".safetensors")]
            else:
                map_files = []
            default_t5_map = next((f for f in map_files if 't5' in f.lower() or 'sauce' in f.lower()), "")
            default_clip_l_map = next((f for f in map_files if 'clip' in f.lower() or 'l' in f.lower()), "")
        except Exception as e:
            print(f"[FluxSemanticEncoder] Warning: Could not scan map directory: {e}")
            map_files, default_t5_map, default_clip_l_map = [], "", ""

        inputs = {"required": {
            "clip": ("CLIP",), 
            "magic_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
            "text": ("STRING", {"forceInput": True}),
            "flux_guidance": ("FLOAT", {"default": 2.7, "min": 0.0, "max": 10.0, "step": 0.1}),
            "multiplier": ("FLOAT", {"default": 1.0, "min": -5.0, "max": 5.0, "step": 0.1}),
            "dry_wet_mix": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            "tensor_modification_mode": (["EQ (Positional)", "Manual Keywords (word:SCALE)", "Magic (Random Scale)"], {"default": "EQ (Positional)"}),
            "magic_scale_min": ("FLOAT", {"default": 0.8, "min": -2.0, "max": 3.0, "step": 0.05}), 
            "magic_scale_max": ("FLOAT", {"default": 1.2, "min": -2.0, "max": 3.0, "step": 0.05}),
            "t5xxl_map_path": (map_files, {"default": default_t5_map}),
            "clip_l_map_path": (map_files, {"default": default_clip_l_map}),
            "target_prompt": (["T5-XXL_Only", "CLIP-L_Only", "Both"],),
            "enable_semantic_shift": ("BOOLEAN", {"default": False}),
            "semantic_shift_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            "semantic_shift_distance": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "semantic_shift_randomness": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
            "semantic_shift_max_neighbors": ("INT", {"default": 50, "min": 1, "max": 50, "step": 1}),
            "enable_token_injection": ("BOOLEAN", {"default": False}),
            "injection_mode": (["ADD", "REPLACE", "MIX"],),
            "num_tokens_to_inject": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
        }}
        
        default_centers = [0.25, 0.5, 0.75, 1.0]
        for i in range(1, s.MAX_EQ_BANDS + 1):
            inputs["required"][f"band_{i}_center"] = ("FLOAT", {"default": default_centers[i-1], "min": 0.0, "max": 1.0, "step": 0.01})
            inputs["required"][f"band_{i}_gain"] = ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05})
            inputs["required"][f"band_{i}_q_factor"] = ("FLOAT", {"default": 1.0, "min": 0.01, "max": 10.0, "step": 0.1})
        
        return inputs

    def _extract_keywords_and_scales(self, prompt_text):
        if not prompt_text: 
            return prompt_text, []
            
        pattern = re.compile(r"\((.*?):(-?[0-9]*\.?[0-9]+)\)")
        matches = list(pattern.finditer(prompt_text))
        if not matches: 
            return prompt_text, []
            
        cleaned_prompt = pattern.sub(r'\1', prompt_text)
        return cleaned_prompt, [{"text": m.group(1).strip(), "scale": float(m.group(2))} for m in matches]

    def _deconstruct_tokens(self, token_data):
        if not token_data: 
            return [], []
        if isinstance(token_data[0], list): 
            token_data = token_data[0]
            
        is_weighted = isinstance(token_data[0], tuple)
        ids = [t[0] if is_weighted else t for t in token_data]
        weights = [t[1] if is_weighted else 1.0 for t in token_data]
        return ids, weights

    def _apply_token_mods(self, ids, weights, modifier, seed, max_len, **kwargs):
        processed_ids = list(ids)
        
        if kwargs.get("enable_semantic_shift"):
            shift_args = {k.replace("semantic_shift_", ""): v for k, v in kwargs.items() if k.startswith("semantic_shift_")}
            processed_ids = modifier.apply_semantic_shift(processed_ids, seed=seed, **shift_args)
            
        if kwargs.get("enable_token_injection"):
            processed_ids = modifier.apply_token_injection(
                processed_ids, 
                kwargs.get("injection_mode"), 
                kwargs.get("num_tokens_to_inject"), 
                seed + 1
            )
        
        if len(processed_ids) > max_len: 
            processed_ids = processed_ids[:max_len]
            
        final_weights = (weights + [1.0] * (len(processed_ids) - len(weights)))[:len(processed_ids)]
        return [(processed_ids[i], final_weights[i]) for i in range(len(processed_ids))]
    
    def _get_cond_and_details(self, encoded):
        if not (encoded and isinstance(encoded[0], list) and len(encoded[0]) == 2): 
            return None, None
        return encoded[0][0], encoded[0][1]

    def encode_and_eq(self, clip, text, **kwargs):
        device = comfy.model_management.get_torch_device()
        prompt_clip_l, prompt_t5xxl = text, text
        flux_guidance, magic_seed = kwargs.get("flux_guidance", 2.7), kwargs.get("magic_seed", 0)
        tensor_mode, target_prompt = kwargs.get("tensor_modification_mode"), kwargs.get("target_prompt")
        
        kw_list_l, kw_list_t5 = [], []
        if tensor_mode == "Manual Keywords (word:SCALE)":
            prompt_clip_l, kw_list_l = self._extract_keywords_and_scales(prompt_clip_l)
            prompt_t5xxl, kw_list_t5 = self._extract_keywords_and_scales(prompt_t5xxl)

        tokens = {}
        if prompt_clip_l: 
            tokens['l'] = clip.tokenize(prompt_clip_l).get('l', [])
        if prompt_t5xxl: 
            tokens['t5xxl'] = clip.tokenize(prompt_t5xxl).get('t5xxl', [])
        
        encoded_base = clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": flux_guidance})
        original_cond, original_details = self._get_cond_and_details(encoded_base)
        if original_cond is None: 
            return (None, None)

        modified_tokens = tokens.copy()
        token_mods_applied = False
        
        if kwargs.get("enable_semantic_shift") or kwargs.get("enable_token_injection"):
            node_path = os.path.dirname(__file__)
            map_dir = os.path.join(node_path, "SemanticShift")
            
            if target_prompt in ["CLIP-L_Only", "Both"] and 'l' in tokens:
                full_map_path = os.path.join(map_dir, kwargs.get("clip_l_map_path", ""))
                if os.path.exists(full_map_path):
                    try:
                        modifier = VocabularyModifier(full_map_path, device, "openai/clip-vit-large-patch14")
                        ids, weights = self._deconstruct_tokens(tokens.get('l'))
                        new_token_data = self._apply_token_mods(ids, weights, modifier, magic_seed, self.MAX_CLIP_SEQUENCE_LENGTH, **kwargs)
                        original_data = tokens.get('l', [])
                        if original_data and new_token_data != original_data[0]: 
                            modified_tokens['l'] = [new_token_data]
                            token_mods_applied = True
                    except Exception as e:
                        print(f"[FluxSemanticEncoder] Error processing CLIP-L tokens: {e}")
            
            if target_prompt in ["T5-XXL_Only", "Both"] and 't5xxl' in tokens:
                full_map_path = os.path.join(map_dir, kwargs.get("t5xxl_map_path", ""))
                if os.path.exists(full_map_path):
                    try:
                        modifier = VocabularyModifier(full_map_path, device, "t5-large")
                        ids, weights = self._deconstruct_tokens(tokens.get('t5xxl'))
                        new_token_data = self._apply_token_mods(ids, weights, modifier, magic_seed + 1, self.MAX_T5XXL_SEQUENCE_LENGTH, **kwargs)
                        original_data = tokens.get('t5xxl', [])
                        if original_data and new_token_data != original_data[0]: 
                            modified_tokens['t5xxl'] = [new_token_data]
                            token_mods_applied = True
                    except Exception as e:
                        print(f"[FluxSemanticEncoder] Error processing T5-XXL tokens: {e}")
        final_len = self.MAX_CLIP_SEQUENCE_LENGTH
        if ('l' in modified_tokens and modified_tokens.get('l') and 
            't5xxl' in modified_tokens and modified_tokens.get('t5xxl')):
            final_len = min(len(modified_tokens['l'][0]), len(modified_tokens['t5xxl'][0]), self.MAX_CLIP_SEQUENCE_LENGTH)
            
        if 'l' in modified_tokens and modified_tokens.get('l'): 
            modified_tokens['l'][0] = modified_tokens['l'][0][:final_len]
        if 't5xxl' in modified_tokens and modified_tokens.get('t5xxl'): 
            modified_tokens['t5xxl'][0] = modified_tokens['t5xxl'][0][:final_len]

        encoded_mod_stage1 = clip.encode_from_tokens_scheduled(modified_tokens, add_dict={"guidance": flux_guidance})
        modified_cond, modified_details = self._get_cond_and_details(encoded_mod_stage1)
        if modified_cond is None: 
            modified_cond = original_cond.clone()
        
        tensor_mod_applied = True
        if tensor_mode == "Manual Keywords (word:SCALE)":
            pass
        elif tensor_mode == "EQ (Positional)":
            parsed_eq_bands = []
            for i in range(1, self.MAX_EQ_BANDS + 1):
                band_gain = kwargs.get(f"band_{i}_gain", 1.0)
                if abs(band_gain - 1.0) > 1e-4: 
                    band_center = kwargs.get(f"band_{i}_center")
                    band_q = kwargs.get(f"band_{i}_q_factor")
                    parsed_eq_bands.append((band_center, band_gain, band_q))
                    
            if parsed_eq_bands:
                seq_len = modified_cond.shape[1]
                if seq_len > 1:
                    target_x = torch.linspace(0.0, 1.0, seq_len, device=modified_cond.device)
                else:
                    target_x = torch.tensor([0.5], device=modified_cond.device)
                    
                for i in range(seq_len): 
                    gain = calculate_eq_gain_for_positional_eq(target_x[i].item(), parsed_eq_bands)
                    modified_cond[0, i, :] *= gain
                    
        elif tensor_mode == "Magic (Random Scale)":
            seq_len = modified_cond.shape[1]
            generator = torch.Generator(device=device).manual_seed(magic_seed)
            scale_range = kwargs.get("magic_scale_max") - kwargs.get("magic_scale_min")
            scales = torch.rand(seq_len, generator=generator, device=device) * scale_range + kwargs.get("magic_scale_min")
            scales = scales.to(modified_cond.device)
            for i in range(seq_len): 
                modified_cond[0, i, :] *= scales[i]
        else:
            tensor_mod_applied = False
        if token_mods_applied or tensor_mod_applied:
            if original_cond.shape[1] == modified_cond.shape[1]:
                effect = (modified_cond - original_cond) * kwargs.get("multiplier", 1.0)
                final_cond = original_cond + (effect * kwargs.get("dry_wet_mix", 1.0))
                
                final_details = original_details.copy() if original_details else {}
                if (original_details and modified_details and 
                    'pooled_output' in original_details and 'pooled_output' in modified_details):
                    pooled_effect = (modified_details['pooled_output'] - original_details['pooled_output']) * kwargs.get("multiplier", 1.0)
                    final_details['pooled_output'] = original_details['pooled_output'] + (pooled_effect * kwargs.get("dry_wet_mix", 1.0))
            else:
                final_cond = modified_cond * kwargs.get("multiplier", 1.0)
                final_details = modified_details if modified_details else original_details
        else:
            final_cond, final_details = original_cond, original_details
        empty_tokens = clip.tokenize("")
        encoded_neg = clip.encode_from_tokens_scheduled(empty_tokens, add_dict={"guidance": flux_guidance})
        
        return ([[final_cond, final_details]], self._ensure_sampler_compatible_conditioning(encoded_neg, flux_guidance))

    def _ensure_sampler_compatible_conditioning(self, encoded, guidance):
        if not (encoded and isinstance(encoded[0], list) and len(encoded[0]) == 2):
            cond = torch.zeros((1, 1, 4096))
            pooled = torch.zeros((1, 1280))
            return [[cond, {"guidance": guidance, "pooled_output": pooled}]]
            
        cond, details = encoded[0][0].clone(), encoded[0][1].copy()
        details["guidance"] = guidance
        
        if "pooled_output" not in details:
            details["pooled_output"] = torch.zeros((cond.shape[0], 1280), device=cond.device, dtype=cond.dtype)
            
        return [[cond, details]]