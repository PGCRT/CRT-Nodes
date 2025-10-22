import torch

class CRT_DynamicPromptScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT", "IMAGE",)
    RETURN_NAMES = ("conditioning", "batch_count", "image_batch",)
    FUNCTION = "schedule"
    CATEGORY = "CRT/Conditioning"

    def schedule(self, clip, **kwargs):
        prompt_image_pairs = []
        
        prompt_keys = sorted([key for key in kwargs.keys() if key.startswith('prompt_')])
        
        for prompt_key in prompt_keys:
            prompt_value = kwargs.get(prompt_key)
            prompt_id = prompt_key.split('_')[1]
            image_key = f'image_{prompt_id}'
            image_value = kwargs.get(image_key)
            
            if isinstance(prompt_value, str) and prompt_value.strip():
                prompt_image_pairs.append((prompt_value, image_value))

        if not prompt_image_pairs:
            prompt_image_pairs.append(("", None))

        batch_count = len(prompt_image_pairs)

        cond_list = []
        pooled_list = []
        image_list = []
        
        has_pooled_output = True

        for idx, (prompt, image) in enumerate(prompt_image_pairs):
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            
            cond_list.append(cond)
            
            if pooled is None:
                has_pooled_output = False
            
            pooled_list.append(pooled)
            
            if image is not None:
                if image.shape[0] > 0:
                    image_list.append(image[0:1])
                else:
                    blank = torch.zeros((1, 64, 64, 3), dtype=image.dtype, device=image.device)
                    image_list.append(blank)
            else:
                blank = torch.zeros((1, 64, 64, 3))
                image_list.append(blank)

        final_cond = torch.cat(cond_list, dim=0)
        conditioning_extras = {}

        if has_pooled_output:
            try:
                final_pooled = torch.cat(pooled_list, dim=0)
                conditioning_extras["pooled_output"] = final_pooled
            except Exception as e:
                print(f"[CRT Dynamic Prompt Scheduler] Warning: Could not concatenate pooled_outputs, skipping. Error: {e}")

        conditioning_batch = [[final_cond, conditioning_extras]]

        if image_list:
            final_image_batch = torch.cat(image_list, dim=0)
        else:
            final_image_batch = torch.zeros((1, 64, 64, 3))
        
        return (conditioning_batch, batch_count, final_image_batch)