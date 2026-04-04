import torch


class CRT_DynamicPromptScheduler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = (
        "CONDITIONING",
        "INT",
        "IMAGE",
    )
    RETURN_NAMES = (
        "conditioning",
        "batch_count",
        "image_batch",
    )
    FUNCTION = "schedule"
    CATEGORY = "CRT/Conditioning"

    def schedule(self, clip, **kwargs):
        prompt_image_pairs = []

        # Sort keys to ensure deterministic order based on prompt ID
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

        for idx, (prompt, image) in enumerate(prompt_image_pairs):
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)

            cond_list.append(cond)

            # --- FIX: Ensure pooled_output is always present ---
            if pooled is None:
                # Create zero tensor if CLIP didn't return one
                # cond shape: [Batch, Seq, Dim]. We want [Batch, Dim]
                dim = cond.shape[-1]
                pooled = torch.zeros(cond.shape[0], dim, device=cond.device, dtype=cond.dtype)

            pooled_list.append(pooled)

            # --- Image handling ---
            if image is not None:
                if image.shape[0] > 0:
                    image_list.append(image[0:1])
                else:
                    blank = torch.zeros((1, 64, 64, 3), dtype=image.dtype, device=image.device)
                    image_list.append(blank)
            else:
                # Default blank image if missing
                blank = torch.zeros((1, 64, 64, 3))
                image_list.append(blank)

        # --- Padding logic to match longest sequence ---
        if cond_list:
            max_seq_len = max(c.shape[1] for c in cond_list)

            padded_cond_list = []
            for c in cond_list:
                current_len = c.shape[1]
                if current_len < max_seq_len:
                    pad_amount = max_seq_len - current_len
                    # Pad sequence dim (2nd to last): (dim_last_left, dim_last_right, dim_seq_left, dim_seq_right)
                    c = torch.nn.functional.pad(c, (0, 0, 0, pad_amount))
                padded_cond_list.append(c)

            final_cond = torch.cat(padded_cond_list, dim=0)
        else:
            final_cond = torch.tensor([])

        # --- Construct Final Dictionary ---
        final_pooled = torch.cat(pooled_list, dim=0)

        # Always inject pooled_output
        conditioning_extras = {"pooled_output": final_pooled}

        conditioning_batch = [[final_cond, conditioning_extras]]

        if image_list:
            final_image_batch = torch.cat(image_list, dim=0)
        else:
            final_image_batch = torch.zeros((1, 64, 64, 3))

        return (conditioning_batch, batch_count, final_image_batch)
