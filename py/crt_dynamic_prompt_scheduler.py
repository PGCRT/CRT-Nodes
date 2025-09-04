import torch

class CRT_DynamicPromptScheduler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": ("CLIP",),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "INT")
    RETURN_NAMES = ("conditioning", "batch_count")
    FUNCTION = "schedule"
    CATEGORY = "CRT/Conditioning"

    def schedule(self, clip, **kwargs):
        """
        Processes all incoming text prompts, encodes them, and returns a single
        batched conditioning tensor, gracefully handling CLIP models that may
        not provide a pooled_output.
        """
        prompts = []
        for key, value in sorted(kwargs.items()):
            if key.startswith('prompt_') and isinstance(value, str) and value.strip():
                prompts.append(value)

        if not prompts:
            prompts.append("")

        batch_count = len(prompts)

        cond_list = []
        pooled_list = []
        # Flag to check if the CLIP model is providing a pooled output.
        has_pooled_output = True

        for prompt in prompts:
            tokens = clip.tokenize(prompt)
            cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
            
            cond_list.append(cond)
            
            # If any prompt results in a missing pooled output, we cannot use it for the batch.
            if pooled is None:
                has_pooled_output = False
            
            # We still append it to keep the list lengths consistent, but we won't use it if the flag is false.
            pooled_list.append(pooled)

        # --- Build the Final Conditioning ---
        
        # The main conditioning tensor is always present.
        final_cond = torch.cat(cond_list, dim=0)
        
        # The dictionary for extra data, like the pooled output.
        conditioning_extras = {}

        # ** THE FIX **
        # Only add the 'pooled_output' to the dictionary if the CLIP model actually provided it.
        if has_pooled_output:
            try:
                final_pooled = torch.cat(pooled_list, dim=0)
                conditioning_extras["pooled_output"] = final_pooled
            except Exception as e:
                # This is a fallback safety net in case of an unexpected error.
                print(f"[CRT Dynamic Prompt Scheduler] Warning: Could not concatenate pooled_outputs, skipping. Error: {e}")

        # The final structure must be a list containing one list: [[main_tensor, {extras_dict}]]
        conditioning_batch = [[final_cond, conditioning_extras]]

        return (conditioning_batch, batch_count)