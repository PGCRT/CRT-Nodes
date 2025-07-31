import torch

class EnableLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "enable": ("BOOLEAN", {"default": True}),
            },
        }

    CATEGORY = "CRT/Latent"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "enable_latent"

    def enable_latent(self, latent, enable):
        print(f"📌 EnableLatent: Starting execution, enable={enable}")

        # If enable is False, return None
        if not enable:
            print(f"⚠️ enable is False. Returning None.")
            return (None,)

        # Validate latent input
        if latent is None:
            print(f"⚠️ Received None as latent input. Returning None.")
            return (None,)

        # Ensure latent is in the correct format (dict with samples or tensor)
        if isinstance(latent, dict) and "samples" in latent and isinstance(latent["samples"], torch.Tensor):
            print(f"📌 Passing through latent with tensor shape: {latent['samples'].shape}")
            return (latent,)
        elif isinstance(latent, torch.Tensor):
            print(f"📌 Passing through raw tensor latent with shape: {latent.shape}")
            return ({"samples": latent},)
        else:
            print(f"❌ Invalid latent input type: {type(latent)}. Expected dict with 'samples' or torch.Tensor. Returning None.")
            return (None,)

    @classmethod
    def IS_CHANGED(cls, latent, enable, **kwargs):
        # Return a unique value to trigger re-execution if enable changes
        return enable

    @classmethod
    def VALIDATE_INPUTS(cls, enable, **kwargs):
        # No specific validation needed for boolean enable
        return True

NODE_CLASS_MAPPINGS = {
    "EnableLatent": EnableLatent
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "EnableLatent": "Enable Latent (CRT)"
}