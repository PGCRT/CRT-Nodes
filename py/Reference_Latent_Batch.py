import hashlib

import torch
import node_helpers


class ReferenceLatentBatch:
    """
    Drop-in replacement for chaining N × ReferenceLatent nodes.

    The original node wraps the entire latent["samples"] tensor [N,C,H,W] as a
    single entry in the reference_latents list.  Downstream code computes token
    counts per list entry from spatial dims only, so a batch of N is treated as
    one image → N-1 frames are silently ignored and KV-cache slicing is wrong.

    This node iterates over the batch dimension and appends each [1,C,H,W] slice
    individually, making a batch of N exactly equivalent to chaining N nodes.

    Permutation invariance is guaranteed by:
      1. Sorting frames by SHA-1 of their full content (collision-free for any
         set of distinct latents, unlike a partial-value key).
      2. Calling .contiguous() after reindexing so every permutation produces a
         tensor with identical memory strides — eliminating the nondeterministic
         CUDA results that arise when the same logical values sit at different
         memory offsets.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "latent":       ("LATENT",),
            }
        }

    RETURN_TYPES  = ("CONDITIONING",)
    FUNCTION      = "execute"
    CATEGORY      = "CRT/Latent"
    DESCRIPTION   = (
        "Accepts a batch of latent images and registers each frame as a separate "
        "reference, equivalent to chaining N ReferenceLatent nodes. "
        "Frames are sorted by full-content hash before registration so that any "
        "permutation of the same images produces byte-identical conditioning."
    )

    @staticmethod
    def _content_hash(t: torch.Tensor) -> bytes:
        # Convert to float32 first: ensures float16 and bfloat16 tensors that
        # represent the same values hash identically regardless of storage dtype.
        return hashlib.sha1(t.cpu().to(torch.float32).numpy().tobytes()).digest()

    def execute(self, conditioning, latent):
        samples = latent["samples"]          # [N, C, H, W]
        N = samples.shape[0]

        if N > 1:
            perm    = sorted(range(N), key=lambda i: self._content_hash(samples[i]))
            idx     = torch.tensor(perm, device=samples.device)
            # .contiguous() forces a fresh allocation with standard C-strides so
            # CUDA sees identical memory layout regardless of the original order.
            samples = samples[idx].contiguous()

        for i in range(N):
            frame = samples[i : i + 1]       # [1, C, H, W]  — keeps batch dim
            conditioning = node_helpers.conditioning_set_values(
                conditioning,
                {"reference_latents": [frame]},
                append=True,
            )
        return (conditioning,)


NODE_CLASS_MAPPINGS = {
    "ReferenceLatentBatch": ReferenceLatentBatch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "ReferenceLatentBatch": "Reference Latent (Batch)",
}
