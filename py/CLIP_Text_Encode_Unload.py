import comfy.model_management


class CRTCLIPTextEncode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": True,
                        "tooltip": "The text to be encoded.",
                    },
                ),
                "clip": (
                    "CLIP",
                    {"tooltip": "The CLIP model used for encoding the text."},
                ),
                "keep_clip_loaded": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": (
                            "Keep CLIP in VRAM after encoding. Disable to offload "
                            "CLIP before main model inference."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("CONDITIONING",)
    OUTPUT_TOOLTIPS = (
        "Conditioning containing the embedded text used to guide the diffusion model.",
    )
    FUNCTION = "encode"
    CATEGORY = "CRT/Conditioning"
    DESCRIPTION = (
        "Encodes a text prompt like the native CLIP Text Encode node, with an "
        "option to offload CLIP after conditioning is generated."
    )

    def encode(self, clip, text, keep_clip_loaded):
        if clip is None:
            raise RuntimeError(
                "ERROR: clip input is invalid: None\n\n"
                "If the clip is from a checkpoint loader node, your checkpoint "
                "does not contain a valid clip or text encoder model."
            )

        tokens = clip.tokenize(text)
        conditioning = clip.encode_from_tokens_scheduled(tokens)

        if not keep_clip_loaded:
            comfy.model_management.unload_model_and_clones(clip.patcher)

        return (conditioning,)


NODE_CLASS_MAPPINGS = {"CRTCLIPTextEncode": CRTCLIPTextEncode}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRTCLIPTextEncode": "CLIP Text Encode + Unload (CRT)"
}
