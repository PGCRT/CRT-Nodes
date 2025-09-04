class CLIPTextEncodeFluxMerged:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "clip": ("CLIP", ),
            "prompt": ("STRING", {"multiline": True, "dynamicPrompts": True, "tooltip": "Text prompt for both CLIP-L and T5XXL encoders"}),
            "guidance": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "CRT/Conditioning"
    
    def encode(self, clip, prompt, guidance):
        # Use the same prompt for both CLIP-L and T5XXL
        tokens = clip.tokenize(prompt)
        tokens["t5xxl"] = clip.tokenize(prompt)["t5xxl"]
        return (clip.encode_from_tokens_scheduled(tokens, add_dict={"guidance": guidance}), )