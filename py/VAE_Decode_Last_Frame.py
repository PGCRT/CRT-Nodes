class CRTVAEDecodeLastFrame:
    """
    Alternative to VAE Decode that decodes only the last item of the latent batch.
    For video latents (batch, channels, frames, height, width) only the last frame
    is decoded, so a single image is returned instead of the whole video.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "The latent to be decoded."}),
                "vae": ("VAE", {"tooltip": "The VAE model used for decoding the latent."}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    OUTPUT_TOOLTIPS = ("The decoded last frame.",)
    FUNCTION = "decode"
    CATEGORY = "CRT/Latent"
    DESCRIPTION = (
        "Decodes latent images back into pixel space, keeping only the last item "
        "(frame) of the batch."
    )

    def decode(self, vae, samples):
        latent = samples["samples"]
        if latent.is_nested:
            # unbind drops the batch dim; restore it so the slicing below matches
            latent = latent.unbind()[-1].unsqueeze(0)
        if latent.ndim == 5:
            latent = latent[-1:, :, -1:]
        else:
            latent = latent[-1:]
        images = vae.decode(latent)
        if len(images.shape) == 5:
            images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
        return (images,)


NODE_CLASS_MAPPINGS = {"CRTVAEDecodeLastFrame": CRTVAEDecodeLastFrame}
NODE_DISPLAY_NAME_MAPPINGS = {"CRTVAEDecodeLastFrame": "VAE Decode Last Frame (CRT)"}