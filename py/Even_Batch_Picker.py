import torch


class CRTEvenBatchPicker:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "count": ("INT", {"default": 2, "min": 1, "max": 4096, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "pick"
    CATEGORY = "CRT/Video"

    def pick(self, images: torch.Tensor, count: int):
        batch_size = images.shape[0]

        if batch_size <= 1 or count >= batch_size:
            return (images,)

        if count == 1:
            return (images[0:1],)

        indices = [int((i * (batch_size - 1)) / (count - 1)) for i in range(count)]
        selected = images[indices]
        return (selected,)


NODE_CLASS_MAPPINGS = {"CRTEvenBatchPicker": CRTEvenBatchPicker}
NODE_DISPLAY_NAME_MAPPINGS = {"CRTEvenBatchPicker": "Even Batch Picker (CRT)"}
