import io
import numpy as np
from PIL import Image
import server as comfy_server


class SaveJpegWebsocket:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"images": ("IMAGE",)}}

    RETURN_TYPES = ()
    FUNCTION = "save_images"
    OUTPUT_NODE = True
    CATEGORY = "image"

    def save_images(self, images):
        s = comfy_server.PromptServer.instance
        for image in images:
            i = 255. * image.cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            # BinaryEventTypes.UNENCODED_PREVIEW_IMAGE = 2
            # Passing "JPEG" tells ComfyUI to encode as JPEG before sending over websocket
            s.send_sync(2, ("JPEG", img, None), s.client_id)
        return ()


NODE_CLASS_MAPPINGS = {"SaveJpegWebsocket": SaveJpegWebsocket}
NODE_DISPLAY_NAME_MAPPINGS = {"SaveJpegWebsocket": "Save JPEG Websocket"}
