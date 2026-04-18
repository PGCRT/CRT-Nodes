import inspect
import torch
import folder_paths

from .Face_Enhancement_Pipeline_With_Injection import (
    UltraDetector,
    inference_segm,
    create_segmasks,
    apply_controlnet_advanced,
    tensor2pil,
    _SimpleSeg,
    colored_print,
    Colors,
)
from .SEGS_Enhancer_Multi import FaceEnhancementWithInjectionSEGS


class Flux1CnetUltralyticMulti:
    def __init__(self):
        self.detectors = {}
        self.segs_enhancer = FaceEnhancementWithInjectionSEGS()
        self._segs_execute_sig = inspect.signature(self.segs_enhancer.execute)
        colored_print("Flux1 Cnet Ultralytics Multi initialized!", Colors.HEADER)

    @classmethod
    def INPUT_TYPES(s):
        try:
            segm_files = folder_paths.get_filename_list("ultralytics_segm")
        except Exception:
            segm_files = []

        base = FaceEnhancementWithInjectionSEGS.INPUT_TYPES()
        required = {}
        for k, v in base["required"].items():
            if k == "segs":
                required["face_segm_model"] = (["segm/" + x for x in segm_files],)
                required["segm_threshold"] = (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                )
                required["grow_crop"] = (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 512,
                        "step": 1,
                        "tooltip": "Expands each detected crop region by this many pixels before enhancement",
                    },
                )
                continue
            if k == "edit_model_flux2klein":
                continue
            if k == "positive":
                required[k] = v
                required["control_net"] = ("CONTROL_NET",)
                continue
            if k == "seed":
                required[k] = v
                required["controlnet_strength"] = (
                    "FLOAT",
                    {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05},
                )
                required["control_end"] = (
                    "FLOAT",
                    {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01},
                )
                continue
            required[k] = v

        return {
            "required": required,
            "optional": {
                "negative": ("CONDITIONING",),
                "vae": ("VAE",),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = (
        "enhanced_image",
        "enhanced_face",
        "cropped_face_before",
        "enhanced_face_alpha",
        "base_face_alpha",
    )
    FUNCTION = "execute"
    CATEGORY = "CRT/Sampling"

    def execute(self, **kwargs):
        image = kwargs["image"]
        face_segm_model = kwargs["face_segm_model"]
        segm_threshold = kwargs["segm_threshold"]
        grow_crop = int(kwargs.get("grow_crop", 0))
        control_net = kwargs["control_net"]
        controlnet_strength = kwargs["controlnet_strength"]
        control_end = kwargs["control_end"]

        positive = kwargs["positive"]
        negative = kwargs.get("negative", None)
        if negative is None:
            negative = []
            for t in positive:
                negative.append([torch.zeros_like(t[0]), t[1].copy()])

        vae = kwargs.get("vae", None)
        cnet_positive, cnet_negative = apply_controlnet_advanced(
            positive,
            negative,
            control_net,
            image,
            controlnet_strength,
            0.0,
            control_end,
            vae,
        )

        segm_filename_only = face_segm_model.split("/")[-1]
        segm_full_path = folder_paths.get_full_path("ultralytics_segm", segm_filename_only)
        if face_segm_model not in self.detectors:
            self.detectors[face_segm_model] = UltraDetector(segm_full_path, "segm")
        segm_detector = self.detectors[face_segm_model]

        pil_image = tensor2pil(image[0])
        detected_results = inference_segm(segm_detector.model, pil_image, segm_threshold)
        segmasks = create_segmasks(detected_results)

        h, w = image.shape[1:3]
        seg_entries = []
        for bbox, mask, _confidence in segmasks:
            x1, y1, x2, y2 = [int(round(v)) for v in bbox]
            if grow_crop > 0:
                x1 -= grow_crop
                y1 -= grow_crop
                x2 += grow_crop
                y2 += grow_crop
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            if x2 <= x1 or y2 <= y1:
                continue

            cropped_mask = mask[y1:y2, x1:x2]
            if cropped_mask.size == 0:
                continue
            seg_entries.append(_SimpleSeg([x1, y1, x2, y2], cropped_mask))

        segs = (None, seg_entries)

        call_kwargs = {}
        for name in self._segs_execute_sig.parameters:
            if name == "self":
                continue
            if name == "segs":
                call_kwargs[name] = segs
            elif name == "edit_model_flux2klein":
                call_kwargs[name] = False
            elif name == "positive":
                call_kwargs[name] = cnet_positive
            elif name == "negative":
                call_kwargs[name] = cnet_negative
            elif name in kwargs:
                call_kwargs[name] = kwargs[name]

        return self.segs_enhancer.execute(**call_kwargs)


NODE_CLASS_MAPPINGS = {"Flux1CnetUltralyticMulti": Flux1CnetUltralyticMulti}
NODE_DISPLAY_NAME_MAPPINGS = {"Flux1CnetUltralyticMulti": "Flux1 Cnet Ultralytics Multi (CRT)"}
