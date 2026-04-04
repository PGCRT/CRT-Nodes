import folder_paths
import comfy.utils


class LoraLoaderStr:

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        file_list = folder_paths.get_filename_list("loras")
        file_list.insert(0, "None")
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "switch": (["Of", "On"],),
                "lora_name": (file_list,),
                "strength_model": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "strength_clip": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.1}),
                "include_strength": (["Yes", "No"],),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "STRING")
    FUNCTION = "load_lora"
    CATEGORY = "CRT/LoRA"

    def load_lora(self, model, clip, switch, lora_name, strength_model, strength_clip, include_strength):
        if strength_model == 0 and strength_clip == 0:
            return (model, clip, "No LoRA Loaded")

        if switch == "Of" or lora_name == "None":
            return (model, clip, "No LoRA Loaded")

        lora_path = folder_paths.get_full_path("loras", lora_name)
        lora = None

        if self.loaded_lora is not None:
            if self.loaded_lora[0] == lora_path:
                lora = self.loaded_lora[1]
            else:
                del self.loaded_lora

        if lora is None:
            try:
                lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                self.loaded_lora = (lora_path, lora)
            except Exception as e:
                print(f"Error loading LoRA from {lora_path}: {e}")
                return (model, clip, "Error loading LoRA")

        try:
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model, clip, lora, strength_model, strength_clip)

            output_string = f"LoRA: {lora_name}"
            if include_strength == "Yes":
                output_string += f" (Strength Model: {strength_model}, Strength Clip: {strength_clip})"

            return (model_lora, clip_lora, output_string)
        except Exception as e:
            print(f"Error applying LoRA: {e}")
            return (model, clip, "Error applying LoRA")
