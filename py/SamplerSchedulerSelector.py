import comfy.samplers

class SamplerSchedulerSelector:

    CATEGORY = "CRT/Utils/Logic & Values"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("sampler_name", "scheduler")
    
    FUNCTION = "get_selections"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS,)
            }
        }

    def get_selections(self, sampler_name, scheduler):
        return (sampler_name, scheduler)