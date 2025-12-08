import comfy.samplers

# --- Wildcard Type for "Universal" Connection ---
class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

ANY = AnyType("*")

class SamplerSchedulerCrawler:

    CATEGORY = "CRT/Utils/Logic & Values"
    
    RETURN_TYPES = (ANY, ANY, "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("sampler_name", "scheduler", "sampler_str", "scheduler_str", "selection_label", "list_info")
    
    FUNCTION = "get_selections"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "scheduler_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "forceInput": True}),
                "sweep_schedulers": ("BOOLEAN", {
                    "default": False, 
                    "label_on": "Sweep ON", 
                    "label_off": "Sweep OFF",
                    # This description appears when hovering the widget
                    "tooltip": "When ENABLED: Automatically increments the Scheduler index only after the Sampler list has completed a full cycle (looped). This allows you to test every possible Sampler/Scheduler combination sequentially by simply incrementing the 'sampler_seed'."
                }),
            }
        }

    def get_selections(self, sampler_seed, scheduler_seed, sweep_schedulers):
        
        # --- Sampler Logic ---
        available_samplers = comfy.samplers.KSampler.SAMPLERS
        if not available_samplers:
            available_samplers = ["euler"]

        len_samplers = len(available_samplers)
        
        # Current Sampler Index
        s_index = sampler_seed % len_samplers
        current_sampler = str(available_samplers[s_index])

        # --- Scheduler Logic ---
        available_schedulers = comfy.samplers.KSampler.SCHEDULERS
        if not available_schedulers:
            available_schedulers = ["normal"]
            
        len_schedulers = len(available_schedulers)

        # Logic: If Sweep is ON, we add the number of full sampler loops to the scheduler seed
        if sweep_schedulers:
            # integer division tells us how many times we've looped through the sampler list
            loops = sampler_seed // len_samplers
            effective_scheduler_seed = scheduler_seed + loops
        else:
            effective_scheduler_seed = scheduler_seed

        # Current Scheduler Index
        sch_index = effective_scheduler_seed % len_schedulers
        current_scheduler = str(available_schedulers[sch_index])

        # --- Create Label ---
        selection_label = f"{current_sampler} - {current_scheduler}"

        # --- Generate List Info ---
        info_lines = []
        info_lines.append(f"Sweep Mode: {'ON' if sweep_schedulers else 'OFF'}")
        
        info_lines.append("\n--- SAMPLERS ---")
        for idx, name in enumerate(available_samplers):
            mark = " <--- SELECTED" if idx == s_index else ""
            info_lines.append(f"{idx}: {name}{mark}")
            
        info_lines.append("\n--- SCHEDULERS ---")
        for idx, name in enumerate(available_schedulers):
            mark = " <--- SELECTED" if idx == sch_index else ""
            info_lines.append(f"{idx}: {name}{mark}")
            
        list_info = "\n".join(info_lines)

        # Return: (Connection Object, Connection Object, Str, Str, Str, Str)
        return (current_sampler, current_scheduler, current_sampler, current_scheduler, selection_label, list_info)

# Registration
NODE_CLASS_MAPPINGS = {
    "SamplerSchedulerCrawler": SamplerSchedulerCrawler
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SamplerSchedulerCrawler": "Sampler & Scheduler Crawler (CRT)"
}