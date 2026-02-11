import folder_paths
import os


class WanVideoLoraSelectMultiImproved:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "lora_batch_config": ("STRING", {"default": "", "multiline": True}),
                "merge_loras": ("BOOLEAN", {"default": True}),
                "low_mem_load": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "blocks": ("SELECTEDBLOCKS",),
            },
        }

    RETURN_TYPES = ("WANVIDLORA", "WANVIDLORA")
    RETURN_NAMES = ("high_lora_stack", "low_lora_stack")
    FUNCTION = "get_lora_stack"
    CATEGORY = "WanVideo/Loaders"

    def get_lora_stack(self, lora_batch_config, merge_loras, low_mem_load, blocks={}):
        if not merge_loras:
            low_mem_load = False  # Unmerged LoRAs don't need low_mem_load

        high_lora_stack = []
        low_lora_stack = []

        if not lora_batch_config:
            return (high_lora_stack, low_lora_stack)

        # Parse config string from JS
        # Format expected: high,hstr,low,lstr,on|...§true

        if "§" not in lora_batch_config:
            return (high_lora_stack, low_lora_stack)

        config_data, valid = lora_batch_config.split("§")

        if valid != "true":
            return (high_lora_stack, low_lora_stack)

        rows = config_data.split("|")

        for row in rows:
            if not row:
                continue

            try:
                parts = row.split(",")
                if len(parts) < 5:
                    continue

                high_name = parts[0]
                high_str = float(parts[1])
                low_name = parts[2]
                low_str = float(parts[3])
                enabled = parts[4] == "true"

                if not enabled:
                    continue

                # Handle "none" string from JS or empty strings
                if high_name == "none":
                    high_name = ""
                if low_name == "none":
                    low_name = ""

                # Add HIGH lora to stack with proper format
                if high_name and high_str != 0.0:
                    high_lora_stack.append(
                        {
                            "path": folder_paths.get_full_path_or_raise("loras", high_name),
                            "strength": round(high_str, 4),
                            "name": os.path.splitext(high_name)[0],
                            "blocks": blocks.get("selected_blocks", {}),
                            "layer_filter": blocks.get("layer_filter", ""),
                            "low_mem_load": low_mem_load,
                            "merge_loras": merge_loras,
                        }
                    )

                # Add LOW lora to stack with proper format
                if low_name and low_str != 0.0:
                    low_lora_stack.append(
                        {
                            "path": folder_paths.get_full_path_or_raise("loras", low_name),
                            "strength": round(low_str, 4),
                            "name": os.path.splitext(low_name)[0],
                            "blocks": blocks.get("selected_blocks", {}),
                            "layer_filter": blocks.get("layer_filter", ""),
                            "low_mem_load": low_mem_load,
                            "merge_loras": merge_loras,
                        }
                    )

            except ValueError as e:
                print(f"WanVideoLoraSelectMultiImproved: Error parsing row '{row}': {e}")
                continue
            except Exception as e:
                print(f"WanVideoLoraSelectMultiImproved: Error processing LoRA '{high_name or low_name}': {e}")
                continue

        return (high_lora_stack, low_lora_stack)


NODE_CLASS_MAPPINGS = {"WanVideoLoraSelectMultiImproved": WanVideoLoraSelectMultiImproved}

NODE_DISPLAY_NAME_MAPPINGS = {"WanVideoLoraSelectMultiImproved": "Wan Video Multi-LoRA Select"}
