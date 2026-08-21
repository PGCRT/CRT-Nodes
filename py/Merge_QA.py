import json


class MergeQA:
    """Merge a generated phrase back into the empty field of a JSON Q/A string."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qa_string": ("STRING", {"forceInput": True, "tooltip": "Incomplete JSON Q/A pair, e.g. {\"user\":\"...\",\"assistant\":\"\"}"}),
                "phrase": ("STRING", {"forceInput": True, "tooltip": "Generated phrase to insert into the missing Q/A field."}),
                "target": (
                    ["auto", "user", "assistant"],
                    {
                        "default": "auto",
                        "tooltip": "Field to fill. 'auto' fills the empty field based on the original string.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("qa_string",)
    FUNCTION = "merge"
    CATEGORY = "CRT/Text"
    DESCRIPTION = "Inserts a generated phrase into the missing field of a JSON Q/A pair, completing it."

    def merge(self, qa_string, phrase, target):
        try:
            data = json.loads(qa_string)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Merge Q/A (CRT): invalid JSON input: {e}")
            return (qa_string,)

        if not isinstance(data, dict):
            print("[ERROR] Merge Q/A (CRT): JSON input is not an object.")
            return (qa_string,)

        if target == "auto":
            user_empty = not str(data.get("user", "")).strip()
            assistant_empty = not str(data.get("assistant", "")).strip()

            if user_empty and not assistant_empty:
                target = "user"
            elif assistant_empty and not user_empty:
                target = "assistant"
            elif user_empty and assistant_empty:
                print("[WARN] Merge Q/A (CRT): both fields empty, defaulting target to assistant.")
                target = "assistant"
            else:
                print("[WARN] Merge Q/A (CRT): both fields already filled, defaulting target to assistant.")
                target = "assistant"

        data[target] = str(phrase)
        return (json.dumps(data, ensure_ascii=False, separators=(",", ":")),)


NODE_CLASS_MAPPINGS = {"MergeQA": MergeQA}
NODE_DISPLAY_NAME_MAPPINGS = {"MergeQA": "Merge Q/A (CRT)"}
