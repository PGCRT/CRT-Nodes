class CRT_OpenCodeGoThinkingDisplay:
    """Standalone live viewer for the OpenCode Go Bridge.

    No inputs and no outputs: drop it anywhere in the workflow and it shows the
    bridge's thinking process as it streams, plus live tok/s.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "unique_id": "UNIQUE_ID",
                "prompt": "PROMPT",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "display"
    OUTPUT_NODE = True
    CATEGORY = "CRT/LLM"
    DESCRIPTION = (
        "Standalone live viewer for the OpenCode Go Bridge. Shows the thinking "
        "process in real time with tok/s. No inputs or outputs."
    )

    def display(self, unique_id=None, prompt=None):
        return {}


NODE_CLASS_MAPPINGS = {
    "CRT_OpenCodeGoThinkingDisplay": CRT_OpenCodeGoThinkingDisplay,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CRT_OpenCodeGoThinkingDisplay": "OpenCode Go Thinking Display (CRT)",
}
