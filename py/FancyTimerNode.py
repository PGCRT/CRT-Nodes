class FancyTimerNode:
    """
    A UI node that displays a real-time timer for the execution pipeline.
    This version is a display-only node with no outputs.
    """
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "hidden": {
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "execute"
    
    OUTPUT_NODE = True
    CATEGORY = "CRT/Utils/UI"

    def execute(self, **kwargs):
        return {}