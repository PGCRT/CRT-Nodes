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
                # These are required for the UI to save its state
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    # By setting RETURN_TYPES to an empty tuple, we remove all output pins.
    RETURN_TYPES = ()
    FUNCTION = "execute"
    
    # This remains True to keep the node's visual style as a terminal node.
    OUTPUT_NODE = True
    CATEGORY = "CRT/Utils/UI"

    def execute(self, **kwargs):
        # This node does nothing on the backend; all logic is in the UI.
        # We return an empty dictionary as there are no outputs.
        return {}