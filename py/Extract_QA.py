import json


class ExtractQA:
    """Extract the user or assistant phrase and build an LLM instruction for it."""

    # Pre-made instructions for Unsloth Studio Bridge — the extracted phrase
    # is appended so the model can either answer the question or create one.
    INSTRUCTION_ANSWER = (
        "Answer the following question accurately and concisely:\n\n"
        "Question: {phrase}\n\nAnswer:"
    )
    INSTRUCTION_CREATE_QUESTION = (
        "Write a clear, concise question that would be directly answered by the following text:\n\n"
        "Answer: {phrase}\n\nQuestion:"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "qa_string": ("STRING", {"forceInput": True, "tooltip": "JSON string in the form {\"user\":\"...\",\"assistant\":\"...\"}"}),
                "target": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "label_on": "user (question)",
                        "label_off": "assistant (answer)",
                        "tooltip": "ON = extract the user question and build an 'answer it' instruction. OFF = extract the assistant answer and build a 'create the question' instruction.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("phrase", "instruction")
    FUNCTION = "extract"
    CATEGORY = "CRT/Text"
    DESCRIPTION = (
        "Extracts the user or assistant phrase from a JSON Q/A pair and "
        "builds a ready-to-use LLM instruction for Unsloth Studio Bridge."
    )

    def extract(self, qa_string, target=True, **kwargs):
        # Backward compat: old workflows used target as "user"/"assistant" combo;
        # briefly used target_is_question as the boolean key.
        raw = kwargs.get("target_is_question", target)
        if isinstance(raw, str):
            is_question = raw == "user"
        else:
            is_question = bool(raw)

        target = "user" if is_question else "assistant"

        try:
            data = json.loads(qa_string)
        except json.JSONDecodeError as e:
            print(f"[ERROR] Extract Q/A (CRT): invalid JSON input: {e}")
            return ("", "")

        if not isinstance(data, dict):
            print("[ERROR] Extract Q/A (CRT): JSON input is not an object.")
            return ("", "")

        phrase = str(data.get(target, ""))

        if is_question:
            instruction = self.INSTRUCTION_ANSWER.format(phrase=phrase)
        else:
            instruction = self.INSTRUCTION_CREATE_QUESTION.format(phrase=phrase)

        return (phrase, instruction)


NODE_CLASS_MAPPINGS = {"ExtractQA": ExtractQA}
NODE_DISPLAY_NAME_MAPPINGS = {"ExtractQA": "Extract Q/A (CRT)"}
