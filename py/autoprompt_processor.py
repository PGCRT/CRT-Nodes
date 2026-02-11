import re


class AutopromptProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "autoprompt": (
                    "STRING",
                    {"forceInput": True, "tooltip": "Connect a string input containing the autoprompt text to process"},
                ),
                # CORRECTED LINE: The key must be a valid Python identifier
                # and match the function parameter name 'custom_prefix'.
                "custom_prefix": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Custom Prompt Prefix / LoRA TriggerWord: Text to add before the processed autoprompt. Leave empty to skip prefix.",
                    },
                ),
                "replacements": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "old_word|new_word\nanother_word|replacement",
                        "tooltip": "Word replacements, one per line using format: old_word|new_word",
                    },
                ),
            }
        }

    CATEGORY = "CRT/Text"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "process_autoprompt"

    def process_autoprompt(self, autoprompt, custom_prefix, replacements):
        """
        Process the autoprompt by replacing words and adding custom prefix.
        """

        processed_text = autoprompt

        # Parse and apply replacements
        if replacements.strip():
            for line in replacements.strip().split('\n'):
                line = line.strip()
                if '|' in line and not line.startswith('#'):
                    parts = line.split('|', 1)
                    if len(parts) == 2:
                        old_word = parts[0].strip()
                        new_word = parts[1].strip()
                        if old_word:
                            # Use whole words only with case insensitive matching
                            pattern = r'\b' + re.escape(old_word) + r'\b'
                            processed_text = re.sub(pattern, new_word, processed_text, flags=re.IGNORECASE)

        # Combine custom prefix with processed autoprompt
        if custom_prefix.strip():
            if processed_text.strip():
                final_text = custom_prefix.strip() + " " + processed_text.strip()
            else:
                final_text = custom_prefix.strip()
        else:
            final_text = processed_text

        return (final_text,)
