class TextRowsCrawl:
    """Select one row from a multiline string using ComfyUI's native seed control."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True, "tooltip": "Multiline text to select a row from."}),
                "seed": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "control_after_generate": True,
                        "tooltip": "Selects the row as seed % number_of_rows. Use ComfyUI's seed control (fixed / increment / decrement / randomize) to crawl.",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("selected_row",)
    FUNCTION = "crawl_rows"
    CATEGORY = "CRT/Text"
    DESCRIPTION = "Outputs one row from a multiline string. The selected row is seed % row_count, driven by ComfyUI's native seed control."

    def _split_rows(self, text):
        if not text:
            return []
        return [line.strip() for line in text.splitlines() if line.strip() != ""]

    def crawl_rows(self, text, seed):
        rows = self._split_rows(text)
        if not rows:
            print("[WARN] Text Rows Crawl (CRT): input text contains no rows.")
            return ("",)

        try:
            seed_int = int(seed)
        except Exception:
            seed_int = 0
        # Guard against NaN / inf that can appear after a frontend refresh
        # when the widget value exceeds JS safe integer handling.
        if not isinstance(seed_int, int) or seed_int != seed_int:  # NaN check
            seed_int = 0
        selected_index = seed_int % len(rows)
        return (rows[selected_index],)


NODE_CLASS_MAPPINGS = {"TextRowsCrawl": TextRowsCrawl}
NODE_DISPLAY_NAME_MAPPINGS = {"TextRowsCrawl": "Text Rows Crawl (CRT)"}
