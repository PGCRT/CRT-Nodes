# A default list of common English "stop words" to be excluded from replacement by default.
# This list can be modified in the UI.
DEFAULT_EXCLUDE_WORDS = (
    "a an the and but or for nor so yet to in on at with by from of about as through over under "
    "i you he she it we they me him her us them my your his its our their "
    "is are was were be been being have has had do does did will would shall should can could may might must "
    "if then else when where why how what who whom which whose that this these those "
    "not no never always often sometimes usually "
    "one some any every each many much few more most "
    "up down out off "
    "mr mrs ms dr prof etc ie eg"
)

class AdvancedStringReplace:
    """
    An advanced string replacement node that can find multiple words from a given string
    and replace them, while also normalizing whitespace and allowing for an exclusion list.
    """
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source": ("STRING", {"multiline": True, "default": ""}),
                "to_replace": ("STRING", {"multiline": True, "default": ""}),
                "replace_with": ("STRING", {"multiline": False, "default": ""}),
                "exclude_words": ("STRING", {"multiline": True, "default": DEFAULT_EXCLUDE_WORDS}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "replace_words"
    CATEGORY = "CRT/Text"

    def replace_words(self, source: str, to_replace: str, replace_with: str, exclude_words: str):
        if not source or not to_replace:
            return (source,)

        # 1. Prepare the sets of words for fast, case-insensitive lookups.
        words_to_find_set = set(to_replace.lower().split())
        words_to_exclude_set = set(exclude_words.lower().split())

        # 2. Split the source string into words to process them one by one.
        source_words = source.split()
        
        # 3. Build the new list of words based on the replacement and exclusion logic.
        result_words = []
        for word in source_words:
            # Use a cleaned, lowercased version for comparison
            # We remove common punctuation from the end for a better match, e.g., "road." matches "road"
            word_for_comparison = word.lower().strip(".,!?;:")

            # THE CORE LOGIC:
            # Replace the word if it's in the 'to_replace' set AND NOT in the 'exclude' set.
            if word_for_comparison in words_to_find_set and word_for_comparison not in words_to_exclude_set:
                if replace_with:
                    result_words.append(replace_with)
                # If replace_with is empty, we simply don't append anything, effectively deleting the word.
            else:
                # The word was not a target for replacement, so we keep the original word
                # to preserve its capitalization.
                result_words.append(word)

        # 4. Join the result back into a single string with single spaces.
        final_string = " ".join(result_words)

        return (final_string,)