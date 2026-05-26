"""
Memory storage pre-filter.

This is the first gate before any memory hits the extractor or store.
It rejects content that is structurally unfit regardless of semantic value:
too short, a bare question, or obviously empty.

The extractor.py has its own filler detection; this filter is intentionally
kept lightweight so the two layers don't overlap in responsibility.
"""


def should_store(text: str) -> bool:
    """
    Return True if *text* is worth sending to the extraction pipeline.

    Rejects
    -------
    - Strings shorter than 5 characters
    - Bare questions (ends with ?, no self-disclosure keywords)
    - Whitespace-only strings
    """
    text = text.strip()

    if len(text) < 5:
        return False

    if not text:
        return False

    # Pure question with no embedded self-disclosure — not worth extracting.
    if text.endswith("?"):
        disclosure_keywords = ("i ", "i'm", "i am", "my ", "i've", "i have")
        if not any(kw in text.lower() for kw in disclosure_keywords):
            return False

    return True