"""
Memory extraction pipeline.

Instead of storing raw user sentences verbatim, this module asks the local
LLM to extract structured, high-value memory units from each user turn.

Extracted memory types
----------------------
preference  — likes, dislikes, favourites       ("I love jazz")
identity    — facts about the user              ("I'm a software engineer")
fact        — world/topic knowledge shared      ("Python uses indentation")
goal        — intentions, plans                 ("I want to learn Rust")
episodic    — specific past events              ("I went to Tokyo last year")
emotional   — emotional states/reactions        ("That conversation upset me")
general     — anything that doesn't fit cleanly (fallback)

Returns an empty list for filler, pure questions, and short acknowledgements
so nothing wasteful reaches the memory store.
"""

import json
import re

from core.runtime.llama_runner import run_llama


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_EXTRACTION_PROMPT = """\
You extract memory-worthy facts from user messages.

Return a JSON array of objects. Each object must have exactly two keys:
  "type"    : one of  preference | identity | fact | goal | episodic | emotional | general
  "content" : a clean, standalone statement written in third person about the user.

Examples
--------
Input:  "I really love jazz music"
Output: [{{"type": "preference", "content": "User loves jazz music"}}]

Input:  "I'm a software engineer working in Berlin"
Output: [
  {{"type": "identity", "content": "User is a software engineer"}},
  {{"type": "identity", "content": "User lives and works in Berlin"}}
]

Input:  "ok"
Output: []

Rules
-----
- Only extract genuinely informative, memorable statements.
- Skip filler, greetings, short acknowledgements, and bare questions.
- A question that contains a self-disclosure ("what's my name, I keep forgetting")
  should still extract the disclosure.
- Keep each content string concise — one sentence maximum.
- Return ONLY the JSON array. No explanation, no markdown fences, no preamble.

User message: {user_input}
""".strip()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_memories(user_input: str) -> list:
    """
    Run the LLM extraction pass on a raw user message.

    Returns
    -------
    list of dicts: [{"type": str, "content": str}, ...]
    Returns [] if nothing is extractable or the LLM call fails.
    """
    if not _worth_extracting(user_input):
        return []

    prompt = _EXTRACTION_PROMPT.format(user_input=user_input.strip())

    try:
        raw = run_llama(prompt).strip()
        return _parse_extractions(raw)
    except Exception as e:
        print(f"[Extractor WARN] LLM call failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_FILLER_RE = re.compile(
    r"^(ok|okay|yes|no|sure|thanks|thank you|hi|hello|bye|goodbye|cool|"
    r"great|got it|alright|right|yep|nope|yup|hmm+|uh+|er+|ah+)[.!?]?$",
    re.IGNORECASE,
)

# Self-disclosure keywords — if any appear we run extraction even on questions.
_SELF_DISCLOSURE = re.compile(
    r"\b(i |i'm|i am|my |i've|i have|i like|i want|i love|i hate|i feel)\b",
    re.IGNORECASE,
)

_MIN_LENGTH = 8   # characters


def _worth_extracting(text: str) -> bool:
    """Cheap pre-filter before spending an LLM call."""
    text = text.strip()

    if len(text) < _MIN_LENGTH:
        return False

    if _FILLER_RE.match(text):
        return False

    # Pure question with no self-disclosure embedded
    if text.endswith("?") and not _SELF_DISCLOSURE.search(text):
        return False

    return True


_VALID_TYPES = {
    "preference", "identity", "fact",
    "goal", "episodic", "emotional", "general",
}


def _parse_extractions(raw: str) -> list:
    """
    Parse the LLM JSON response into a list of extraction dicts.
    Tolerates minor formatting issues (stray backticks, preamble text).
    """
    # Strip markdown fences the model sometimes adds despite instructions
    raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()

    # Find the outermost [...] block to tolerate any preamble
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if not match:
        return []

    try:
        items = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue

        mem_type = item.get("type", "general").strip().lower()
        content  = item.get("content", "").strip()

        if not content:
            continue

        if mem_type not in _VALID_TYPES:
            mem_type = "general"

        results.append({"type": mem_type, "content": content})

    return results