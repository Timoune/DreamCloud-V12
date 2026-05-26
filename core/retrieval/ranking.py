"""
Multi-factor, type-aware memory ranking.

Changes from v2
---------------
* TYPE_RETRIEVAL_WEIGHT from schema.py is applied as a multiplier on the
  final score so identity/preference memories naturally surface above
  general/fact memories when scores are otherwise close.

* reliability is folded into the fallback score path (used outside engine).

* Exported rank_memories() is the single canonical sort used by engine.py.
"""

import time

from core.memory.schema import TYPE_RETRIEVAL_WEIGHT


def rank_memories(memories: list, now: float = None) -> list:
    """
    Sort memories by final_score (engine path) or a blended fallback.

    Applies TYPE_RETRIEVAL_WEIGHT on top of whichever base score is present.

    Parameters
    ----------
    memories : list[Memory]
    now      : unix timestamp — defaults to time.time()

    Returns
    -------
    list[Memory] sorted best-first
    """
    if now is None:
        now = time.time()

    def _score(m) -> float:
        type_weight = TYPE_RETRIEVAL_WEIGHT.get(m.type, 1.0)

        base = m.metadata.get("final_score")
        if base is not None:
            return base * type_weight

        # Fallback for paths that bypass engine scoring (e.g. unit tests)
        age_days = (now - m.timestamp) / 86400
        recency  = max(0.0, 1.0 - age_days / 30)
        raw      = (0.6 * m.importance) + (0.3 * recency) + (0.1 * m.reliability)
        return raw * type_weight

    return sorted(memories, key=_score, reverse=True)