# core/telemetry/memory_starvation_detector.py
"""
MemoryStarvationDetector — identifies memories that have not been
retrieved within the analytics sliding window.

A memory is considered "starved" when it does not appear at all in the
recent retrieval distribution.  A high starvation ratio means most of
the memory store is being ignored, which may indicate that the FAISS
index or scoring weights are overly concentrated on a small cluster.

check() is rate-limited by check_interval to avoid loading all memories
on every query.
"""

import time

from core.telemetry.retrieval_analytics import RetrievalAnalytics
from core.memory.store import load_all_memories
from core.infrastructure.logger import logger


class MemoryStarvationDetector:

    def __init__(
        self,
        analytics: RetrievalAnalytics,
        check_interval: float = 300.0,
    ):
        """
        Parameters
        ----------
        analytics      : RetrievalAnalytics — shared analytics object.
        check_interval : float — minimum seconds between full checks
                         (avoids loading all memories on every query).
        """
        self._analytics       = analytics
        self._check_interval  = check_interval
        self._last_check: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check(self) -> dict:
        """
        Identify starved memories (not in the recent retrieval window).

        Returns
        -------
        dict
            starved_ids     : list[str] — IDs of starved memories
            starved_ratio   : float     — fraction of all memories starved
            total_memories  : int       — total memory count at check time
        """
        now = time.time()

        # Rate limit — return empty result during cooldown.
        if now - self._last_check < self._check_interval:
            return {
                "starved_ids":    [],
                "starved_ratio":  0.0,
                "total_memories": 0,
            }

        all_memories  = load_all_memories()
        all_ids       = {m.id for m in all_memories}
        retrieved_ids = set(self._analytics.distribution().keys())

        starved = [mid for mid in all_ids if mid not in retrieved_ids]
        ratio   = len(starved) / len(all_ids) if all_ids else 0.0

        if starved:
            logger.info(
                f"[StarvationDetector] {len(starved)}/{len(all_ids)} "
                f"memories starved ({ratio:.1%})"
            )

        self._last_check = now
        return {
            "starved_ids":    starved,
            "starved_ratio":  ratio,
            "total_memories": len(all_ids),
        }
