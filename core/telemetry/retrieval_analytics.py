# core/telemetry/retrieval_analytics.py
"""
RetrievalAnalytics — records every retrieval event and maintains a
frequency distribution across memory IDs.  The distribution is consumed
by ClusterEntropyMonitor to compute Shannon entropy.
"""
import time
from collections import defaultdict
from core.infrastructure.logger import logger


class RetrievalAnalytics:
    """
    Stores per-memory retrieval counts with a sliding window so the
    entropy signal reflects recent cognitive behaviour rather than the
    entire history.
    """

    def __init__(self, window_seconds: float = 3600.0):
        # {memory_id: [(timestamp, count_in_that_retrieval), ...]}
        self._history: dict[str, list[tuple[float, int]]] = defaultdict(list)
        self._window = window_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record(self, memory_ids: list[str]) -> None:
        """
        Call once per retrieval operation.  *memory_ids* are the IDs
        returned by the retrieval pipeline (after graph expansion).
        """
        now = time.time()
        for mid in memory_ids:
            self._history[mid].append((now, 1))
        self._expire(now)

    def distribution(self) -> dict[str, float]:
        """
        Returns a probability distribution over memory IDs:
            P(id) = count(id) / total_retrievals

        Only considers entries within the sliding window.
        """
        self._expire()
        total = sum(len(events) for events in self._history.values())
        if total == 0:
            return {}
        return {mid: len(events) / total for mid, events in self._history.items()}

    def top_retrieved(self, n: int = 10) -> list[tuple[str, int]]:
        """Memory IDs with the highest retrieval count in the window."""
        self._expire()
        ranked = sorted(self._history.items(), key=lambda x: len(x[1]), reverse=True)
        return [(mid, len(events)) for mid, events in ranked[:n]]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _expire(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        cutoff = now - self._window
        for mid in list(self._history.keys()):
            self._history[mid] = [e for e in self._history[mid] if e[0] >= cutoff]
            if not self._history[mid]:
                del self._history[mid]