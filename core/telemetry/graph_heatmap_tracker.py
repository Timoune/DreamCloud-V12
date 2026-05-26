# core/telemetry/graph_heatmap_tracker.py
"""
GraphHeatmapTracker — tracks graph-edge traversal frequency during
retrieval expansion.  A small set of edges dominating traversal is
another signal of cognitive concentration.
"""
import time
from collections import defaultdict
from core.infrastructure.logger import logger


class GraphHeatmapTracker:

    def __init__(self, window_seconds: float = 3600.0):
        # (src_id, dst_id) → list of timestamps
        self._traversals: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._window = window_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def record_traversal(self, src_id: str, dst_id: str) -> None:
        """Call for each edge traversed during graph expansion."""
        now = time.time()
        key = (src_id, dst_id)
        self._traversals[key].append(now)

    def record_expansion(self, visited_ids: list[str]) -> None:
        """
        Convenience: records all consecutive pairs in the BFS visited set
        as traversed edges.
        """
        for i in range(len(visited_ids) - 1):
            self.record_traversal(visited_ids[i], visited_ids[i + 1])

    def top_edges(self, n: int = 10) -> list[tuple[tuple[str, str], int]]:
        """Most-traversed edges in the window."""
        self._expire()
        ranked = sorted(self._traversals.items(), key=lambda x: len(x[1]), reverse=True)
        return [(edge, len(events)) for edge, events in ranked[:n]]

    def edge_entropy(self) -> float:
        """
        Shannon entropy of the edge-traversal distribution.
        Low entropy → few edges dominate graph walks.
        """
        import math
        self._expire()
        total = sum(len(v) for v in self._traversals.values())
        if total <= 1:
            return 0.0
        entropy = 0.0
        for events in self._traversals.values():
            p = len(events) / total
            if p > 0:
                entropy -= p * math.log2(p)
        return entropy

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _expire(self, now: float | None = None) -> None:
        if now is None:
            now = time.time()
        cutoff = now - self._window
        for key in list(self._traversals.keys()):
            self._traversals[key] = [t for t in self._traversals[key] if t >= cutoff]
            if not self._traversals[key]:
                del self._traversals[key]