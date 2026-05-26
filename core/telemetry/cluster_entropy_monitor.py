# core/telemetry/cluster_entropy_monitor.py
"""
ClusterEntropyMonitor — computes Shannon entropy of the retrieval
frequency distribution maintained by RetrievalAnalytics.

High entropy  → many different memories are retrieved  (healthy diversity)
Low entropy   → a small cluster of memories dominates  (concentration risk)

The AutonomicRegulator polls evaluate() and activates corrective measures
when entropy falls below ENTROPY_THRESHOLD.
"""

import math

from core.telemetry.retrieval_analytics import RetrievalAnalytics
from core.infrastructure.logger import logger


class ClusterEntropyMonitor:

    def __init__(
        self,
        analytics: RetrievalAnalytics,
        threshold: float = 2.0,
    ):
        """
        Parameters
        ----------
        analytics : RetrievalAnalytics
            The shared analytics object updated by the engine.
        threshold : float
            Entropy (bits) below which retrieval is flagged as
            over-concentrated.  Default 2.0 bits.
        """
        self._analytics  = analytics
        self._threshold  = threshold

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def entropy(self) -> float:
        """
        Compute Shannon entropy H = -Σ p(x) log₂ p(x) over the
        retrieval distribution within the analytics window.

        Returns 0.0 when no retrievals have been recorded yet.
        """
        dist = self._analytics.distribution()
        if not dist:
            return 0.0

        h = 0.0
        for p in dist.values():
            if p > 0:
                h -= p * math.log2(p)
        return h

    def evaluate(self) -> dict:
        """
        Evaluate the current retrieval diversity.

        Returns
        -------
        dict
            entropy          : float  — current Shannon entropy in bits
            is_low_diversity : bool   — True when entropy < threshold
                                        AND at least one retrieval has occurred
        """
        h      = self.entropy()
        is_low = (h < self._threshold) and (h > 0.0)

        if is_low:
            logger.info(
                f"[EntropyMonitor] Low diversity detected — "
                f"entropy={h:.3f} bits (threshold={self._threshold})"
            )

        return {
            "entropy":          h,
            "is_low_diversity": is_low,
        }
