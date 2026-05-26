# core/telemetry/activation_tracker.py
"""
ActivationTracker — samples the activation values of all memories
and computes aggregate statistics (mean, max, variance).  A high mean
activation indicates that the same memories are being retrieved
repeatedly, which correlates with low retrieval entropy and signals
cognitive concentration.

Activation values are fed in by the engine after each scoring pass
via record(); stats() returns a snapshot of the current distribution.
"""

from core.infrastructure.logger import logger


class ActivationTracker:
    """
    Maintains a live map of memory_id → current activation level.
    The engine calls record() after updating m.activation during scoring.
    """

    def __init__(self):
        # {memory_id: activation_value}
        self._activations: dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record(self, memory_id: str, activation: float) -> None:
        """Update the tracked activation for a single memory."""
        self._activations[memory_id] = activation

    def record_batch(self, memories: list) -> None:
        """
        Convenience: record activation for a list of Memory objects.
        Reads m.activation from each; skips if attribute is absent.
        """
        for m in memories:
            self._activations[m.id] = getattr(m, "activation", 0.0)

    def stats(self) -> dict:
        """
        Return aggregate statistics over all tracked activations.

        Returns
        -------
        dict with keys: mean, max, variance, count
        """
        if not self._activations:
            return {"mean": 0.0, "max": 0.0, "variance": 0.0, "count": 0}

        values = list(self._activations.values())
        n      = len(values)
        mean   = sum(values) / n
        max_v  = max(values)
        var    = sum((v - mean) ** 2 for v in values) / n

        return {"mean": mean, "max": max_v, "variance": var, "count": n}

    def clear(self) -> None:
        """Reset all tracked activations (e.g. between DreamCycle runs)."""
        self._activations.clear()
