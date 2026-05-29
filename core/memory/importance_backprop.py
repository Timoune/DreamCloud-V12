"""
importance_backprop.py — compatibility alias for importance_backdrop.py

dream_cycle.py imports from ``core.memory.importance_backprop`` but the
canonical implementation lives in ``importance_backdrop.py``.
This module re-exports everything so both import paths work.
"""

from core.memory.importance_backdrop import (   # noqa: F401
    ImportanceBackpropEngine,
    BackpropUpdate,
    BackpropQueue,
)

__all__ = [
    "ImportanceBackpropEngine",
    "BackpropUpdate",
    "BackpropQueue",
]
