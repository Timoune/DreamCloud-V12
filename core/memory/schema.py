"""
Memory schema — v4.

Changes from v3
---------------
* `activation` (float, default 0.0) tracks cumulative retrieval pressure.
  Decays exponentially between retrievals.  Used by the Cognitive Homeostasis
  scorer to penalise over-retrieved memories and restore diversity.

* `last_activated` (float, default 0.0) records the unix timestamp of the
  most recent retrieval, used to compute time-decayed activation.

All v3 features (typed memory, reliability, reinforcement_count, importance
hard-cap) are unchanged.
"""

from dataclasses import dataclass, field
from typing import List, Dict
import time
import uuid

from config.memory_config import SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Memory type constants
# ---------------------------------------------------------------------------

class MemoryType:
    PREFERENCE = "preference"
    IDENTITY   = "identity"
    FACT       = "fact"
    GOAL       = "goal"
    EPISODIC   = "episodic"
    EMOTIONAL  = "emotional"
    CONCEPT    = "concept"
    GENERAL    = "general"


VALID_TYPES = {
    MemoryType.PREFERENCE,
    MemoryType.IDENTITY,
    MemoryType.FACT,
    MemoryType.GOAL,
    MemoryType.EPISODIC,
    MemoryType.EMOTIONAL,
    MemoryType.CONCEPT,
    MemoryType.GENERAL,
}

TYPE_RETRIEVAL_WEIGHT = {
    MemoryType.IDENTITY:   1.5,
    MemoryType.PREFERENCE: 1.3,
    MemoryType.GOAL:       1.3,
    MemoryType.CONCEPT:    1.2,
    MemoryType.EPISODIC:   1.1,
    MemoryType.EMOTIONAL:  1.0,
    MemoryType.FACT:       1.0,
    MemoryType.GENERAL:    0.8,
}

MAX_IMPORTANCE = 5.0


# ---------------------------------------------------------------------------
# Memory dataclass
# ---------------------------------------------------------------------------

@dataclass
class Memory:

    id:             str   = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp:      float = field(default_factory=time.time)
    schema_version: int   = field(default_factory=lambda: SCHEMA_VERSION)

    content:   str         = ""
    embedding: List[float] = field(default_factory=list)

    type: str = MemoryType.GENERAL

    metadata:    Dict      = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------------
    access_count:        int   = 0
    last_accessed:       float = 0.0
    last_reinforced:     float = 0.0
    reinforcement_count: int   = 0

    _importance: float = field(default=0.0, repr=False)

    reliability: float = 0.5

    # ------------------------------------------------------------------
    # Cognitive Homeostasis — retrieval pressure tracking (v4)
    # ------------------------------------------------------------------
    activation:     float = 0.0   # cumulative retrieval pressure (decays over time)
    last_activated: float = 0.0   # unix timestamp of most recent retrieval

    # ------------------------------------------------------------------
    # DreamCycle concept layer
    # ------------------------------------------------------------------
    is_concept: bool      = False
    confidence: float     = 0.0
    source_ids: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Importance property with hard cap
    # ------------------------------------------------------------------

    @property
    def importance(self) -> float:
        return self._importance

    @importance.setter
    def importance(self, value: float):
        self._importance = min(float(value), MAX_IMPORTANCE)

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "_importance"}
        d["importance"] = self._importance
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "Memory":
        importance = data.pop("importance", 0.0)
        known = {f for f in cls.__dataclass_fields__ if f != "_importance"}
        data = {k: v for k, v in data.items() if k in known}
        obj = cls(**data)
        obj.importance = importance
        return obj
