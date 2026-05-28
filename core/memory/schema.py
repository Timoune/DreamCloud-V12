"""
Memory schema — v6.

Changes from v5
---------------
* `retention_policy` (dict, default VOLATILE) replaces the implicit boolean
  archival flag pattern.  Every memory now carries a structured policy object
  that records *class* (volatile / standard / protected / critical), *source*
  (who assigned it), *reason* (why), and an optional *expires* timestamp.

  Critical memories (security alerts, catastrophic-precursor logs, one-time
  events) are assigned RetentionClass.CRITICAL and are immune to DreamCycle
  pruning regardless of their importance score.

  See ``core/memory/retention.py`` for the full policy API.

Changes from v4
---------------
* v5 added full decay_strategy support (no new fields).

Changes from v3
---------------
* v4: `activation` and `last_activated` for Cognitive Homeostasis.

All earlier features (typed memory, reliability, reinforcement_count,
importance hard-cap) are unchanged.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import time
import uuid

from config.memory_config import SCHEMA_VERSION
from core.memory.retention import DEFAULT_POLICY


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
    # Retention policy (v6)
    # ------------------------------------------------------------------
    # Stored as a plain dict so Memory stays JSON-serialisable without a
    # custom encoder.  Use RetentionPolicy.from_dict(m.retention_policy)
    # to work with the structured object.
    retention_policy: Dict = field(default_factory=lambda: dict(DEFAULT_POLICY))

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
