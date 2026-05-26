"""
Memory schema — v3.

Changes from v2
---------------
* `type` uses MemoryType constants instead of a free-form "general" string.
  Retrieval and ranking now behave differently per type via TYPE_RETRIEVAL_WEIGHT.

* `reliability` (0.0–1.0) tracks source trustworthiness independently of
  `importance`. Starts at 0.5 (neutral). Falls when a memory is penalised
  for contradiction, rises when corroborated.

* `importance` is hard-capped at MAX_IMPORTANCE (5.0) through a property
  setter, preventing LLM-usage boost from compounding into belief drift.

* `reinforcement_count` records lifetime boost count so retrieve.py can
  apply saturation (diminishing returns after many boosts).
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
    PREFERENCE = "preference"   # likes, dislikes, favourites
    IDENTITY   = "identity"     # facts about who the user is
    FACT       = "fact"         # world / topic knowledge
    GOAL       = "goal"         # intentions, plans
    EPISODIC   = "episodic"     # specific past events
    EMOTIONAL  = "emotional"    # emotional states / reactions
    CONCEPT    = "concept"      # DreamCycle synthesised concept
    GENERAL    = "general"      # fallback / uncategorised


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

# Retrieval weight multiplier applied per type in ranking.py.
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

# Hard cap on importance — prevents compounding boosts from distorting memory.
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

    # Typed memory category — always use MemoryType constants.
    type: str = MemoryType.GENERAL

    metadata:    Dict      = field(default_factory=dict)
    connections: List[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Reinforcement
    # ------------------------------------------------------------------
    access_count:        int   = 0
    last_accessed:       float = 0.0
    last_reinforced:     float = 0.0
    reinforcement_count: int   = 0   # lifetime boost count for saturation

    # Internal importance storage — always access via the property below.
    _importance: float = field(default=0.0, repr=False)

    # Source reliability [0.0, 1.0].  0.5 = unknown/neutral.
    reliability: float = 0.5

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

    # ------------------------------------------------------------------
    # Serialisation
    # store.py calls to_dict() instead of __dict__ so the capped value
    # is written correctly.
    # ------------------------------------------------------------------

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
        obj.importance = importance   # goes through the capped setter
        return obj