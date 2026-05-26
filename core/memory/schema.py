# core/memory/schema.py

from dataclasses import dataclass, field
from typing import List, Dict, Any

@dataclass
class Memory:
    id: str
    content: str
    embedding: List[float]
    timestamp: float
    importance: float
    reliability: float
    type: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

    # Reinforcement tracking
    access_count: int = 0
    last_accessed: float = 0.0
    last_reinforced: float = 0.0
    reinforcement_count: int = 0

    # --- Homeostasis Tracking Fields ---
    activation: float = 0.0
    last_activated: float = 0.0