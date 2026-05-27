"""
Activation Decay Architectures for DreamCloud V14
"""
import time
import math
from typing import List
from dataclasses import dataclass
from config.memory_config import (
    DECAY_STRATEGY, GLOBAL_DECAY_RATE, LAZY_DECAY_LAMBDA,
    HYBRID_CONSOLIDATION_INTERVAL, ANOMALY_DETECTION_ENABLED,
    RUNAWAY_REINFORCEMENT_THRESHOLD, OVER_ACTIVATION_THRESHOLD,
    MONOPOLIZATION_THRESHOLD
)

@dataclass
class DecayStats:
    total_memories: int
    avg_activation: float
    max_activation: float
    monopolization_score: float


class ActivationDecayManager:
    def __init__(self):
        self.last_consolidation = time.time()
        self.strategy = DECAY_STRATEGY

    def get_effective_activation(self, memory) -> float:
        if not hasattr(memory, 'activation'):
            return 0.0
        if self.strategy in ["lazy", "hybrid"] and hasattr(memory, 'last_activated'):
            delta = time.time() - memory.last_activated
            decayed = memory.activation * math.exp(-LAZY_DECAY_LAMBDA * delta)
            return max(0.0, decayed)
        return memory.activation

    def apply_global_sweep(self, memories: List) -> None:
        for mem in memories:
            if hasattr(mem, 'activation'):
                mem.activation *= GLOBAL_DECAY_RATE
                if mem.activation < 0.01:
                    mem.activation = 0.0

    def consolidate(self, memories: List):
        now = time.time()
        if self.strategy == "hybrid" and (now - self.last_consolidation > HYBRID_CONSOLIDATION_INTERVAL):
            self.apply_global_sweep(memories)
            self.last_consolidation = now
        return self.get_stats(memories)

    def get_stats(self, memories) -> DecayStats:
        activations = [getattr(m, 'activation', 0.0) for m in memories]
        total = len(activations)
        avg = sum(activations) / total if total else 0.0
        max_act = max(activations) if activations else 0.0

        # BUG FIX #7: monopolization_score was always 0.0.
        # Compute the fraction of total activation held by the top cluster
        # (memories above OVER_ACTIVATION_THRESHOLD) as a rough monopolization
        # signal consistent with MONOPOLIZATION_THRESHOLD semantics.
        monopolization_score = 0.0
        if total > 0 and max_act > 0:
            total_activation = sum(activations)
            if total_activation > 0:
                top_activations = [a for a in activations if a >= OVER_ACTIVATION_THRESHOLD]
                monopolization_score = sum(top_activations) / total_activation

        return DecayStats(
            total_memories=total,
            avg_activation=avg,
            max_activation=max_act,
            monopolization_score=monopolization_score,
        )

    def update_on_access(self, memory) -> None:
        memory.activation = self.get_effective_activation(memory) + 1.0
        memory.last_activated = time.time()
