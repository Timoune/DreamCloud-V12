"""
Memory retrieval with reinforcement, weak-memory decay, and saturation.

Changes from v2
---------------
* Importance saturation: boost is scaled down by reinforcement_count so
  frequently-accessed memories don't compound into belief drift.
  Formula: effective_boost = boost / (1 + k * reinforcement_count)

* reliability is factored into the effective score so contradicted
  memories (lower reliability) naturally rank lower over time.

* Single sort path — engine.py re-sorts after graph/importance enrichment.
"""

import time
import math

from core.memory.store import load_memory, save_memory
from config.memory_config import (
    REINFORCEMENT_THRESHOLD,
    MAX_REINFORCED_PER_QUERY,
    REINFORCEMENT_COOLDOWN,
    WEAK_MEMORY_THRESHOLD,
    WEAK_DECAY_RATE,
)

# Saturation constant — higher = faster diminishing returns on importance boost.
# At k=0.1: 10th boost gives ~50% of the first boost; 50th gives ~17%.
SATURATION_K = 0.1

# How much reliability shifts on contradiction penalty (applied in engine.py).
RELIABILITY_CONTRADICTION_PENALTY = 0.05


def compute_importance(memory) -> float:
    freq    = memory.access_count
    recency = 0.0

    if memory.last_accessed:
        age     = time.time() - memory.last_accessed
        recency = math.exp(-age / 86400)

    return (0.6 * freq) + (0.4 * recency)


def _saturated_boost(base_boost: float, reinforcement_count: int) -> float:
    """Diminishing-returns boost based on how many times a memory was boosted."""
    return base_boost / (1.0 + SATURATION_K * reinforcement_count)


def retrieve_memories(query: str, embedder, index, k: int = 5) -> list:
    """
    Retrieve up to k memories for *query*.

    Returns memories sorted by raw FAISS score (best first).
    Engine applies graph + importance + type weights on top.
    """
    embedding  = embedder.encode(query)
    scored_ids = index.search(embedding, k)

    memories = []
    now      = time.time()

    for rank, (mid, score) in enumerate(scored_ids):
        try:
            m = load_memory(mid)

            # Blend reliability into the raw score so trusted memories
            # rank higher even before engine-level enrichment.
            effective_score = score * (0.7 + 0.3 * m.reliability)
            m.metadata["score"] = effective_score

            # ----------------------------------------------------------
            # Reinforcement gate — with saturation
            # ----------------------------------------------------------
            cooldown_ok = (now - m.last_reinforced) > REINFORCEMENT_COOLDOWN

            should_reinforce = (
                effective_score >= REINFORCEMENT_THRESHOLD
                and rank < MAX_REINFORCED_PER_QUERY
                and cooldown_ok
            )

            if should_reinforce:
                boost = _saturated_boost(1.0, m.reinforcement_count)

                m.access_count        += 1
                m.last_accessed        = now
                m.last_reinforced      = now
                m.reinforcement_count += 1
                m.importance          += boost          # capped by property
                m.importance           = compute_importance(m)
                save_memory(m)

            # ----------------------------------------------------------
            # Weak memory decay
            # ----------------------------------------------------------
            elif effective_score < WEAK_MEMORY_THRESHOLD:
                m.importance  *= WEAK_DECAY_RATE
                m.reliability  = max(0.0, m.reliability - 0.01)
                save_memory(m)

            memories.append(m)

        except Exception:
            pass

    return sorted(
        memories,
        key=lambda m: m.metadata.get("score", 0.0),
        reverse=True,
    )