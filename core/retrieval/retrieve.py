# core/retrieval/retrieve.py

import time
import math
from core.memory.store import save_memory, load_all_memories, load_memory
from config.memory_config import SIMILARITY_THRESHOLD, ACTIVATION_DECAY_CONSTANT
from core.memory.schema import MAX_IMPORTANCE

def _saturated_boost(base_boost: float, count: int) -> float:
    # Logistic saturation curve to prevent infinite runaway importance
    return base_boost * (1.0 / (1.0 + math.exp(0.2 * (count - 10))))

def retrieve_memories(query: str, embedder, index, k=5, should_reinforce=True) -> list:
    now = time.time()
    query_embedding = embedder.encode(query)               # BUG FIX #2: was embedder.embed()

    # Search the index
    results = index.search(query_embedding, k=k)
    retrieved = []

    for mid, score in results:
        if score < SIMILARITY_THRESHOLD:
            continue

        try:
            m = load_memory(mid)
        except Exception:
            continue

        if should_reinforce:
            boost = _saturated_boost(1.0, m.reinforcement_count)
            m.access_count += 1
            m.last_accessed = now
            m.last_reinforced = now
            m.reinforcement_count += 1
            # BUG FIX #8: cap at MAX_IMPORTANCE (5.0) not 10.0; setter enforces
            # this but the literal was misleading
            m.importance = min(MAX_IMPORTANCE, m.importance + boost)

            # BUG FIX #4: activation is now managed exclusively by the engine
            # scoring loop to avoid double-incrementing per query.
            # retrieve_memories only handles reinforcement bookkeeping; the
            # engine applies the decayed-activation update after scoring.

            save_memory(m)

        m.metadata["score"] = score
        retrieved.append(m)

    return retrieved
