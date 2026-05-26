# core/retrieval/retrieve.py

import time
import math
from core.memory.store import save_memory, load_all_memories
from config.memory_config import SIMILARITY_THRESHOLD, ACTIVATION_DECAY_CONSTANT

def _saturated_boost(base_boost: float, count: int) -> float:
    # Logistic saturation curve to prevent infinite runaway importance
    return base_boost * (1.0 / (1.0 + math.exp(0.2 * (count - 10))))

def retrieve_memories(query: str, embedder, index, k=5, should_reinforce=True) -> list:
    now = time.time()
    query_embedding = embedder.embed(query)
    
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
            m.importance = min(10.0, m.importance + boost)
            
            # --- Update Homeostasis Retrieval Pressure ---
            dt = now - m.last_activated
            # Decay old activation before adding current retrieval spike
            decayed_activation = m.activation * math.exp(-dt / ACTIVATION_DECAY_CONSTANT) if dt > 0 else m.activation
            m.activation = decayed_activation + 1.0  # Increment current pressure
            m.last_activated = now
            
            save_memory(m)
            
        m.metadata["score"] = score
        retrieved.append(m)
        
    return retrieved