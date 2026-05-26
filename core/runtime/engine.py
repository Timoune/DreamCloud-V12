# core/runtime/engine.py

import time
import math
from core.memory.store import load_memory, save_memory
from core.graph.graph_manager import MemoryGraph
from config.memory_config import (
    ACTIVATION_DECAY_CONSTANT,
    ACTIVATION_PENALTY_WEIGHT,
    NOVELTY_WEIGHT,
    NOVELTY_DECAY_CONSTANT
)

class DreamCloudEngine:
    def __init__(self):
        self.graph = MemoryGraph()
        self.similarity_map = {}
        self.running = False

    def process_query_context(self, query_id: str, discovered_ids: list):
        self.running = True
        now = time.time()
        
        # Unique collection of IDs discovered via vector and relational links
        all_ids = list(set(discovered_ids))
        
        # --- Score enrichment via Cognitive Homeostasis ---
        merged = []
        for mid in all_ids:
            try:
                m = load_memory(mid)
                base_score = self.similarity_map.get(mid, 0.0)
                importance = m.importance
                
                neighbours = self.graph.get_neighbors(mid, top_k=3)
                graph_boost = sum(e['weight'] for _, e in neighbours)
                
                # 1. Calculate Activation Penalty (Decays exponentially over time)
                dt = now - m.last_activated
                decayed_activation = m.activation * math.exp(-dt / ACTIVATION_DECAY_CONSTANT) if dt > 0 else m.activation
                activation_penalty = ACTIVATION_PENALTY_WEIGHT * decayed_activation
                
                # 2. Calculate Novelty Bonus (Decays exponentially as the memory ages)
                age = now - m.timestamp
                novelty_bonus = NOVELTY_WEIGHT * math.exp(-age / NOVELTY_DECAY_CONSTANT) if age > 0 else NOVELTY_WEIGHT
                
                # 3. Final Homeostatic Equation
                m.metadata["final_score"] = (
                    base_score 
                    + (0.1 * graph_boost) 
                    + (0.2 * importance) 
                    + (0.1 * m.reliability)
                    - activation_penalty
                    + novelty_bonus
                )
                merged.append(m)
            except Exception:
                continue
                
        # Sort by final score descending
        merged.sort(key=lambda x: x.metadata.get("final_score", 0.0), reverse=True)
        return merged