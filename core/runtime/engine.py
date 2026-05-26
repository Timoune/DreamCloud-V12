"""
DreamCloud Engine — orchestrates retrieval, graph expansion, and
cognitive-homeostasis scoring.  The new telemetry layer monitors
retrieval diversity and autonomically adjusts scoring parameters
when entropy drops.
"""

import time
import math
import random
from core.memory.store import load_memory, load_all_memories, save_memory
from core.graph.graph_manager import MemoryGraph
from core.retrieval.retrieve import retrieve_memories
from config.memory_config import (
    ACTIVATION_DECAY_CONSTANT,
    ACTIVATION_PENALTY_WEIGHT,
    NOVELTY_WEIGHT,
    NOVELTY_DECAY_CONSTANT,
)
# ── Telemetry imports ─────────────────────────────────────────────
from core.telemetry import (
    RetrievalAnalytics,
    ActivationTracker,
    ClusterEntropyMonitor,
    MemoryStarvationDetector,
    GraphHeatmapTracker,
    AutonomicRegulator,
)
import config.memory_config as mem_cfg


class DreamCloudEngine:
    def __init__(self):
        self.graph = MemoryGraph()
        self.similarity_map = {}
        self.running = False

        # ── Telemetry layer ────────────────────────────────────────
        self.analytics = RetrievalAnalytics(
            window_seconds=mem_cfg.TELEMETRY_WINDOW_SECONDS
        )
        self.activation_tracker = ActivationTracker()
        self.entropy_monitor = ClusterEntropyMonitor(
            analytics=self.analytics,
            threshold=mem_cfg.ENTROPY_THRESHOLD,
        )
        self.starvation_detector = MemoryStarvationDetector(
            analytics=self.analytics
        )
        self.heatmap_tracker = GraphHeatmapTracker(
            window_seconds=mem_cfg.TELEMETRY_WINDOW_SECONDS
        )
        self.regulator = AutonomicRegulator(
            entropy_monitor=self.entropy_monitor,
            config_module=mem_cfg,
        )

    # ------------------------------------------------------------------
    def process_query_context(self, query_id: str, discovered_ids: list):
        """
        Main scoring pipeline.
        1. Records retrieval distribution for telemetry.
        2. Evaluates entropy and optionally triggers autonomic regulation.
        3. Enriches memories with homeostatic scoring using regulated weights.
        """
        self.running = True
        now = time.time()

        # ── Record retrieval analytics ─────────────────────────────
        self.analytics.record(discovered_ids)

        # ── Record graph traversals ────────────────────────────────
        self.heatmap_tracker.record_expansion(discovered_ids)

        all_ids = list(set(discovered_ids))

        # ── Autonomic regulation ───────────────────────────────────
        regulation_result = self.regulator.regulate()

        # ── Use the possibly-altered scoring weights ───────────────
        effective_novelty = self.regulator.novelty_weight
        effective_activation_penalty = self.regulator.activation_penalty_weight

        # ── Exploratory injection (when entropy is low) ────────────
        if self.regulator.is_exploratory:
            all_mems = load_all_memories()
            other_ids = [m.id for m in all_mems if m.id not in all_ids]
            if other_ids:
                injected = random.sample(
                    other_ids,
                    min(mem_cfg.EXPLORATORY_INJECTION_COUNT, len(other_ids)),
                )
                all_ids.extend(injected)
                all_ids = list(set(all_ids))

        # ── Score enrichment (Cognitive Homeostasis) ───────────────
        merged = []
        for mid in all_ids:
            try:
                m = load_memory(mid)
                base_score = self.similarity_map.get(mid, 0.0)
                importance = m.importance
                neighbours = self.graph.get_neighbors(mid, top_k=3)
                graph_boost = sum(e['weight'] for _, e in neighbours)

                # 1. Activation Penalty (with regulated weight)
                dt = now - m.last_activated
                decayed_activation = (
                    m.activation * math.exp(-dt / ACTIVATION_DECAY_CONSTANT)
                    if dt > 0 else m.activation
                )
                activation_penalty = effective_activation_penalty * decayed_activation

                # 2. Novelty Bonus (with regulated weight)
                age = now - m.timestamp
                novelty_bonus = (
                    effective_novelty * math.exp(-age / NOVELTY_DECAY_CONSTANT)
                    if age > 0 else effective_novelty
                )

                # 3. Final score
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

        merged.sort(key=lambda x: x.metadata.get("final_score", 0.0), reverse=True)
        return merged