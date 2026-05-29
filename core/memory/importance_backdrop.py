"""
importance_backdrop.py — Retrospective Importance Revaluation (v15)

DreamCloud Feature: Backpropagation of importance through the memory graph.

When a high-importance memory is discovered (importance > BACKPROP_TRIGGER_THRESHOLD),
this engine walks backwards through the graph, boosting the importance of
memories that causally or structurally contributed to it.

Edge weight table (from memory_config.BACKPROP_EDGE_WEIGHTS):
    causal          1.00  — strongest; causes inherit urgency from effects
    goal_dependency 0.90  — preconditions inherit goal urgency
    derived_from    0.85  — source material inherits signal
    temporal        0.70  — precursors inherit some relevance
    supports        0.65  — corroborating evidence inherits moderate signal
    semantic        0.50  — pure similarity; weak propagation
    contradicts     0.20  — opposing beliefs should not amplify each other

Two modes (BACKPROP_DELAYED_CONSOLIDATION):
    False (default) — updates applied immediately during run_full_pass()
    True            — updates queued; applied during flush_queue() (next DreamCycle)
"""

import time
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from config.memory_config import (
    BACKPROP_MAX_DEPTH,
    BACKPROP_DEPTH_DECAY,
    BACKPROP_IMPORTANCE_CAP,
    BACKPROP_MIN_DELTA,
    BACKPROP_TRIGGER_THRESHOLD,
    BACKPROP_MAX_UPDATES_PER_PASS,
    BACKPROP_EDGE_WEIGHTS,
    BACKPROP_DELAYED_CONSOLIDATION,
)
from core.infrastructure.logger import logger


# ---------------------------------------------------------------------------
# BackpropUpdate — a pending importance delta
# ---------------------------------------------------------------------------

@dataclass
class BackpropUpdate:
    """
    A pending importance update queued for delayed consolidation.

    Attributes
    ----------
    memory_id : target memory to update
    delta     : importance delta to apply (always positive)
    version   : backprop_version of the triggering memory (for audit)
    queued_at : unix timestamp of when the update was queued
    """
    memory_id: str
    delta:     float
    version:   int   = 0
    queued_at: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# BackpropQueue — deduplication + persistence (in-memory only for now)
# ---------------------------------------------------------------------------

class BackpropQueue:
    """
    Manages pending BackpropUpdates for delayed-consolidation mode.

    Deduplication: for each memory_id, only the largest pending delta
    is kept.  Incremental updates from the same pass are merged so that
    the final flush applies the maximum computed boost once.
    """

    def __init__(self):
        # memory_id → BackpropUpdate (largest delta wins on collision)
        self._pending: Dict[str, BackpropUpdate] = {}

    def enqueue(self, updates: List[BackpropUpdate]) -> None:
        for upd in updates:
            existing = self._pending.get(upd.memory_id)
            if existing is None or upd.delta > existing.delta:
                self._pending[upd.memory_id] = upd

    def drain(self) -> List[BackpropUpdate]:
        """Return and clear all pending updates."""
        updates = list(self._pending.values())
        self._pending.clear()
        return updates

    def __len__(self) -> int:
        return len(self._pending)


# ---------------------------------------------------------------------------
# ImportanceBackpropEngine — main engine
# ---------------------------------------------------------------------------

class ImportanceBackpropEngine:
    """
    Retrospective Importance Revaluation engine.

    Usage
    -----
    engine = ImportanceBackpropEngine(graph)

    # Immediate mode (BACKPROP_DELAYED_CONSOLIDATION=False):
    n_updated = engine.run_full_pass(memories)

    # Delayed mode (BACKPROP_DELAYED_CONSOLIDATION=True):
    n_queued  = engine.run_full_pass(memories)   # queues updates
    n_applied = engine.flush_queue(memories)      # applies them next cycle
    """

    def __init__(self, graph):
        """
        Parameters
        ----------
        graph : MemoryGraph
            The memory graph used for neighbor traversal.
        """
        self.graph = graph
        self.queue = BackpropQueue()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_full_pass(self, memories: list) -> int:
        """
        Evaluate all memories and propagate importance backwards through
        the graph for any that exceed BACKPROP_TRIGGER_THRESHOLD.

        In immediate mode  → updates are applied and saved inline.
        In delayed mode    → updates are queued for flush_queue().

        Parameters
        ----------
        memories : list[Memory]
            Full current memory set (already loaded from store).

        Returns
        -------
        int — number of memories updated (immediate) or queued (delayed).
        """
        from core.memory.store import save_memory

        memory_map: Dict[str, object] = {m.id: m for m in memories}

        # Collect all trigger memories (importance above threshold).
        triggers = [
            m for m in memories
            if not m.is_concept and m.importance >= BACKPROP_TRIGGER_THRESHOLD
        ]

        if not triggers:
            logger.debug("[BACKPROP] No trigger memories found.")
            return 0

        all_updates: Dict[str, float] = {}  # memory_id → accumulated delta
        updates_cap_reached = False

        for trigger in triggers:
            if len(all_updates) >= BACKPROP_MAX_UPDATES_PER_PASS:
                updates_cap_reached = True
                break

            self._propagate(
                source_id=trigger.id,
                source_importance=trigger.importance,
                memory_map=memory_map,
                accumulated=all_updates,
            )

        if updates_cap_reached:
            logger.warning(
                f"[BACKPROP] Update cap ({BACKPROP_MAX_UPDATES_PER_PASS}) reached; "
                f"some propagations were skipped."
            )

        if not all_updates:
            return 0

        if BACKPROP_DELAYED_CONSOLIDATION:
            pending = [
                BackpropUpdate(memory_id=mid, delta=delta)
                for mid, delta in all_updates.items()
            ]
            self.queue.enqueue(pending)
            logger.info(f"[BACKPROP] Queued {len(pending)} deferred update(s).")
            return len(pending)
        else:
            return self._apply_updates(all_updates, memory_map, save_memory)

    def flush_queue(self, memories: list) -> int:
        """
        Apply all queued BackpropUpdates (delayed consolidation mode).

        Called by DreamCycle at the start of _run_backprop_pass() when
        BACKPROP_DELAYED_CONSOLIDATION=True.

        Parameters
        ----------
        memories : list[Memory]
            Fresh memory set loaded at the start of this DreamCycle pass.

        Returns
        -------
        int — number of memories updated.
        """
        from core.memory.store import save_memory

        pending = self.queue.drain()
        if not pending:
            return 0

        memory_map: Dict[str, object] = {m.id: m for m in memories}
        accumulated = {upd.memory_id: upd.delta for upd in pending}
        return self._apply_updates(accumulated, memory_map, save_memory)

    # ------------------------------------------------------------------
    # Internal — graph traversal
    # ------------------------------------------------------------------

    def _propagate(
        self,
        source_id:          str,
        source_importance:  float,
        memory_map:         Dict[str, object],
        accumulated:        Dict[str, float],
        _visited:           Optional[set] = None,
        _depth:             int = 0,
    ) -> None:
        """
        Recursive BFS/DFS propagation of importance from *source_id*.

        At each hop:
            delta = source_importance * edge_weight * depth_decay^depth

        Recurse into neighbors if delta >= BACKPROP_MIN_DELTA and
        depth < BACKPROP_MAX_DEPTH.
        """
        if _depth >= BACKPROP_MAX_DEPTH:
            return
        if len(accumulated) >= BACKPROP_MAX_UPDATES_PER_PASS:
            return

        if _visited is None:
            _visited = {source_id}

        neighbors = self.graph.get_typed_neighbors(source_id, top_k=10)

        for neighbor_id, edge_data in neighbors:
            if neighbor_id in _visited:
                continue
            if neighbor_id not in memory_map:
                continue

            edge_type   = edge_data.get("edge_type", "semantic")
            edge_weight = BACKPROP_EDGE_WEIGHTS.get(edge_type, 0.5)

            depth_factor = BACKPROP_DEPTH_DECAY ** _depth
            delta        = source_importance * edge_weight * depth_factor

            if delta < BACKPROP_MIN_DELTA:
                continue

            # Accumulate (keep the maximum delta per memory).
            existing = accumulated.get(neighbor_id, 0.0)
            accumulated[neighbor_id] = max(existing, delta)

            _visited.add(neighbor_id)

            # Recurse deeper.
            self._propagate(
                source_id=neighbor_id,
                source_importance=delta,   # attenuated importance
                memory_map=memory_map,
                accumulated=accumulated,
                _visited=_visited,
                _depth=_depth + 1,
            )

    # ------------------------------------------------------------------
    # Internal — apply computed deltas
    # ------------------------------------------------------------------

    def _apply_updates(
        self,
        accumulated: Dict[str, float],
        memory_map:  Dict[str, object],
        save_fn,
    ) -> int:
        """
        Write the accumulated importance deltas back to the memory objects
        and persist them.

        Returns the count of memories actually updated.
        """
        applied = 0

        for memory_id, delta in accumulated.items():
            m = memory_map.get(memory_id)
            if m is None:
                continue

            old_importance = m.importance
            new_importance = min(old_importance + delta, BACKPROP_IMPORTANCE_CAP)

            if new_importance - old_importance < BACKPROP_MIN_DELTA:
                continue  # delta too small after cap enforcement

            m.importance        = new_importance
            m.backprop_boost    = getattr(m, "backprop_boost",    0.0) + delta
            m.backprop_version  = getattr(m, "backprop_version",  0)   + 1

            try:
                save_fn(m)
                applied += 1
                logger.debug(
                    f"[BACKPROP] {m.id[:8]} importance "
                    f"{old_importance:.3f} → {new_importance:.3f} "
                    f"(delta={delta:.3f}, depth_chain)"
                )
            except Exception as exc:
                logger.warning(f"[BACKPROP] Failed to save {memory_id[:8]}: {exc}")

        if applied:
            logger.info(f"[BACKPROP] Applied {applied} importance update(s).")

        return applied
