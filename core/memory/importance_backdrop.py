"""
Retrospective Importance Revaluation — DreamCloud v15.

New information can change the meaning of old information.

Classic example
---------------
A "minor voltage fluctuation" stored with importance 0.8 becomes a critical
precursor once a "catastrophic system failure" arrives with importance 4.5.
Without retroactive revaluation the original memory remains low-priority and
risks being pruned before it can inform future reasoning.

Mechanism: Importance Backpropagation
--------------------------------------
When a high-importance memory (importance >= BACKPROP_TRIGGER_THRESHOLD) is
identified, the engine performs a BFS traversal of the graph starting from
that memory.  For each connected memory it computes a *propagated importance*:

    propagated = trigger.importance
               * BACKPROP_EDGE_WEIGHTS[edge_type]
               * BACKPROP_DEPTH_DECAY ** distance

If the proposed importance exceeds the target memory's current importance by
at least BACKPROP_MIN_DELTA, the update is recorded.

Safeguards
----------
depth decay          — BACKPROP_DEPTH_DECAY^d shrinks signal exponentially.
importance cap       — No memory can be pushed above BACKPROP_IMPORTANCE_CAP.
confidence threshold — Memories with confidence < BACKPROP_CONFIDENCE_THRESHOLD
                       are skipped (too uncertain to merit retroactive elevation).
max-update cap       — BACKPROP_MAX_UPDATES_PER_PASS limits blast radius.
delayed consolidation — When BACKPROP_DELAYED_CONSOLIDATION=True, updates are
                        queued in a JSON file and applied during DreamCycle
                        instead of during live inference.

Public API
----------
ImportanceBackpropEngine.propagate_from(trigger, memory_map)
    → list[BackpropUpdate]          (calculate updates, do not apply)

ImportanceBackpropEngine.run_full_pass(memories)
    → int                           (apply or queue updates for all triggers)

ImportanceBackpropEngine.flush_queue(memory_map)
    → int                           (apply queued updates, called by DreamCycle)
"""

import json
import os
import time
import uuid
from collections import deque
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

from config.memory_config import (
    CACHE_PATH,
    BACKPROP_MAX_DEPTH,
    BACKPROP_DEPTH_DECAY,
    BACKPROP_IMPORTANCE_CAP,
    BACKPROP_MIN_DELTA,
    BACKPROP_CONFIDENCE_THRESHOLD,
    BACKPROP_TRIGGER_THRESHOLD,
    BACKPROP_DELAYED_CONSOLIDATION,
    BACKPROP_MAX_UPDATES_PER_PASS,
    BACKPROP_EDGE_WEIGHTS,
)
from core.infrastructure.logger import logger

os.makedirs(CACHE_PATH, exist_ok=True)

_QUEUE_FILE = os.path.join(CACHE_PATH, "backprop_queue.json")

# Edge types that backpropagation traverses.
# Semantic edges are included at reduced weight; purely structural edges
# (e.g. concept→source) are traversed via the graph but treated as semantic.
_TRAVERSAL_EDGE_TYPES = frozenset({"causal", "temporal", "derived_from", "semantic"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class BackpropUpdate:
    """
    A single proposed importance change calculated by the backprop engine.

    Attributes
    ----------
    update_id        : unique ID for idempotent queue processing
    trigger_id       : the high-importance memory that initiated propagation
    target_id        : the memory whose importance will be raised
    old_importance   : importance before the update
    new_importance   : proposed importance after the update
    backprop_boost   : delta applied (new_importance - old_importance)
    distance         : hop count from trigger to target
    edge_type        : type of the edge closest to the trigger on this path
    created_at       : unix timestamp when the update was calculated
    """
    update_id:      str
    trigger_id:     str
    target_id:      str
    old_importance: float
    new_importance: float
    backprop_boost: float
    distance:       int
    edge_type:      str
    created_at:     float

    @classmethod
    def create(
        cls,
        trigger_id: str,
        target_id: str,
        old_importance: float,
        new_importance: float,
        distance: int,
        edge_type: str,
    ) -> "BackpropUpdate":
        return cls(
            update_id=str(uuid.uuid4()),
            trigger_id=trigger_id,
            target_id=target_id,
            old_importance=old_importance,
            new_importance=new_importance,
            backprop_boost=new_importance - old_importance,
            distance=distance,
            edge_type=edge_type,
            created_at=time.time(),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "BackpropUpdate":
        return cls(**d)


# ---------------------------------------------------------------------------
# Backprop queue (delayed consolidation)
# ---------------------------------------------------------------------------

class BackpropQueue:
    """
    Persistent JSON-backed queue of pending BackpropUpdates.

    When BACKPROP_DELAYED_CONSOLIDATION=True the engine writes calculated
    updates here instead of applying them immediately.  DreamCycle calls
    flush() at consolidation time to apply them in bulk.

    The queue is deduplicated by target_id: if multiple triggers propose an
    update to the same target, only the highest proposed_importance wins.
    """

    def __init__(self, path: str = _QUEUE_FILE):
        self._path = path
        self._items: Dict[str, BackpropUpdate] = {}  # target_id → best update
        self._load()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def enqueue(self, update: BackpropUpdate) -> None:
        """
        Add *update* to the queue, keeping only the highest proposed importance
        per target memory (last-write-wins if equal).
        """
        existing = self._items.get(update.target_id)
        if existing is None or update.new_importance > existing.new_importance:
            self._items[update.target_id] = update
        self._save()

    def enqueue_batch(self, updates: List[BackpropUpdate]) -> None:
        for u in updates:
            existing = self._items.get(u.target_id)
            if existing is None or u.new_importance > existing.new_importance:
                self._items[u.target_id] = u
        self._save()

    def drain(self) -> List[BackpropUpdate]:
        """Return all queued updates and clear the queue."""
        items = list(self._items.values())
        self._items.clear()
        self._save()
        return items

    def __len__(self) -> int:
        return len(self._items)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            payload = [u.to_dict() for u in self._items.values()]
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.warning(f"[BACKPROP] Queue save failed: {exc}")

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for d in payload:
                u = BackpropUpdate.from_dict(d)
                self._items[u.target_id] = u
        except Exception as exc:
            logger.warning(f"[BACKPROP] Queue load failed, starting fresh: {exc}")
            self._items.clear()


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class ImportanceBackpropEngine:
    """
    Retrospective Importance Revaluation engine (v15).

    Usage (inside DreamCycle)
    -------------------------
    engine = ImportanceBackpropEngine(graph)

    # Calculate and queue / apply updates
    n_updated = engine.run_full_pass(memories)

    # If BACKPROP_DELAYED_CONSOLIDATION=True, flush the queue explicitly:
    n_flushed = engine.flush_queue(memory_map)
    """

    def __init__(self, graph, delayed: Optional[bool] = None):
        """
        Parameters
        ----------
        graph   : MemoryGraph instance
        delayed : Override BACKPROP_DELAYED_CONSOLIDATION for testing.
                  Pass None to use the config value.
        """
        self.graph   = graph
        self.delayed = BACKPROP_DELAYED_CONSOLIDATION if delayed is None else delayed
        self.queue   = BackpropQueue()

    # ------------------------------------------------------------------
    # Core propagation logic
    # ------------------------------------------------------------------

    def propagate_from(
        self,
        trigger,
        memory_map: Dict[str, object],
    ) -> List[BackpropUpdate]:
        """
        Calculate importance updates propagating from *trigger*.

        This is a pure calculation — no memory objects are modified.

        Parameters
        ----------
        trigger    : Memory — the high-importance source of propagation
        memory_map : dict mapping memory_id → Memory for fast lookup

        Returns
        -------
        List of BackpropUpdate objects (may be empty).
        At most BACKPROP_MAX_UPDATES_PER_PASS items are returned.
        """
        if trigger.importance < BACKPROP_TRIGGER_THRESHOLD:
            return []

        updates: List[BackpropUpdate] = []

        # BFS: (memory_id, propagated_importance_at_this_node, depth)
        # We track visited nodes to avoid cycles.
        visited: Dict[str, int] = {trigger.id: 0}   # id → depth first seen
        queue:   deque          = deque()
        queue.append((trigger.id, trigger.importance, 0))

        while queue and len(updates) < BACKPROP_MAX_UPDATES_PER_PASS:

            current_id, current_importance, depth = queue.popleft()

            if depth >= BACKPROP_MAX_DEPTH:
                continue

            # Traverse all typed edges from this node.
            for neighbor_id, edge_data in self.graph.get_typed_neighbors(
                current_id,
                edge_types=_TRAVERSAL_EDGE_TYPES,
            ):
                if neighbor_id in visited:
                    continue

                edge_type   = edge_data.get("edge_type", "semantic")
                edge_weight = BACKPROP_EDGE_WEIGHTS.get(edge_type, 0.5)
                new_depth   = depth + 1

                # Formula: propagated = trigger.importance * edge_weight * decay^d
                # We always propagate from the *trigger*'s importance (not the
                # intermediate node's) so that signal strength is determined
                # solely by distance from the originating event.
                propagated = (
                    trigger.importance
                    * edge_weight
                    * (BACKPROP_DEPTH_DECAY ** new_depth)
                )

                # Drop if the propagated signal is negligible.
                if propagated < BACKPROP_CONFIDENCE_THRESHOLD:
                    continue

                target = memory_map.get(neighbor_id)
                if target is None:
                    continue

                # Skip uncertain memories.
                if getattr(target, "confidence", 1.0) < BACKPROP_CONFIDENCE_THRESHOLD:
                    continue

                # Only raise importance — backprop never lowers it.
                proposed = min(
                    max(target.importance, propagated),
                    BACKPROP_IMPORTANCE_CAP,
                )

                delta = proposed - target.importance
                if delta < BACKPROP_MIN_DELTA:
                    # Still traverse onward if the path may reach more impactful nodes.
                    visited[neighbor_id] = new_depth
                    queue.append((neighbor_id, propagated, new_depth))
                    continue

                updates.append(
                    BackpropUpdate.create(
                        trigger_id=trigger.id,
                        target_id=neighbor_id,
                        old_importance=target.importance,
                        new_importance=proposed,
                        distance=new_depth,
                        edge_type=edge_type,
                    )
                )

                visited[neighbor_id] = new_depth
                queue.append((neighbor_id, propagated, new_depth))

        return updates

    # ------------------------------------------------------------------
    # Full pass (called by DreamCycle)
    # ------------------------------------------------------------------

    def run_full_pass(self, memories: list) -> int:
        """
        Run a full backpropagation sweep over *memories*.

        Each memory at or above BACKPROP_TRIGGER_THRESHOLD is used as a
        trigger.  Calculated updates are either applied immediately
        (delayed=False) or enqueued for the next flush() call (delayed=True).

        Returns
        -------
        Number of memories that received an importance update (or were
        enqueued for one).
        """
        from core.memory.store import save_memory

        # Build a fast lookup map.
        memory_map: Dict[str, object] = {m.id: m for m in memories}

        triggers = [
            m for m in memories
            if not m.is_concept and m.importance >= BACKPROP_TRIGGER_THRESHOLD
        ]

        if not triggers:
            logger.info("[BACKPROP] No trigger memories above threshold — pass skipped.")
            return 0

        logger.info(
            f"[BACKPROP] Starting full pass with {len(triggers)} trigger(s) "
            f"across {len(memories)} memories."
        )

        all_updates: List[BackpropUpdate] = []
        seen_targets: set = set()

        for trigger in triggers:
            updates = self.propagate_from(trigger, memory_map)
            for u in updates:
                if u.target_id not in seen_targets:
                    all_updates.append(u)
                    seen_targets.add(u.target_id)

            # Respect the global update cap.
            if len(all_updates) >= BACKPROP_MAX_UPDATES_PER_PASS:
                all_updates = all_updates[:BACKPROP_MAX_UPDATES_PER_PASS]
                break

        if not all_updates:
            logger.info("[BACKPROP] No updates generated in this pass.")
            return 0

        if self.delayed:
            self.queue.enqueue_batch(all_updates)
            logger.info(
                f"[BACKPROP] Queued {len(all_updates)} update(s) for "
                f"delayed consolidation (queue depth: {len(self.queue)})."
            )
            return len(all_updates)
        else:
            return self._apply_updates(all_updates, memory_map, save_memory)

    # ------------------------------------------------------------------
    # Flush queued updates (called by DreamCycle during consolidation)
    # ------------------------------------------------------------------

    def flush_queue(self, memories: list) -> int:
        """
        Apply all queued backpropagation updates.

        Safe to call even when the queue is empty.  Called automatically by
        DreamCycle when BACKPROP_DELAYED_CONSOLIDATION=True.

        Returns
        -------
        Number of memories updated.
        """
        from core.memory.store import save_memory

        pending = self.queue.drain()

        if not pending:
            return 0

        memory_map: Dict[str, object] = {m.id: m for m in memories}

        logger.info(f"[BACKPROP] Flushing {len(pending)} queued update(s).")
        return self._apply_updates(pending, memory_map, save_memory)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_updates(
        self,
        updates: List[BackpropUpdate],
        memory_map: Dict[str, object],
        save_fn,
    ) -> int:
        """Apply *updates* to in-memory objects and persist via *save_fn*."""
        applied = 0

        for u in updates:
            target = memory_map.get(u.target_id)
            if target is None:
                continue  # memory may have been pruned

            # Re-check delta (importance may have changed since calculation).
            if u.new_importance - target.importance < BACKPROP_MIN_DELTA:
                continue

            old = target.importance
            target.importance      = u.new_importance
            target.backprop_boost  = getattr(target, "backprop_boost", 0.0) + u.backprop_boost
            target.backprop_version = getattr(target, "backprop_version", 0) + 1

            save_fn(target)
            applied += 1

            logger.info(
                f"[BACKPROP] {target.id[:8]}({target.type}) "
                f"{old:.2f} → {target.importance:.2f} "
                f"(+{u.backprop_boost:.2f}, depth={u.distance}, "
                f"via={u.edge_type}, trigger={u.trigger_id[:8]})"
            )

        logger.info(f"[BACKPROP] Applied {applied} importance update(s).")
        return applied
