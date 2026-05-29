"""
contradiction_system.py — DreamCloud Feature 6: Contradiction Handling & Belief Systems

Architecture
------------
New Memory
    ↓
Semantic Candidate Search
    ↓
NLI Validation  (contradiction | entailment | neutral)
    ↓
ContradictionEvent Queue
    ↓
GhostMind Arbitration  (stub; configurable via GHOSTMIND_ARBITRATION_ENABLED)

Motivation
----------
V12 contradiction handling was REACTIVE: contradictions were detected during
retrieval, but by then both conflicting memories were already competing in the
prompt context window.  By the time the weaker memory was penalized, it may
have already influenced the LLM response.

V16 contradiction handling is STRUCTURAL:
  1. Contradiction events are detected AND classified at storage time.
  2. The full 3-class NLI result is used — not just contradiction.
     - Entailment   → merge / strengthen / reinforce reliability.
     - Neutral       → beliefs coexist without interference.
     - Contradiction → penalize the weaker belief.
  3. Events are persisted in ContradictionEventQueue for audit and for
     asynchronous GhostMind arbitration.

Belief Outcomes
---------------
CONTRADICTION  → weaker memory's importance × CONTRADICTION_PENALTY,
                 reliability decreased, contradiction_count incremented.
ENTAILMENT     → both memories get reliability boost; candidate importance
                 is gently raised; entailment_count incremented.
NEUTRAL        → no action; beliefs are compatible and coexist.

GhostMind Hook
--------------
Set GHOSTMIND_ARBITRATION_ENABLED=True to route all ContradictionEvents to
GhostMind instead of applying them inline.  DreamCycle will call
BeliefSystem.flush_pending_ghostmind_events() as a safety net so that events
are never lost even if GhostMind is unavailable.

Time-Aware Beliefs (future)
---------------------------
Memories will carry `belief_valid_from` / `belief_valid_until` windows.
Arbitration will respect temporal ordering before assigning a "winner":
a belief about today cannot contradict a belief about last week.
"""

import json
import os
import time
import uuid
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Optional, Dict

from config.memory_config import (
    CONTRADICTION_PENALTY,
    CONTRADICTION_QUEUE_PATH,
    ENTAILMENT_RELIABILITY_BOOST,
    ENTAILMENT_IMPORTANCE_BOOST,
    GHOSTMIND_ARBITRATION_ENABLED,
)
from core.infrastructure.logger import logger
from core.runtime.nli_validator import classify_relationship, NLIRelationship


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class BeliefOutcome(str, Enum):
    """
    The resolved relationship between two memories after NLI classification.

    CONTRADICTION — memories conflict; weaker will be penalized.
    ENTAILMENT    — memories agree; both will be reinforced.
    NEUTRAL       — memories are compatible but independent; no action.
    """
    CONTRADICTION = "contradiction"
    ENTAILMENT    = "entailment"
    NEUTRAL       = "neutral"


# ---------------------------------------------------------------------------
# ContradictionEvent
# ---------------------------------------------------------------------------

@dataclass
class ContradictionEvent:
    """
    A single resolved belief event in the contradiction / belief pipeline.

    Created for every non-NEUTRAL NLI result between two memories.
    Persisted in ContradictionEventQueue for arbitration and auditing.

    Attributes
    ----------
    event_id           : unique identifier for idempotent processing
    new_memory_id      : the memory that triggered the comparison
    candidate_id       : the existing memory that was compared against
    outcome            : BeliefOutcome (CONTRADICTION | ENTAILMENT | NEUTRAL)
    contradiction_score: NLI softmax probability for contradiction
    entailment_score   : NLI softmax probability for entailment
    neutral_score      : NLI softmax probability for neutral
    new_memory_text    : snapshot of new memory content (for audit)
    candidate_text     : snapshot of candidate content (for audit)
    resolved           : True once arbitration has been applied
    resolved_at        : unix timestamp of resolution
    created_at         : unix timestamp of event creation
    """
    event_id:            str
    new_memory_id:       str
    candidate_id:        str
    outcome:             BeliefOutcome
    contradiction_score: float
    entailment_score:    float
    neutral_score:       float
    new_memory_text:     str
    candidate_text:      str
    resolved:            bool  = False
    resolved_at:         float = 0.0
    created_at:          float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        new_memory_id:       str,
        candidate_id:        str,
        outcome:             BeliefOutcome,
        contradiction_score: float,
        entailment_score:    float,
        neutral_score:       float,
        new_memory_text:     str,
        candidate_text:      str,
    ) -> "ContradictionEvent":
        return cls(
            event_id=str(uuid.uuid4()),
            new_memory_id=new_memory_id,
            candidate_id=candidate_id,
            outcome=outcome,
            contradiction_score=contradiction_score,
            entailment_score=entailment_score,
            neutral_score=neutral_score,
            new_memory_text=new_memory_text,
            candidate_text=candidate_text,
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["outcome"] = self.outcome.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ContradictionEvent":
        d = dict(d)
        d["outcome"] = BeliefOutcome(d["outcome"])
        return cls(**d)


# ---------------------------------------------------------------------------
# ContradictionEventQueue
# ---------------------------------------------------------------------------

class ContradictionEventQueue:
    """
    Persistent JSON-backed queue of ContradictionEvents.

    Deduplication: for each (new_memory_id, candidate_id) pair, only the most
    recent event is kept.  Multiple NLI runs for the same pair update the
    existing entry rather than accumulating duplicates.

    When GHOSTMIND_ARBITRATION_ENABLED=False, events are marked resolved
    immediately after inline arbitration.  When True, they persist here until
    GhostMind consumes them.
    """

    def __init__(self, path: str = CONTRADICTION_QUEUE_PATH):
        self._path   = path
        # event_id → ContradictionEvent
        self._events: Dict[str, ContradictionEvent] = {}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._load()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def enqueue(self, event: ContradictionEvent) -> None:
        """
        Add *event* to the queue.

        If an unresolved event for the same (new_memory_id, candidate_id) pair
        already exists, replace it with the new event (fresher NLI result wins).
        """
        # Remove stale unresolved event for the same pair.
        stale_key = None
        for eid, existing in self._events.items():
            if (
                not existing.resolved
                and existing.new_memory_id == event.new_memory_id
                and existing.candidate_id  == event.candidate_id
            ):
                stale_key = eid
                break

        if stale_key:
            del self._events[stale_key]

        self._events[event.event_id] = event
        self._save()

        logger.info(
            f"[BELIEF] Queued ContradictionEvent({event.outcome.value}): "
            f"new={event.new_memory_id[:8]} ↔ candidate={event.candidate_id[:8]}"
        )

    def drain_unresolved(self) -> List[ContradictionEvent]:
        """Return all unresolved events without clearing them from the queue."""
        return [e for e in self._events.values() if not e.resolved]

    def mark_resolved(self, event_id: str) -> None:
        e = self._events.get(event_id)
        if e:
            e.resolved    = True
            e.resolved_at = time.time()
            self._save()

    def __len__(self) -> int:
        return sum(1 for e in self._events.values() if not e.resolved)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self) -> None:
        try:
            payload = [e.to_dict() for e in self._events.values()]
            with open(self._path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
        except Exception as exc:
            logger.warning(f"[BELIEF] Queue save failed: {exc}")

    def _load(self) -> None:
        if not os.path.exists(self._path):
            return
        try:
            with open(self._path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            for d in payload:
                e = ContradictionEvent.from_dict(d)
                self._events[e.event_id] = e
        except Exception as exc:
            logger.warning(f"[BELIEF] Queue load failed, starting fresh: {exc}")
            self._events.clear()


# ---------------------------------------------------------------------------
# BeliefSystem
# ---------------------------------------------------------------------------

class BeliefSystem:
    """
    Structured contradiction detection and belief resolution.

    Pipeline  (per memory pair)
    ---------------------------
    1. classify_relationship(new, candidate)  →  NLIClassification
    2. Map NLIRelationship → BeliefOutcome
    3. Create ContradictionEvent, enqueue
    4. GHOSTMIND_ARBITRATION_ENABLED=False  → _arbitrate() inline
       GHOSTMIND_ARBITRATION_ENABLED=True   → _dispatch_to_ghostmind()

    Arbitration rules
    -----------------
    CONTRADICTION:
        - Weaker memory (lower final_score) loses.
        - Age tiebreak: newer information wins.
        - Weaker memory: importance × CONTRADICTION_PENALTY,
          reliability - 0.05, contradiction_count += 1.
        - A "contradicts" graph edge is registered between the pair.

    ENTAILMENT:
        - Both memories: reliability += ENTAILMENT_RELIABILITY_BOOST.
        - Candidate importance += ENTAILMENT_IMPORTANCE_BOOST (new memory
          confirms the existing belief, making it more trustworthy).
        - entailment_count += 1 for both.
        - A "supports" graph edge is registered between the pair.

    NEUTRAL:
        - No action. Logged at DEBUG level.
    """

    def __init__(self, graph=None):
        """
        Parameters
        ----------
        graph : MemoryGraph | None
            If provided, the BeliefSystem registers typed graph edges
            (contradicts / supports) when it resolves events.
        """
        self.queue = ContradictionEventQueue()
        self.graph = graph

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def evaluate(
        self,
        new_memory,
        candidates: list,
        save_fn=None,
    ) -> List[ContradictionEvent]:
        """
        Run NLI validation between *new_memory* and each *candidate*.

        Parameters
        ----------
        new_memory : Memory
            The newly stored or retrieved memory acting as the reference.
        candidates : list[Memory]
            Semantically similar memories to compare against.
        save_fn    : callable(Memory) | None
            Persistence function; defaults to core.memory.store.save_memory.

        Returns
        -------
        List of ContradictionEvents that were queued (NEUTRAL outcomes are
        skipped — they produce no events).
        """
        from core.memory.store import save_memory as _default_save

        save = save_fn or _default_save
        events_queued: List[ContradictionEvent] = []

        for candidate in candidates:
            if candidate.id == new_memory.id:
                continue

            try:
                classification = classify_relationship(
                    new_memory.content,
                    candidate.content,
                )
            except Exception as exc:
                logger.warning(
                    f"[BELIEF] NLI failed for pair "
                    f"({new_memory.id[:8]}, {candidate.id[:8]}): {exc}"
                )
                continue

            rel = classification.relationship

            if rel == NLIRelationship.NEUTRAL:
                logger.debug(
                    f"[BELIEF] NEUTRAL: "
                    f"{new_memory.id[:8]} ↔ {candidate.id[:8]} "
                    f"(neutral={classification.neutral_score:.3f})"
                )
                continue  # compatible beliefs — no action needed

            outcome = (
                BeliefOutcome.CONTRADICTION
                if rel == NLIRelationship.CONTRADICTION
                else BeliefOutcome.ENTAILMENT
            )

            event = ContradictionEvent.create(
                new_memory_id=new_memory.id,
                candidate_id=candidate.id,
                outcome=outcome,
                contradiction_score=classification.contradiction_score,
                entailment_score=classification.entailment_score,
                neutral_score=classification.neutral_score,
                new_memory_text=new_memory.content,
                candidate_text=candidate.content,
            )

            self.queue.enqueue(event)
            events_queued.append(event)

            if not GHOSTMIND_ARBITRATION_ENABLED:
                self._arbitrate(event, new_memory, candidate, save)
                self.queue.mark_resolved(event.event_id)
            else:
                self._dispatch_to_ghostmind(event)

        return events_queued

    # ------------------------------------------------------------------
    # Arbitration (inline / immediate mode)
    # ------------------------------------------------------------------

    def _arbitrate(
        self,
        event: ContradictionEvent,
        new_memory,
        candidate,
        save_fn,
    ) -> None:
        """Dispatch to the correct resolution handler based on event outcome."""
        if event.outcome == BeliefOutcome.CONTRADICTION:
            self._resolve_contradiction(new_memory, candidate, save_fn)
        elif event.outcome == BeliefOutcome.ENTAILMENT:
            self._resolve_entailment(new_memory, candidate, save_fn)

    def _resolve_contradiction(self, new_memory, candidate, save_fn) -> None:
        """
        Penalize the weaker of the two conflicting memories.

        Selection rules (in order):
        1. Lower final_score (scoring pipeline result) → weaker.
        2. Equal scores: older memory → weaker (newer information wins).

        The losing memory's importance and reliability are reduced.
        A 'contradicts' graph edge is registered between the pair.
        """
        new_score  = new_memory.metadata.get("final_score", new_memory.importance)
        cand_score = candidate.metadata.get("final_score", candidate.importance)

        if abs(new_score - cand_score) < 0.01:
            # Age tiebreak: newer wins.
            weaker, stronger = (
                (candidate, new_memory)
                if new_memory.timestamp >= candidate.timestamp
                else (new_memory, candidate)
            )
        elif new_score >= cand_score:
            weaker, stronger = candidate, new_memory
        else:
            weaker, stronger = new_memory, candidate

        weaker.importance          = max(0.0, weaker.importance * CONTRADICTION_PENALTY)
        weaker.reliability         = max(0.0, weaker.reliability - 0.05)
        weaker.contradiction_count = getattr(weaker, "contradiction_count", 0) + 1
        save_fn(weaker)

        # Register structural contradiction edge in the graph.
        if self.graph is not None:
            self.graph.connect_contradicts(weaker.id, stronger.id)

        logger.info(
            f"[BELIEF] CONTRADICTION → penalized {weaker.id[:8]}({weaker.type}) "
            f"importance={weaker.importance:.3f}, reliability={weaker.reliability:.3f} | "
            f"winner={stronger.id[:8]}({stronger.type})"
        )

    def _resolve_entailment(self, new_memory, candidate, save_fn) -> None:
        """
        Reinforce both memories when they are confirmed to agree.

        - Both memories get a reliability boost (mutual confirmation).
        - The existing candidate's importance is gently raised (the new
          memory acts as corroborating evidence).
        - Both memories get an entailment_count increment (used by DreamCycle
          to prioritise concept strengthening for well-confirmed beliefs).
        - A 'supports' graph edge is registered between the pair.
        """
        for m in (new_memory, candidate):
            m.reliability      = min(1.0, m.reliability + ENTAILMENT_RELIABILITY_BOOST)
            m.entailment_count = getattr(m, "entailment_count", 0) + 1
            save_fn(m)

        # Raise candidate importance (new memory corroborates it).
        candidate.importance = min(
            candidate.importance + ENTAILMENT_IMPORTANCE_BOOST,
            5.0,
        )
        save_fn(candidate)

        # Register structural support edge in the graph.
        if self.graph is not None:
            self.graph.connect_supports(new_memory.id, candidate.id)

        logger.info(
            f"[BELIEF] ENTAILMENT → reinforced "
            f"{new_memory.id[:8]} + {candidate.id[:8]} | "
            f"candidate importance={candidate.importance:.3f}, "
            f"reliability={candidate.reliability:.3f}"
        )

    # ------------------------------------------------------------------
    # GhostMind arbitration hook
    # ------------------------------------------------------------------

    def _dispatch_to_ghostmind(self, event: ContradictionEvent) -> None:
        """
        Hand off a ContradictionEvent to GhostMind for external arbitration.

        Current status: STUB.
        The event persists in ContradictionEventQueue.  GhostMind is expected
        to call drain_unresolved() → apply arbitration → mark_resolved().

        Future: emit over a message bus, gRPC channel, or shared-memory ring
        buffer depending on the GhostMind integration architecture.
        """
        logger.info(
            f"[BELIEF→GHOSTMIND] Event {event.event_id[:8]} "
            f"({event.outcome.value}) queued for GhostMind arbitration."
        )
        # TODO: implement GhostMind IPC channel when integration is ready.

    # ------------------------------------------------------------------
    # DreamCycle integration — flush pending GhostMind events
    # ------------------------------------------------------------------

    def flush_pending_ghostmind_events(
        self,
        memory_map: dict,
        save_fn=None,
    ) -> int:
        """
        Apply all unresolved ContradictionEvents during DreamCycle consolidation.

        This is the safety net for GhostMind mode: if GhostMind has not
        consumed the queue before the next DreamCycle pass, this method
        applies the pending events directly so no belief resolution is lost.

        Parameters
        ----------
        memory_map : dict mapping memory_id → Memory
        save_fn    : persistence callable

        Returns
        -------
        Number of events applied.
        """
        from core.memory.store import save_memory as _default_save

        save    = save_fn or _default_save
        applied = 0

        for event in self.queue.drain_unresolved():
            new_mem  = memory_map.get(event.new_memory_id)
            cand_mem = memory_map.get(event.candidate_id)

            if new_mem is None or cand_mem is None:
                # One of the memories was pruned — nothing to arbitrate.
                self.queue.mark_resolved(event.event_id)
                continue

            self._arbitrate(event, new_mem, cand_mem, save)
            self.queue.mark_resolved(event.event_id)
            applied += 1

        if applied:
            logger.info(
                f"[BELIEF] DreamCycle flushed {applied} pending belief event(s) "
                f"from GhostMind queue."
            )

        return applied
