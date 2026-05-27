"""
DreamCloud runtime engine — v5.

Changes from v4
---------------
1. Cognitive Homeostasis telemetry layer added (non-breaking).
   After every retrieval cycle the engine:
     a. Records retrieved IDs in RetrievalAnalytics (sliding window).
     b. Records graph-edge traversals in GraphHeatmapTracker.
     c. Tracks per-memory activation pressure via ActivationTracker.
     d. Polls AutonomicRegulator — if Shannon entropy of the retrieval
        distribution drops below ENTROPY_THRESHOLD, the regulator
        raises NOVELTY_WEIGHT and lowers ACTIVATION_PENALTY_WEIGHT,
        counteracting concentration and restoring diversity.

2. Scoring formula extended with two new homeostasis terms:
     - activation_penalty : penalises over-retrieved memories so they
       step back and allow less-seen memories to surface.
     - novelty_bonus      : rewards recently-stored memories that have
       had little retrieval exposure so far.
   Both terms use regulated weights from AutonomicRegulator and decay
   exponentially with time (ACTIVATION_DECAY_CONSTANT / NOVELTY_DECAY_CONSTANT).

3. All v4 behaviour (contradiction detection, background extraction,
   DreamCycle, graph expansion, typed ranking) is unchanged.
"""

import traceback
import time
import threading
import queue
import re
import math
import random

from core.embeddings.embedder import Embedder
from core.retrieval.faiss_index import FaissIndex
from core.retrieval.retrieve import retrieve_memories, _saturated_boost
from core.retrieval.ranking import rank_memories

from core.memory.store import (
    load_all_memories,
    load_memory,
    save_memory,
)
from core.memory.schema import Memory, MemoryType
from core.memory.memory_filter import should_store
from core.memory.extractor import extract_memories

from core.runtime.prompt_builder import build_prompt
from core.runtime.llama_runner import run_llama
from core.runtime.nli_validator import is_contradiction as nli_is_contradiction
from core.runtime.response_cleaner import clean_response
from core.runtime.session import SessionState

from core.infrastructure.logger import logger
from core.dreamcycle.dream_cycle import DreamCycle

from core.memory.memory_core import MemoryCore
from core.memory.pipeline import MemoryPipeline

from core.infrastructure.event_bus import EventBus
from core.graph.graph_manager import MemoryGraph

from config.memory_config import (
    CONTRADICTION_PENALTY,
    LLM_USAGE_BOOST,
    ACTIVATION_DECAY_CONSTANT,
    NOVELTY_DECAY_CONSTANT,
    EXPLORATORY_INJECTION_COUNT,
)
import config.memory_config as mem_cfg

# Telemetry / Cognitive Homeostasis
from core.telemetry import (
    RetrievalAnalytics,
    ActivationTracker,
    ClusterEntropyMonitor,
    MemoryStarvationDetector,
    GraphHeatmapTracker,
    AutonomicRegulator,
    ActivationDecayManager,             # BUG FIX #5: was never imported or used
)

DREAM_CYCLE_INTERVAL = 60


# ---------------------------------------------------------------------------
# Contradiction detection — heuristic pre-filter + NLI validation
# ---------------------------------------------------------------------------

_NEGATIONS = {
    "not", "no", "never", "n't", "neither", "nor",
    "don't", "doesn't", "didn't", "won't", "can't",
    "cannot", "isn't", "aren't", "wasn't", "weren't",
}

_STOP_WORDS = {
    "i", "my", "me", "the", "a", "an", "is", "am", "are",
    "was", "were", "it", "its", "this", "that", "and", "or",
    "but", "of", "to", "in", "on", "at", "with", "for",
    "do", "does", "did", "have", "has", "had", "be", "been",
}


def _tokenize(text: str) -> set:
    return set(re.findall(r"\b\w+\b", text.lower()))


def _content_words(tokens: set) -> set:
    return tokens - _STOP_WORDS - _NEGATIONS


def detect_contradiction(a: str, b: str) -> bool:
    """
    Two-stage contradiction detection.

    Stage 1 — Heuristic pre-filter (free):
        The two statements must share >= 2 content words AND exactly one must
        contain a negation token.

    Stage 2 — NLI model validation (only when Stage 1 fires):
        nli_validator.is_contradiction() runs a single CrossEncoder forward
        pass and returns True if the contradiction class probability exceeds
        NLI_CONTRADICTION_THRESHOLD (default 0.7).

        If the NLI call raises, we log a warning and fall back to the
        heuristic result.
    """
    ta = _tokenize(a)
    tb = _tokenize(b)
    shared = _content_words(ta) & _content_words(tb)

    if len(shared) < 2:
        return False

    heuristic_hit = bool(ta & _NEGATIONS) != bool(tb & _NEGATIONS)
    if not heuristic_hit:
        return False

    try:
        return nli_is_contradiction(a, b)
    except Exception:
        logger.warning(
            "[Contradiction] NLI validation failed — falling back to heuristic."
        )
        return heuristic_hit


# ---------------------------------------------------------------------------
# Bounded background extraction queue
# ---------------------------------------------------------------------------

_SENTINEL = object()

_extraction_queue: queue.Queue = queue.Queue(maxsize=20)


def _start_extraction_worker(engine: "DreamCloudEngine") -> threading.Thread:
    """
    Spawn ONE persistent daemon thread that drains _extraction_queue.
    """

    def _worker():
        while True:
            item = _extraction_queue.get()
            if item is _SENTINEL:
                _extraction_queue.task_done()
                break
            user_input = item
            try:
                extractions = extract_memories(user_input)
                for ex in extractions:
                    m         = Memory()
                    m.content = ex["content"]
                    m.type    = ex["type"]
                    engine.memory.process(m)
            except Exception:
                traceback.print_exc()
            finally:
                _extraction_queue.task_done()

    t = threading.Thread(target=_worker, daemon=True, name="ExtractionWorker")
    t.start()
    return t


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class DreamCloudEngine:

    def __init__(self):
        logger.info("Initializing DreamCloud")

        self.embedder  = Embedder()
        self.index     = FaissIndex()
        self.session   = SessionState()
        self.event_bus = EventBus()

        self.memory_core = MemoryCore(self.embedder, self.index)
        self.memory      = MemoryPipeline(self.memory_core, self.event_bus)
        self.graph       = MemoryGraph()

        self.last_retrieved_ids: list = []
        self.similarity_map:     dict = {}

        self._rebuild_index()

        self.dream_cycle = DreamCycle(
            embedder=self.embedder,
            graph=self.graph,
            index=self.index,
        )

        self.last_dream_time = time.time()
        self._dream_lock     = threading.Lock()

        # Start the single persistent extraction worker.
        self._extraction_thread = _start_extraction_worker(self)

        # ── Cognitive Homeostasis telemetry ───────────────────────────
        self.analytics          = RetrievalAnalytics(
            window_seconds=mem_cfg.TELEMETRY_WINDOW_SECONDS
        )
        self.activation_tracker = ActivationTracker()
        self.entropy_monitor    = ClusterEntropyMonitor(
            analytics=self.analytics,
            threshold=mem_cfg.ENTROPY_THRESHOLD,
        )
        self.starvation_detector = MemoryStarvationDetector(
            analytics=self.analytics
        )
        self.heatmap_tracker    = GraphHeatmapTracker(
            window_seconds=mem_cfg.TELEMETRY_WINDOW_SECONDS
        )
        self.regulator          = AutonomicRegulator(
            entropy_monitor=self.entropy_monitor,
            config_module=mem_cfg,
        )
        # BUG FIX #5: ActivationDecayManager was imported (incorrectly) but
        # never instantiated or called anywhere in the engine.  Wire it in now.
        self.decay_manager      = ActivationDecayManager()

    # ------------------------------------------------------------------
    # Startup
    # ------------------------------------------------------------------

    def _rebuild_index(self):
        memories = load_all_memories()
        for m in memories:
            if not m.embedding:
                m.embedding = self.embedder.encode(m.content)
                save_memory(m)
        self.index.rebuild(memories)

    # ------------------------------------------------------------------
    # Background DreamCycle
    # ------------------------------------------------------------------

    def _run_dreamcycle_background(self):
        if not self._dream_lock.acquire(blocking=False):
            return

        def _worker():
            try:
                self.dream_cycle.run()
            except Exception:
                traceback.print_exc()
            finally:
                self.last_dream_time = time.time()
                self._dream_lock.release()

        threading.Thread(target=_worker, daemon=True).start()

    def _maybe_trigger_dreamcycle(self):
        if time.time() - self.last_dream_time > DREAM_CYCLE_INTERVAL:
            print("\n[DreamCycle Triggered — running in background]\n")
            self._run_dreamcycle_background()

    # ------------------------------------------------------------------
    # Background memory storage — bounded queue edition
    # ------------------------------------------------------------------

    def _store_user_memories_background(self, user_input: str):
        if not should_store(user_input):
            return
        try:
            _extraction_queue.put_nowait(user_input)
        except queue.Full:
            logger.warning(
                "[Engine] Extraction queue full — skipping memory storage for this turn."
            )

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def start(self):
        while True:
            try:
                self._maybe_trigger_dreamcycle()

                user_input = input("You: ").strip()
                if not user_input:
                    continue

                self.session.add_user_message(user_input)

                # -- Semantic retrieval ---------------------------------
                memories = retrieve_memories(
                    user_input, self.embedder, self.index
                )

                self.last_retrieved_ids = [m.id for m in memories]
                self.similarity_map = {
                    m.id: m.metadata.get("score", 0) for m in memories
                }

                # -- Graph expansion -----------------------------------
                expanded_ids = self.graph.expand(
                    self.last_retrieved_ids, depth=2, top_k=2
                )

                all_ids = list(set(self.last_retrieved_ids + expanded_ids))

                # -- Telemetry: record retrieval & graph traversals ----
                self.analytics.record(all_ids)
                self.heatmap_tracker.record_expansion(expanded_ids)

                # -- Autonomic regulation (entropy-based) --------------
                regulation_result = self.regulator.regulate()
                effective_novelty            = self.regulator.novelty_weight
                effective_activation_penalty = self.regulator.activation_penalty_weight

                # -- Exploratory injection (when entropy is critically low)
                if self.regulator.is_exploratory:
                    all_mems   = load_all_memories()
                    other_ids  = [m.id for m in all_mems if m.id not in all_ids]
                    if other_ids:
                        injected = random.sample(
                            other_ids,
                            min(EXPLORATORY_INJECTION_COUNT, len(other_ids)),
                        )
                        all_ids = list(set(all_ids + injected))

                # -- Score enrichment ----------------------------------
                merged = []
                now    = time.time()

                for mid in all_ids:
                    try:
                        m          = load_memory(mid)
                        base_score = self.similarity_map.get(mid, 0.0)
                        importance = m.importance

                        neighbours  = self.graph.get_neighbors(mid, top_k=3)
                        graph_boost = sum(e["weight"] for _, e in neighbours)

                        recency_penalty = (now - m.timestamp) / 86400

                        # ── Cognitive Homeostasis terms ────────────────
                        # 1. Activation penalty — penalises over-retrieved memories.
                        dt = now - m.last_activated
                        decayed_activation = (
                            m.activation * math.exp(-dt / ACTIVATION_DECAY_CONSTANT)
                            if dt > 0 else m.activation
                        )
                        activation_penalty = effective_activation_penalty * decayed_activation

                        # 2. Novelty bonus — rewards less-recently-retrieved memories.
                        age = now - m.timestamp
                        novelty_bonus = (
                            effective_novelty * math.exp(-age / NOVELTY_DECAY_CONSTANT)
                            if age > 0 else effective_novelty
                        )

                        m.metadata["final_score"] = (
                            base_score
                            + (0.1 * graph_boost)
                            + (0.3 * importance)
                            + (0.1 * m.reliability)
                            - (0.01 * recency_penalty)
                            - activation_penalty       # homeostasis penalty
                            + novelty_bonus            # homeostasis bonus
                        )

                        # Update activation pressure on this memory.
                        m.activation     = decayed_activation + 1.0
                        m.last_activated = now

                        merged.append(m)

                    except Exception:
                        continue

                # -- Canonical type-aware sort -------------------------
                memories = rank_memories(merged, now=now)

                # -- Track activations for diagnostics -----------------
                self.activation_tracker.record_batch(memories)

                # -- Contradiction resolution (NLI-validated) ----------
                for i, m1 in enumerate(memories):
                    for m2 in memories[i + 1:]:
                        if detect_contradiction(m1.content, m2.content):
                            s1 = m1.metadata.get("final_score", 0)
                            s2 = m2.metadata.get("final_score", 0)

                            if s1 >= s2:
                                m2.importance  *= CONTRADICTION_PENALTY
                                m2.reliability  = max(0.0, m2.reliability - 0.05)
                                save_memory(m2)
                            else:
                                m1.importance  *= CONTRADICTION_PENALTY
                                m1.reliability  = max(0.0, m1.reliability - 0.05)
                                save_memory(m1)

                # -- Save updated activation values --------------------
                for m in merged:
                    save_memory(m)

                # -- Prompt & inference --------------------------------
                prompt       = build_prompt(
                    user_input=user_input,
                    memories=memories,
                    session=self.session,
                    graph=self.graph,
                )
                raw_response = run_llama(prompt)
                response     = clean_response(raw_response)

                print(f"\nMiniVon: {response}\n")

                self.session.add_assistant_message(response)

                # -- LLM usage boost (saturation-aware) ----------------
                for m in memories[:5]:
                    if m.content.lower() in response.lower():
                        boost        = _saturated_boost(LLM_USAGE_BOOST, m.reinforcement_count)
                        m.importance += boost
                        m.reinforcement_count += 1
                        save_memory(m)

                # -- Enqueue user turn for typed-memory extraction -----
                self._store_user_memories_background(user_input)

                # -- Periodic starvation check -------------------------
                self.starvation_detector.check()

                # -- Periodic decay consolidation (hybrid sweep) -------
                # BUG FIX #5: ActivationDecayManager.consolidate() was never
                # called; it now runs the hybrid sweep when its interval elapses.
                all_mems_for_decay = load_all_memories()
                decay_stats = self.decay_manager.consolidate(all_mems_for_decay)

            except KeyboardInterrupt:
                print("\nShutting down DreamCloud.")
                try:
                    _extraction_queue.put_nowait(_SENTINEL)
                except queue.Full:
                    pass
                break

            except Exception:
                print("\n[ENGINE ERROR]\n")
                traceback.print_exc()
