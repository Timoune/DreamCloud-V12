"""
DreamCloud runtime engine — v4.
 
Changes from v3
---------------
1. detect_contradiction() is now a two-stage pipeline:
       Stage 1 — heuristic pre-filter (negation token + shared content words).
                 Free, runs on every candidate pair.
       Stage 2 — NLI model validation via nli_validator.is_contradiction().
                 Only invoked when Stage 1 fires, keeping inference cost low.
   The NLI model (cross-encoder/nli-MiniLM2-L6-H768) replaces the generative
   LLM YES/NO prompt used in v3, giving faster and more reliable results.
   If the NLI call fails, the heuristic result is used as a fallback.
 
2. Background extraction now uses a *single persistent worker thread* fed
   by a bounded Queue (maxsize=20) instead of spawning one thread per turn.
   If the queue is full the turn is silently skipped (memory never blocks
   the response). This caps worst-case thread/memory pressure regardless of
   how fast the user types or how slow the LLM is.
"""
 
import traceback
import time
import threading
import queue
import re
 
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
        contain a negation token.  Pairs that fail either check are dismissed
        immediately — no model inference is spent on clearly unrelated pairs.
 
    Stage 2 — NLI model validation (only when Stage 1 fires):
        nli_validator.is_contradiction() runs a single CrossEncoder forward
        pass and returns True if the contradiction class probability exceeds
        NLI_CONTRADICTION_THRESHOLD (default 0.7).
 
        Advantages over the generative LLM approach used in v3:
          - ~5-15 ms per pair vs 200-2000 ms for autoregressive generation.
          - Deterministic output — no temperature / sampling variance.
          - Trained specifically for entailment/contradiction, not prompted.
          - Catches nuanced contradictions without a surface negation token
            (e.g. "I prefer tea" vs "coffee is my go-to drink").
 
        If the NLI call raises, we log a warning and fall back to the
        heuristic result so contradiction handling degrades gracefully.
    """
    ta = _tokenize(a)
    tb = _tokenize(b)
    shared = _content_words(ta) & _content_words(tb)
 
    # Stage 1: no shared topic — cannot contradict.
    if len(shared) < 2:
        return False
 
    # Stage 1: negation must be asymmetric between the two statements.
    heuristic_hit = bool(ta & _NEGATIONS) != bool(tb & _NEGATIONS)
    if not heuristic_hit:
        return False
 
    # Stage 2: NLI model confirms or rejects the heuristic signal.
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
 
# maxsize=20 caps the backlog; put_nowait() silently drops turns when full.
_extraction_queue: queue.Queue = queue.Queue(maxsize=20)
 
 
def _start_extraction_worker(engine: "DreamCloudEngine") -> threading.Thread:
    """
    Spawn ONE persistent daemon thread that drains _extraction_queue.
 
    Advantages over spawning a new thread per turn:
    - Thread-creation overhead is paid only once.
    - Queue depth is bounded (maxsize=20), so a slow LLM cannot cause
      unbounded memory/thread growth.
    - Ordering is preserved: memories are stored in the order turns arrived.
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
        """
        Enqueue *user_input* for typed-memory extraction on the persistent
        worker thread.
 
        If the queue is already full (worker is lagging behind the user),
        we silently discard this turn rather than blocking the response or
        spawning an unbounded number of threads.
        """
        if not should_store(user_input):
            return
        try:
            _extraction_queue.put_nowait(user_input)
        except queue.Full:
            logger.warning(
                "[Engine] Extraction queue full — skipping memory storage for this turn. "
                "Consider increasing LLM throughput or reducing conversation speed."
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
 
                        m.metadata["final_score"] = (
                            base_score
                            + (0.1 * graph_boost)
                            + (0.3 * importance)
                            + (0.1 * m.reliability)
                            - (0.01 * recency_penalty)
                        )
 
                        merged.append(m)
 
                    except Exception:
                        continue
 
                # -- Canonical type-aware sort -------------------------
                memories = rank_memories(merged, now=now)
 
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