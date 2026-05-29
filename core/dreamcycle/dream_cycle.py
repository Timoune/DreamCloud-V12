import math
import re
 
import numpy as np
import faiss as _faiss
 
from config.memory_config import (
    CONCEPT_SIMILARITY_THRESHOLD,
    CONCEPT_MIN_SIZE,
    PRUNE_IMPORTANCE_THRESHOLD,
)
 
from core.memory.store import (
    load_all_memories,
    save_memory,
    delete_memory,
)
 
from core.embeddings.embedder import Embedder
from core.runtime.llama_runner import run_llama
from core.graph.graph_manager import MemoryGraph
from core.dreamcycle.concept_manager import create_concept
from core.infrastructure.logger import logger
from core.memory.retention import RetentionPolicy, RetentionClass, is_prune_safe
from core.memory.importance_backprop import ImportanceBackpropEngine
from core.memory.contradiction_system import BeliefSystem
from config.memory_config import (
    RETENTION_AUDIT_LOG_ENABLED,
    RETENTION_AUTO_CRITICAL_IMPORTANCE,
    RETENTION_CRITICAL_TYPES,
    BACKPROP_DELAYED_CONSOLIDATION,
)
 
# Maximum number of ANN neighbours to inspect per memory during clustering.
# Increasing this raises recall at the cost of more comparisons, but it
# remains O(N * _ANN_K) rather than O(NÃÂ²).
_ANN_K = 16
 
 
class DreamCycle:
 
    def __init__(self, embedder=None, graph=None, index=None):
        self.embedder = embedder or Embedder()
        self.graph    = graph    or MemoryGraph()
        self.index    = index
        self.backprop      = ImportanceBackpropEngine(self.graph)
        self.belief_system = BeliefSystem(graph=self.graph)
 
    def run(self):
        logger.info("DreamCycle started...")
 
        memories = load_all_memories()
 
        if len(memories) < 5:
            logger.info("Not enough memories for consolidation.")
            return
 
        self._ensure_embeddings(memories)
 
        clusters = self._cluster(memories)
 
        self._create_concepts(clusters)
 
        # Reload so concept memories and any importance changes from this
        # session are visible to backprop before pruning runs.
        memories = load_all_memories()
 
        self._run_backprop_pass(memories)
 
        self._flush_belief_events(memories)

        self._prune(memories)
 
        logger.info("DreamCycle complete.")
 
    def _ensure_embeddings(self, memories):
        for m in memories:
            if not m.embedding:
                m.embedding = self.embedder.encode(m.content)
                save_memory(m)
 
    # ------------------------------------------------------------------
    # Kept for _create_concepts() Ã¢ÂÂ per-member similarity to graph.connect
    # ------------------------------------------------------------------
 
    def _cosine(self, a, b) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        na  = math.sqrt(sum(x * x for x in a))
        nb  = math.sqrt(sum(x * x for x in b))
        if na == 0 or nb == 0:
            return 0.0
        return dot / (na * nb)
 
    # ------------------------------------------------------------------
    # FAISS-accelerated clustering  (replaces O(NÃÂ²) loop)
    # ------------------------------------------------------------------
 
    def _cluster(self, memories) -> list:
        """
        Group raw memories by semantic similarity using FAISS ANN search.
 
        Algorithm
        ---------
        1. Embed all raw memories into a float32 matrix and L2-normalise so
           IndexFlatIP becomes a cosine similarity index.
        2. Search each memory's top-_ANN_K neighbours in one batched call.
        3. Greedy assignment: the first unseen memory that passes the
           CONCEPT_SIMILARITY_THRESHOLD becomes the cluster seed; all
           qualifying neighbours in its ANN result are folded in immediately.
           This mirrors the original single-pass greedy strategy but runs in
           O(N * _ANN_K) time instead of O(NÃÂ²).
 
        Falls back gracefully: if any memory lacks an embedding it is placed
        in a singleton cluster so it is never silently dropped.
        """
        raw_memories = [m for m in memories if not m.is_concept]
 
        if not raw_memories:
            return []
 
        if len(raw_memories) == 1:
            return [raw_memories]
 
        # ----------------------------------------------------------------
        # Build a normalised embedding matrix.
        # Memories without embeddings are separated out and handled below.
        # ----------------------------------------------------------------
        valid   = [m for m in raw_memories if m.embedding]
        invalid = [m for m in raw_memories if not m.embedding]
 
        if not valid:
            # Nothing to cluster Ã¢ÂÂ return singletons.
            return [[m] for m in raw_memories]
 
        dim     = len(valid[0].embedding)
        vectors = np.array([m.embedding for m in valid], dtype=np.float32)
        _faiss.normalize_L2(vectors)  # in-place normalisation for cosine
 
        # ----------------------------------------------------------------
        # Temporary FAISS index Ã¢ÂÂ IndexFlatIP is exact but still O(N * k)
        # for the batched search; no training required.
        # ----------------------------------------------------------------
        ann_index = _faiss.IndexFlatIP(dim)
        ann_index.add(vectors)
 
        k             = min(len(valid), _ANN_K)
        scores, idxs  = ann_index.search(vectors, k)   # (N, k) each
 
        # ----------------------------------------------------------------
        # Greedy single-pass assignment.
        # ----------------------------------------------------------------
        used     = set()
        clusters = []
 
        for i, m1 in enumerate(valid):
            if m1.id in used:
                continue
 
            cluster = [m1]
            used.add(m1.id)
 
            for pos in range(k):
                j   = int(idxs[i][pos])
                sim = float(scores[i][pos])
 
                if j == i or j < 0 or j >= len(valid):
                    continue
 
                m2 = valid[j]
                if m2.id not in used and sim >= CONCEPT_SIMILARITY_THRESHOLD:
                    cluster.append(m2)
                    used.add(m2.id)
 
            clusters.append(cluster)
 
        # Memories that had no embedding become singleton clusters so the
        # rest of the pipeline can still process them (even if unlikely to
        # form a concept on their own).
        for m in invalid:
            clusters.append([m])
 
        return clusters
 
    # ------------------------------------------------------------------
    # Summary via LLM
    # ------------------------------------------------------------------
 
    def _validated_summary(self, texts) -> "str | None":
        prompt = (
            "Summarize ONLY explicit shared themes.\n"
            "Do not invent facts.\n"
            "One sentence only.\n\n"
        )
        for t in texts:
            prompt += f"- {t}\n"
        prompt += "\nSummary:"
 
        try:
            raw_summary = run_llama(prompt).strip()
 
            if ":" in raw_summary and len(raw_summary.split(":")[0]) < 25:
                summary = raw_summary.split(":", 1)[1].strip()
            else:
                summary = raw_summary
 
            summary = re.sub(r'^["\']|["\']$', '', summary).strip()
 
            return summary if len(summary) >= 5 else None
        except Exception:
            return None
 
    # ------------------------------------------------------------------
    # Concept creation
    # ------------------------------------------------------------------
 
    def _create_concepts(self, clusters):
        for cluster in clusters:
            if len(cluster) < CONCEPT_MIN_SIZE:
                continue
 
            texts   = [m.content for m in cluster]
            summary = self._validated_summary(texts)
 
            if not summary:
                continue
 
            concept = create_concept(summary, cluster, self.embedder)
 
            if concept is None:
                continue
 
            logger.info(f"[CONCEPT] {concept.content}")
 
            if self.index is not None and concept.embedding:
                self.index.add(concept.embedding, concept.id)
 
            for m in cluster:
                similarity = self._cosine(concept.embedding, m.embedding)
                self.graph.connect(
                    concept.id,
                    m.id,
                    base_weight=3.0,
                    similarity=similarity,
                )
 
    # ------------------------------------------------------------------
    # Pruning
    # ------------------------------------------------------------------
 

    # ------------------------------------------------------------------
    # Retrospective Importance Revaluation (v15)
    # ------------------------------------------------------------------

    def _run_backprop_pass(self, memories: list) -> None:
        """
        Run the Retrospective Importance Revaluation engine.

        Sequence
        --------
        1. If delayed consolidation is enabled, flush updates queued during
           the previous DreamCycle or during live inference.
        2. Run a fresh full-pass calculation over the current memory set,
           queueing or applying updates per the BACKPROP_DELAYED_CONSOLIDATION
           config flag.

        Called *after* concept creation (so concept nodes can act as triggers)
        and *before* pruning (so retroactively elevated memories survive).
        """
        logger.info("[BACKPROP] Starting Retrospective Importance Revaluation pass...")

        if BACKPROP_DELAYED_CONSOLIDATION:
            flushed = self.backprop.flush_queue(memories)
            if flushed:
                memories[:] = load_all_memories()

        n = self.backprop.run_full_pass(memories)

        if not BACKPROP_DELAYED_CONSOLIDATION:
            logger.info(f"[BACKPROP] Immediate mode: {n} importance(s) updated.")
        else:
            logger.info(f"[BACKPROP] Delayed mode: {n} update(s) queued for next DreamCycle.")

    # ------------------------------------------------------------------
    # Retention policy helpers
    # ------------------------------------------------------------------

    def _flush_belief_events(self, memories: list) -> None:
        """
        Apply any pending ContradictionEvents during DreamCycle consolidation.

        This is the GhostMind safety net: if GhostMind arbitration is enabled
        but GhostMind has not yet consumed the queue, we apply the pending
        events here so that no belief resolution is lost between cycles.

        Also runs in non-GhostMind mode (events will already be resolved, so
        the call is a no-op, but it keeps the code path uniform).
        """
        memory_map = {m.id: m for m in memories}
        self.belief_system.flush_pending_ghostmind_events(memory_map)

    def _apply_auto_retention(self, memories) -> None:
        """
        Auto-elevate memories to CRITICAL when they meet importance and type
        thresholds, persisting the updated policy back to the store.

        This runs before pruning so that newly critical memories are never
        accidentally deleted in the same DreamCycle pass.
        """
        for m in memories:
            if m.is_concept:
                continue

            rp = RetentionPolicy.from_dict(m.retention_policy)
            if rp.class_ == RetentionClass.CRITICAL:
                continue  # already at max protection

            if (
                m.importance >= RETENTION_AUTO_CRITICAL_IMPORTANCE
                and m.type in RETENTION_CRITICAL_TYPES
            ):
                m.retention_policy = RetentionPolicy.make_critical(
                    source="dreamcycle",
                    reason="high_importance",
                ).to_dict()
                save_memory(m)
                logger.info(
                    f"[RETENTION] Auto-elevated memory {m.id[:8]}... to CRITICAL "
                    f"(importance={m.importance:.2f}, type={m.type})"
                )

    def _audit_critical_memories(self, memories) -> None:
        """Emit telemetry for all CRITICAL memories (observability only)."""
        if not RETENTION_AUDIT_LOG_ENABLED:
            return
        critical = [
            m for m in memories
            if RetentionPolicy.from_dict(m.retention_policy).class_ == RetentionClass.CRITICAL
        ]
        if critical:
            logger.info(
                f"[RETENTION] {len(critical)} CRITICAL memories protected from pruning: "
                + ", ".join(f"{m.id[:8]}({m.type})" for m in critical)
            )

    # ------------------------------------------------------------------
    # Pruning Ã¢ÂÂ retention-policy aware
    # ------------------------------------------------------------------

    def _prune(self, memories):
        """
        Delete low-importance raw memories, respecting retention policies.

        Memories with a PROTECTED or CRITICAL retention_policy are never
        deleted regardless of their importance score.  This prevents rare
        but high-value memories (security alerts, one-time events,
        catastrophic precursor logs) from being lost during consolidation.
        """
        self._apply_auto_retention(memories)
        self._audit_critical_memories(memories)

        removed = 0
        skipped_protected = 0

        for m in memories:
            if m.is_concept:
                continue
            if m.importance >= PRUNE_IMPORTANCE_THRESHOLD:
                continue

            if not is_prune_safe(m):
                skipped_protected += 1
                rp = RetentionPolicy.from_dict(m.retention_policy)
                logger.info(
                    f"[RETENTION] Skipped prune for memory {m.id[:8]}... "
                    f"Ã¢ÂÂ {rp.describe()}"
                )
                continue

            delete_memory(m.id)
            removed += 1

        logger.info(
            f"[PRUNE] Removed {removed} memories, "
            f"protected {skipped_protected} by retention policy"
        )