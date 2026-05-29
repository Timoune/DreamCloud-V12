# DreamCloud V16 - Typed Graph Edges + Belief Systems

MEMORY_PATH = "data/memories/"
FAISS_PATH  = "data/faiss/"
LOG_PATH    = "data/logs/"
CACHE_PATH  = "data/cache/"
GRAPH_PATH  = "data/graph/"

TOP_K         = 10
EMBEDDING_DIM = 384
SCHEMA_VERSION = 8   # v8 adds belief tracking (contradiction_count, entailment_count, belief_version)

# ===========================================================================
# Activation Decay Architectures (V14)
# ===========================================================================

DECAY_STRATEGY = "hybrid"  # Options: "global_sweep", "lazy", "hybrid"

# Global Sweep Decay
GLOBAL_DECAY_RATE = 0.985          # Multiplicative decay per sweep
SWEEP_INTERVAL_SECONDS = 300       # Every 5 minutes

# Lazy Decay
LAZY_DECAY_LAMBDA = 0.0015         # Decay constant (higher = faster decay)

# Hybrid Model
HYBRID_CONSOLIDATION_INTERVAL = 1800  # 30 min full consolidation
ANOMALY_DETECTION_ENABLED = True
RUNAWAY_REINFORCEMENT_THRESHOLD = 25.0
OVER_ACTIVATION_THRESHOLD = 8.0
MONOPOLIZATION_THRESHOLD = 0.65     # % of top activations from single cluster

# reinforcement
REINFORCEMENT_THRESHOLD    = 0.55
MAX_REINFORCED_PER_QUERY   = 3
REINFORCEMENT_COOLDOWN     = 10

WEAK_MEMORY_THRESHOLD = 0.3
WEAK_DECAY_RATE       = 0.98

CONTRADICTION_PENALTY = 0.8
LLM_USAGE_BOOST       = 1.5

# persistence
FAISS_INDEX_FILE = f"{FAISS_PATH}index.faiss"
FAISS_META_FILE  = f"{FAISS_PATH}index_meta.json"

# dreamcycle
CONCEPT_SIMILARITY_THRESHOLD  = 0.75
CONCEPT_MIN_SIZE              = 3
CONCEPT_MAX_SOURCES           = 12
CONCEPT_CONFIDENCE_THRESHOLD  = 0.45

# pruning
PRUNE_IMPORTANCE_THRESHOLD = 0.2

# nli contradiction detection
NLI_CONTRADICTION_THRESHOLD = 0.7

# ===========================================================================
# Cognitive Homeostasis / Telemetry (kept for compatibility)
# ===========================================================================

ACTIVATION_DECAY_CONSTANT = 60.0
ACTIVATION_PENALTY_WEIGHT = 0.4
NOVELTY_WEIGHT = 0.25
NOVELTY_DECAY_CONSTANT = 300.0
ENTROPY_THRESHOLD = 2.0
TELEMETRY_WINDOW_SECONDS = 3600
REGULATION_COOLDOWN_SECONDS = 30
MAX_DIVERSITY_BONUS_FACTOR = 3.0
MIN_ACTIVATION_PENALTY_WEIGHT = 0.05
EXPLORATORY_INJECTION_COUNT = 3

# ===========================================================================
# Retention Policies (v6)
# ===========================================================================

RETENTION_AUDIT_LOG_ENABLED = True
RETENTION_EXPIRY_GRACE_SECONDS = 300
RETENTION_AUTO_CRITICAL_IMPORTANCE = 4.5
RETENTION_CRITICAL_TYPES = {"emotional", "episodic"}

# ===========================================================================
# Retrospective Importance Revaluation / Backpropagation (v15)
# ===========================================================================

BACKPROP_MAX_DEPTH             = 3
BACKPROP_DEPTH_DECAY           = 0.6
BACKPROP_IMPORTANCE_CAP        = 4.8
BACKPROP_MIN_DELTA             = 0.05
BACKPROP_CONFIDENCE_THRESHOLD  = 0.1
BACKPROP_TRIGGER_THRESHOLD     = 3.0
BACKPROP_DELAYED_CONSOLIDATION = False
BACKPROP_MAX_UPDATES_PER_PASS  = 50

# Edge weights â how strongly importance propagates through each typed edge.
#
# Causal (1.0):         Strongest: if B is the effect, A (the cause) is critical.
# Goal dependency(0.90):Goals depend on preconditions; preconditions inherit urgency.
# Derived from (0.85):  Source material directly contributed; inherit most signal.
# Temporal (0.70):      Precursor relationship; causality is uncertain but plausible.
# Supports (0.65):      Corroborating evidence; moderate inheritance.
# Semantic (0.50):      Pure similarity; weakest legitimate propagation path.
# Contradicts (0.20):   Minimal backprop â opposing beliefs should not freely amplify.
BACKPROP_EDGE_WEIGHTS = {
    "causal":          1.00,
    "goal_dependency": 0.90,
    "derived_from":    0.85,
    "temporal":        0.70,
    "supports":        0.65,
    "semantic":        0.50,
    "contradicts":     0.20,
}

# ===========================================================================
# Typed Graph Edges (Feature 5)
# ===========================================================================

# All valid typed edge labels (graph_manager enforces this set).
# Legacy edges without an edge_type are treated as "semantic".
GRAPH_VALID_EDGE_TYPES = frozenset({
    "semantic",
    "causal",
    "temporal",
    "derived_from",
    "supports",
    "contradicts",
    "goal_dependency",
})

# Directed edge types: edges that carry a meaningful AâB ordering.
# Stored in a separate directed structure in addition to the undirected graph.
GRAPH_DIRECTED_TYPES = frozenset({"causal", "goal_dependency", "temporal"})

# ===========================================================================
# Contradiction Handling & Belief Systems (Feature 6)
# ===========================================================================

# Full 3-class NLI thresholds.
# Scores are softmax probabilities that sum to 1.0.
NLI_ENTAILMENT_THRESHOLD = 0.60   # min entailment score â treat as ENTAILMENT
NLI_NEUTRAL_THRESHOLD    = 0.50   # min neutral score   â treat as NEUTRAL (fallback)

# Persistent queue for ContradictionEvents.
CONTRADICTION_QUEUE_PATH = "data/cache/contradiction_queue.json"

# Entailment resolution tuning.
ENTAILMENT_RELIABILITY_BOOST = 0.05   # reliability delta per confirmed entailment
ENTAILMENT_IMPORTANCE_BOOST  = 0.10   # importance delta for the existing candidate
MERGE_SIMILARITY_THRESHOLD   = 0.85   # cosine similarity threshold for concept merge

# GhostMind arbitration hook.
# False  â ContradictionEvents are applied immediately by BeliefSystem.
# True   â Events are queued; GhostMind is expected to consume them.
GHOSTMIND_ARBITRATION_ENABLED = False
