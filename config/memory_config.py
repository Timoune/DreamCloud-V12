# DreamCloud V14 - Activation Decay Architectures

MEMORY_PATH = "data/memories/"
FAISS_PATH  = "data/faiss/"
LOG_PATH    = "data/logs/"
CACHE_PATH  = "data/cache/"
GRAPH_PATH  = "data/graph/"

TOP_K         = 10
EMBEDDING_DIM = 384
SCHEMA_VERSION = 5   # v5 adds full decay_strategy support

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
