# ==============================================================================
# DreamCloud Memory Configuration
# ==============================================================================

# File Paths
MEMORY_PATH = "data/memories/"
FAISS_PATH  = "data/faiss/"
LOG_PATH    = "data/logs/"
CACHE_PATH  = "data/cache/"
GRAPH_PATH  = "data/graph/"

# Core Settings
TOP_K          = 10
EMBEDDING_DIM  = 384
SCHEMA_VERSION = 4  # Bumped to support cognitive homeostasis tracking

# Retrieval & Ranking
SIMILARITY_THRESHOLD   = 0.40
MAX_RETRIEVED_MEMORIES = 10

# Reinforcement & Penalties
REINFORCEMENT_THRESHOLD  = 0.55
MAX_REINFORCED_PER_QUERY = 3
REINFORCEMENT_COOLDOWN   = 10
CONTRADICTION_PENALTY    = 0.6  # Updated
LLM_USAGE_BOOST          = 0.15 # Updated

# Memory Decay
WEAK_MEMORY_THRESHOLD = 0.3
WEAK_DECAY_RATE       = 0.98

# Cognitive Homeostasis Parameters
ACTIVATION_DECAY_CONSTANT = 60.0    # Time constant (s) for retrieval pressure decay
ACTIVATION_PENALTY_WEIGHT = 0.4     # Sensitivity multiplier for activation penalty
NOVELTY_WEIGHT            = 0.25    # Weight of the novelty bonus
NOVELTY_DECAY_CONSTANT    = 300.0   # Time constant (s) for memory novelty fading

# Persistence
FAISS_INDEX_FILE = f"{FAISS_PATH}index.faiss"
FAISS_META_FILE  = f"{FAISS_PATH}index_meta.json"

# DreamCycle
CONCEPT_SIMILARITY_THRESHOLD = 0.75
CONCEPT_MIN_SIZE             = 3
CONCEPT_MAX_SOURCES          = 12
CONCEPT_CONFIDENCE_THRESHOLD = 0.45

# Pruning
PRUNE_IMPORTANCE_THRESHOLD = 0.2

# NLI Contradiction Detection
# Minimum contradiction probability (0.0–1.0) for the NLI model to flag a pair.
NLI_CONTRADICTION_THRESHOLD = 0.7

# =============================================================================
# Telemetry Configuration
# =============================================================================

# Shannon entropy threshold (bits) below which retrieval is considered
# over-concentrated and autonomic regulation is triggered.
ENTROPY_THRESHOLD = 2.0

# Sliding window (seconds) for retrieval-frequency tracking.
TELEMETRY_WINDOW_SECONDS = 3600  # 1 hour

# Cooldown between autonomic regulation steps (seconds).
REGULATION_COOLDOWN_SECONDS = 30

# Maximum factor by which NOVELTY_WEIGHT can be increased during correction.
MAX_DIVERSITY_BONUS_FACTOR = 3.0

# Minimum ACTIVATION_PENALTY_WEIGHT during correction (floor).
MIN_ACTIVATION_PENALTY_WEIGHT = 0.05

# Exploratory mode: when active, this many random memories are injected
# into the retrieval result regardless of similarity threshold.
EXPLORATORY_INJECTION_COUNT = 3