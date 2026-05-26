MEMORY_PATH = "data/memories/"
FAISS_PATH  = "data/faiss/"
LOG_PATH    = "data/logs/"
CACHE_PATH  = "data/cache/"
GRAPH_PATH  = "data/graph/"
 
TOP_K         = 10
EMBEDDING_DIM = 384
SCHEMA_VERSION = 3   # bumped: v3 adds reliability, reinforcement_count, typed memory
 
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
# Minimum contradiction probability (0.0–1.0) for the NLI model to flag a pair.
# Lower  → more sensitive, higher false-positive risk.
# Higher → more conservative, only clear-cut opposites are flagged.
NLI_CONTRADICTION_THRESHOLD = 0.7