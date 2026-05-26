# config/memory_config.py

SCHEMA_VERSION = 4  # Bumped from 3 to support cognitive homeostasis tracking

# Retrieval Thresholds
SIMILARITY_THRESHOLD = 0.40
MAX_RETRIEVED_MEMORIES = 10

# Reinforcement & Penalties
CONTRADICTION_PENALTY = 0.6
LLM_USAGE_BOOST = 0.15

# Cognitive Homeostasis Parameters
ACTIVATION_DECAY_CONSTANT = 60.0    # Time constant (seconds) for exponential decay of retrieval pressure
ACTIVATION_PENALTY_WEIGHT = 0.4     # Sensitivity multiplier for the activation penalty
NOVELTY_WEIGHT            = 0.25    # Weight of the novelty bonus
NOVELTY_DECAY_CONSTANT    = 300.0   # Time constant (seconds) after which memory novelty fades