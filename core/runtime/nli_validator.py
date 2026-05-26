"""
nli_validator.py
 
Thin wrapper around a CrossEncoder NLI model for semantic contradiction
detection.
 
The model is lazy-loaded on first call and reused for all subsequent checks.
This avoids paying the load cost at startup and keeps the module importable
even if the model file is not yet present (it will only error on first use).
 
Model
-----
cross-encoder/nli-MiniLM2-L6-H768
 
    - 3-class NLI: contradiction | entailment | neutral
    - ~90 MB on disk
    - Single forward pass per pair (~5–15 ms on CPU)
    - Trained specifically for premise/hypothesis inference — more reliable
      than prompting a generative LLM with a YES/NO question.
 
Label order
-----------
The CrossEncoder variant used here outputs logits in the order:
    [contradiction, entailment, neutral]
after softmax these become probabilities that sum to 1.0.
 
Threshold
---------
Controlled by NLI_CONTRADICTION_THRESHOLD in memory_config.py (default 0.7).
Lower → more sensitive (catches subtle contradictions, higher false-positive
risk).  Higher → more conservative (only flags clear-cut opposites).
"""
 
from sentence_transformers import CrossEncoder
 
from config.memory_config import NLI_CONTRADICTION_THRESHOLD
from config.model_config import NLI_MODEL_PATH
from core.infrastructure.logger import logger
 
# Singleton model instance — populated on first call to is_contradiction().
_model: "CrossEncoder | None" = None
 
# Label order matches the cross-encoder/nli-MiniLM2-L6-H768 checkpoint.
_LABEL_ORDER = ["contradiction", "entailment", "neutral"]
 
 
def _get_model() -> CrossEncoder:
    """Load the NLI CrossEncoder once and cache it for the process lifetime."""
    global _model
    if _model is None:
        logger.info(f"[NLI] Loading model from {NLI_MODEL_PATH}")
        _model = CrossEncoder(NLI_MODEL_PATH, num_labels=3)
        logger.info("[NLI] Model loaded.")
    return _model
 
 
def is_contradiction(a: str, b: str) -> bool:
    """
    Return True if statements *a* and *b* semantically contradict each other.
 
    Parameters
    ----------
    a, b : str
        The two memory content strings to compare.
 
    Returns
    -------
    bool
        True  — NLI contradiction score ≥ NLI_CONTRADICTION_THRESHOLD.
        False — otherwise, or if the model call raises an exception.
 
    Raises
    ------
    Does not raise.  All exceptions are caught and logged so that the
    caller's heuristic fallback can take over without crashing the engine.
    """
    try:
        model  = _get_model()
        # predict() returns a (1, 3) array; [0] gives the single pair's scores.
        scores = model.predict([(a, b)], apply_softmax=True)[0]
        label_scores = dict(zip(_LABEL_ORDER, scores))
 
        contradiction_score = float(label_scores["contradiction"])
        logger.info(
            f"[NLI] contradiction={contradiction_score:.3f} | "
            f"entailment={label_scores['entailment']:.3f} | "
            f"neutral={label_scores['neutral']:.3f} | "
            f"A='{a[:60]}' B='{b[:60]}'"
        )
        return contradiction_score >= NLI_CONTRADICTION_THRESHOLD
 
    except Exception as e:
        logger.warning(f"[NLI] Model call failed: {e}")
        raise  # re-raise so engine.py can apply its heuristic fallback