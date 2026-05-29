"""
nli_validator.py — DreamCloud Feature 6: Full 3-class NLI

Changes from v1
---------------
v2 — Feature 6: expose full 3-class NLI result via classify_relationship().

    is_contradiction()       — unchanged; backward-compatible bool check.
    classify_relationship()  — NEW; returns NLIClassification with all scores
                               and the dominant NLIRelationship label.

Model
-----
cross-encoder/nli-MiniLM2-L6-H768

    - 3-class NLI: contradiction | entailment | neutral
    - ~90 MB on disk
    - Single forward pass per pair (~5–15 ms on CPU)

Label order
-----------
The CrossEncoder outputs logits in the order:
    [contradiction, entailment, neutral]
After softmax these become probabilities summing to 1.0.

Thresholds (configurable in memory_config.py)
---------------------------------------------
NLI_CONTRADICTION_THRESHOLD  — min contradiction score → CONTRADICTION  (0.70)
NLI_ENTAILMENT_THRESHOLD     — min entailment score    → ENTAILMENT     (0.60)
NLI_NEUTRAL_THRESHOLD        — fallback                → NEUTRAL        (0.50)

Decision logic (in priority order):
    1. If contradiction_score >= NLI_CONTRADICTION_THRESHOLD → CONTRADICTION
    2. elif entailment_score >= NLI_ENTAILMENT_THRESHOLD     → ENTAILMENT
    3. else                                                   → NEUTRAL

This priority ensures that a clearly contradicting pair is never mis-classified
as entailment even if the entailment score is also elevated.
"""

from dataclasses import dataclass
from enum import Enum

from sentence_transformers import CrossEncoder

from config.memory_config import (
    NLI_CONTRADICTION_THRESHOLD,
    NLI_ENTAILMENT_THRESHOLD,
)
from config.model_config import NLI_MODEL_PATH
from core.infrastructure.logger import logger

# Singleton model instance — populated on first use.
_model: "CrossEncoder | None" = None

# Label order matches cross-encoder/nli-MiniLM2-L6-H768.
_LABEL_ORDER = ["contradiction", "entailment", "neutral"]


# ---------------------------------------------------------------------------
# Enumerations & result dataclass
# ---------------------------------------------------------------------------

class NLIRelationship(str, Enum):
    """
    The dominant semantic relationship between two statements.

    CONTRADICTION — the statements are mutually exclusive.
    ENTAILMENT    — one statement implies / confirms the other.
    NEUTRAL       — the statements are logically compatible but independent.
    """
    CONTRADICTION = "contradiction"
    ENTAILMENT    = "entailment"
    NEUTRAL       = "neutral"


@dataclass
class NLIClassification:
    """
    Full result of a 3-class NLI forward pass.

    Attributes
    ----------
    relationship        : dominant NLIRelationship label
    contradiction_score : softmax probability [0, 1]
    entailment_score    : softmax probability [0, 1]
    neutral_score       : softmax probability [0, 1]
    """
    relationship:        NLIRelationship
    contradiction_score: float
    entailment_score:    float
    neutral_score:       float

    def is_contradiction(self) -> bool:
        return self.relationship == NLIRelationship.CONTRADICTION

    def is_entailment(self) -> bool:
        return self.relationship == NLIRelationship.ENTAILMENT

    def is_neutral(self) -> bool:
        return self.relationship == NLIRelationship.NEUTRAL


# Sentinel returned when the model call fails (safe fallback to NEUTRAL).
_NEUTRAL_FALLBACK = NLIClassification(
    relationship=NLIRelationship.NEUTRAL,
    contradiction_score=0.0,
    entailment_score=0.0,
    neutral_score=1.0,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _get_model() -> CrossEncoder:
    """Load the NLI CrossEncoder once and cache it for the process lifetime."""
    global _model
    if _model is None:
        logger.info(f"[NLI] Loading model from {NLI_MODEL_PATH}")
        _model = CrossEncoder(NLI_MODEL_PATH, num_labels=3)
        logger.info("[NLI] Model loaded.")
    return _model


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify_relationship(a: str, b: str) -> NLIClassification:
    """
    Run a full 3-class NLI forward pass and return a structured result.

    Parameters
    ----------
    a, b : str
        The two memory content strings to compare.

    Returns
    -------
    NLIClassification
        Contains the dominant relationship and all three probability scores.
        Returns _NEUTRAL_FALLBACK on model failure (never raises).

    Decision priority
    -----------------
    1. contradiction_score >= NLI_CONTRADICTION_THRESHOLD  → CONTRADICTION
    2. entailment_score    >= NLI_ENTAILMENT_THRESHOLD     → ENTAILMENT
    3. otherwise                                            → NEUTRAL
    """
    try:
        model  = _get_model()
        scores = model.predict([(a, b)], apply_softmax=True)[0]
        label_scores = dict(zip(_LABEL_ORDER, scores))

        contradiction_score = float(label_scores["contradiction"])
        entailment_score    = float(label_scores["entailment"])
        neutral_score       = float(label_scores["neutral"])

        logger.info(
            f"[NLI] contradiction={contradiction_score:.3f} | "
            f"entailment={entailment_score:.3f} | "
            f"neutral={neutral_score:.3f} | "
            f"A='{a[:60]}' B='{b[:60]}'"
        )

        # Priority: contradiction > entailment > neutral
        if contradiction_score >= NLI_CONTRADICTION_THRESHOLD:
            rel = NLIRelationship.CONTRADICTION
        elif entailment_score >= NLI_ENTAILMENT_THRESHOLD:
            rel = NLIRelationship.ENTAILMENT
        else:
            rel = NLIRelationship.NEUTRAL

        return NLIClassification(
            relationship=rel,
            contradiction_score=contradiction_score,
            entailment_score=entailment_score,
            neutral_score=neutral_score,
        )

    except Exception as exc:
        logger.warning(f"[NLI] classify_relationship failed: {exc}")
        return _NEUTRAL_FALLBACK


def is_contradiction(a: str, b: str) -> bool:
    """
    Return True if statements *a* and *b* semantically contradict each other.

    This is a thin backward-compatible wrapper around classify_relationship().
    Raises on model failure so that engine.py can apply its heuristic fallback.

    Parameters
    ----------
    a, b : str

    Returns
    -------
    bool — True if contradiction_score >= NLI_CONTRADICTION_THRESHOLD.
    """
    try:
        result = classify_relationship(a, b)
        return result.is_contradiction()
    except Exception as exc:
        logger.warning(f"[NLI] Model call failed: {exc}")
        raise  # re-raise so engine.py heuristic fallback activates
