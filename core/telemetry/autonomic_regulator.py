# core/telemetry/autonomic_regulator.py
"""
AutonomicRegulator — translates telemetry signals into autonomic
parameter adjustments, closing the feedback loop.

Trigger condition:
    retrieval_entropy < threshold

Response:
    1. increase_diversity_bonus()   → raises NOVELTY_WEIGHT
    2. reduce_activation_weight()   → lowers ACTIVATION_PENALTY_WEIGHT
    3. trigger_exploratory_mode()   → injects exploratory memories into retrieval

The regulator gradually returns parameters to baseline when entropy recovers.
"""
import time
import math
from core.infrastructure.logger import logger


class AutonomicRegulator:
    """
    Holds references to the configurable parameters and adjusts them
    in response to entropy signals.
    """

    def __init__(
        self,
        entropy_monitor: "ClusterEntropyMonitor",
        config_module=None,  # the memory_config module
    ):
        self._entropy_monitor = entropy_monitor
        self._config = config_module

        # Baseline values (snapshot at init)
        self._baseline_novelty_weight = getattr(config_module, "NOVELTY_WEIGHT", 0.25)
        self._baseline_activation_penalty = getattr(config_module, "ACTIVATION_PENALTY_WEIGHT", 0.4)

        # Current regulated values
        self._novelty_weight = self._baseline_novelty_weight
        self._activation_penalty_weight = self._baseline_activation_penalty

        # State
        self._exploratory_mode: bool = False
        self._last_regulation: float = 0.0
        self._regulation_cooldown: float = 30.0  # seconds between adjustments

    # ------------------------------------------------------------------
    # Public API — called by the engine after each retrieval
    # ------------------------------------------------------------------
    def regulate(self) -> dict:
        """
        Evaluate entropy and apply regulation if needed.

        Returns a dict describing the action taken (useful for logging/dashboards).
        """
        now = time.time()
        if now - self._last_regulation < self._regulation_cooldown:
            return {"action": "cooldown", "timestamp": now}

        status = self._entropy_monitor.evaluate()
        entropy = status["entropy"]
        is_low = status["is_low_diversity"]

        action = {"action": "none", "timestamp": now, "entropy": entropy}

        if is_low:
            action = self._apply_correction(entropy)
        else:
            action = self._relax_correction(entropy)

        self._last_regulation = now
        return action

    # ------------------------------------------------------------------
    # Regulation primitives
    # ------------------------------------------------------------------
    def increase_diversity_bonus(self, factor: float = 1.5) -> None:
        """
        Raises NOVELTY_WEIGHT to give less-recently-seen memories a
        stronger scoring boost, counteracting concentration.
        """
        self._novelty_weight = min(
            self._baseline_novelty_weight * factor,
            2.0,  # hard cap
        )
        if self._config:
            self._config.NOVELTY_WEIGHT = self._novelty_weight
        logger.info(f"[Regulator] diversity_bonus ↑ → NOVELTY_WEIGHT={self._novelty_weight:.3f}")

    def reduce_activation_weight(self, factor: float = 0.5) -> None:
        """
        Lowers ACTIVATION_PENALTY_WEIGHT so frequently-retrieved memories
        are penalised less, allowing other memories to compete.
        """
        self._activation_penalty_weight = max(
            self._baseline_activation_penalty * factor,
            0.05,  # floor
        )
        if self._config:
            self._config.ACTIVATION_PENALTY_WEIGHT = self._activation_penalty_weight
        logger.info(
            f"[Regulator] activation_weight ↓ → "
            f"ACTIVATION_PENALTY_WEIGHT={self._activation_penalty_weight:.3f}"
        )

    def trigger_exploratory_mode(self) -> None:
        """
        Signals the retrieval pipeline to inject a random-exploration
        bonus, pulling in memories that would otherwise be outside the
        similarity threshold.
        """
        self._exploratory_mode = True
        logger.info("[Regulator] exploratory_mode = ON")

    def reset_exploratory_mode(self) -> None:
        self._exploratory_mode = False
        logger.info("[Regulator] exploratory_mode = OFF")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------
    def _apply_correction(self, entropy: float) -> dict:
        """Called when entropy is below threshold."""
        self.increase_diversity_bonus(factor=1.5)
        self.reduce_activation_weight(factor=0.5)
        self.trigger_exploratory_mode()
        return {
            "action": "correct",
            "entropy": entropy,
            "novelty_weight": self._novelty_weight,
            "activation_penalty_weight": self._activation_penalty_weight,
            "exploratory_mode": True,
            "timestamp": time.time(),
        }

    def _relax_correction(self, entropy: float) -> dict:
        """Gradually return parameters to baseline when entropy is healthy."""
        # Exponential moving average back to baseline
        decay = 0.1
        self._novelty_weight += (self._baseline_novelty_weight - self._novelty_weight) * decay
        self._activation_penalty_weight += (
            self._baseline_activation_penalty - self._activation_penalty_weight
        ) * decay

        if self._config:
            self._config.NOVELTY_WEIGHT = self._novelty_weight
            self._config.ACTIVATION_PENALTY_WEIGHT = self._activation_penalty_weight

        if self._exploratory_mode:
            self.reset_exploratory_mode()

        return {
            "action": "relax",
            "entropy": entropy,
            "novelty_weight": self._novelty_weight,
            "activation_penalty_weight": self._activation_penalty_weight,
            "exploratory_mode": False,
            "timestamp": time.time(),
        }

    # ------------------------------------------------------------------
    # Properties for the engine to read
    # ------------------------------------------------------------------
    @property
    def novelty_weight(self) -> float:
        return self._novelty_weight

    @property
    def activation_penalty_weight(self) -> float:
        return self._activation_penalty_weight

    @property
    def is_exploratory(self) -> bool:
        return self._exploratory_mode