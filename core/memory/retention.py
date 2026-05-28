"""
Retention Policies & Archival Safety — DreamCloud v6.

Replaces boolean archival flags with structured policy objects that carry
provenance, classification, and expiry information.

Policy classes
--------------
VOLATILE    — default; eligible for normal pruning/decay
STANDARD    — retained longer than volatile; pruning still allowed
PROTECTED   — never pruned but subject to expiry
CRITICAL    — permanent; never pruned, never expired

Policy sources
--------------
user        — explicitly set by the user
system      — set by the runtime (security alerts, anomaly logs, etc.)
dreamcycle  — assigned during DreamCycle consolidation
llm         — assigned by the LLM during extraction

Key insight: low frequency != low value.
A security alert fired once is more important than a preference
retrieved daily.  Retention class captures that distinction explicitly.
"""

from dataclasses import dataclass
from typing import Optional
import time


# ---------------------------------------------------------------------------
# Class / Source / Reason constants
# ---------------------------------------------------------------------------

class RetentionClass:
    """Ordered ladder of protection strength."""
    VOLATILE   = "volatile"   # default — normal lifecycle
    STANDARD   = "standard"   # retained longer; still prunable
    PROTECTED  = "protected"  # never pruned; may expire
    CRITICAL   = "critical"   # permanent; immune to pruning and expiry

    # Ordered list, weakest → strongest
    _LADDER = ["volatile", "standard", "protected", "critical"]

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls._LADDER

    @classmethod
    def strength(cls, value: str) -> int:
        """Return numeric strength (higher = more protected)."""
        try:
            return cls._LADDER.index(value)
        except ValueError:
            return 0


class RetentionSource:
    """Who assigned the retention policy."""
    USER       = "user"
    SYSTEM     = "system"
    DREAMCYCLE = "dreamcycle"
    LLM        = "llm"

    _VALID = {"user", "system", "dreamcycle", "llm"}

    @classmethod
    def is_valid(cls, value: str) -> bool:
        return value in cls._VALID


# Well-known reason codes for auditability.
# Open-ended — unknown reasons are stored as-is, just not pre-validated.
KNOWN_REASONS = frozenset({
    "security_alert",
    "catastrophic_precursor",
    "one_time_event",
    "user_explicit",
    "concept_anchor",
    "anomaly_detected",
    "high_importance",
    "policy_default",
})

# Retention classes that are completely immune to pruning
_PRUNE_IMMUNE = frozenset({RetentionClass.PROTECTED, RetentionClass.CRITICAL})

# Default policy dict applied to every new memory
DEFAULT_POLICY: dict = {
    "class":   RetentionClass.VOLATILE,
    "source":  RetentionSource.SYSTEM,
    "reason":  "policy_default",
    "expires": None,
}


# ---------------------------------------------------------------------------
# RetentionPolicy dataclass
# ---------------------------------------------------------------------------

@dataclass
class RetentionPolicy:
    """
    Structured retention descriptor attached to a Memory.

    Attributes
    ----------
    class_ : str
        One of RetentionClass constants.  Governs pruning immunity.
    source : str
        Who assigned this policy (RetentionSource constant).
    reason : str
        Human-readable / machine-parseable reason code for auditability.
    expires : float | None
        Unix timestamp after which a PROTECTED policy downgrades to STANDARD.
        CRITICAL policies ignore this field entirely.
        None means the policy never expires.
    """

    class_:  str            = RetentionClass.VOLATILE
    source:  str            = RetentionSource.SYSTEM
    reason:  str            = "policy_default"
    expires: Optional[float] = None

    # ------------------------------------------------------------------
    # Core predicates
    # ------------------------------------------------------------------

    def is_prune_immune(self) -> bool:
        """Return True if this policy prevents the memory from being pruned."""
        return self.effective_class() in _PRUNE_IMMUNE

    def is_expired(self) -> bool:
        """
        Return True if the policy has passed its expiry timestamp.

        CRITICAL policies never expire regardless of the ``expires`` field.
        """
        if self.class_ == RetentionClass.CRITICAL:
            return False
        if self.expires is None:
            return False
        return time.time() > self.expires

    def effective_class(self) -> str:
        """
        Active retention class after accounting for expiry.

        An expired PROTECTED policy automatically downgrades to STANDARD,
        making the memory eligible for pruning again.
        """
        if self.class_ == RetentionClass.PROTECTED and self.is_expired():
            return RetentionClass.STANDARD
        return self.class_

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "class":   self.class_,
            "source":  self.source,
            "reason":  self.reason,
            "expires": self.expires,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RetentionPolicy":
        if not data:
            return cls()
        return cls(
            class_=data.get("class",   RetentionClass.VOLATILE),
            source=data.get("source",  RetentionSource.SYSTEM),
            reason=data.get("reason",  "policy_default"),
            expires=data.get("expires", None),
        )

    # ------------------------------------------------------------------
    # Convenience factories
    # ------------------------------------------------------------------

    @classmethod
    def make_critical(cls, source: str, reason: str) -> "RetentionPolicy":
        """
        Create a permanent, prune-immune policy.

        Typical use-cases: security_alert, catastrophic_precursor.
        """
        return cls(
            class_=RetentionClass.CRITICAL,
            source=source,
            reason=reason,
            expires=None,
        )

    @classmethod
    def make_protected(
        cls,
        source: str,
        reason: str,
        ttl_seconds: Optional[float] = None,
    ) -> "RetentionPolicy":
        """
        Create a protected (non-prunable) policy with an optional TTL.

        After ``ttl_seconds``, the policy downgrades to STANDARD so the
        memory rejoins the normal pruning lifecycle.  Passing None means
        the protection never expires.
        """
        expires = (time.time() + ttl_seconds) if ttl_seconds is not None else None
        return cls(
            class_=RetentionClass.PROTECTED,
            source=source,
            reason=reason,
            expires=expires,
        )

    @classmethod
    def make_volatile(cls) -> "RetentionPolicy":
        """Return the default volatile policy."""
        return cls()

    # ------------------------------------------------------------------
    # Human-readable summary (useful for audit logs / telemetry)
    # ------------------------------------------------------------------

    def describe(self) -> str:
        eff = self.effective_class()
        suffix = ""
        if self.expires is not None and self.class_ != RetentionClass.CRITICAL:
            import datetime
            dt = datetime.datetime.fromtimestamp(self.expires).isoformat(timespec="seconds")
            suffix = f" (expires {dt})"
        return f"[{eff.upper()}] source={self.source} reason={self.reason}{suffix}"


# ---------------------------------------------------------------------------
# Module-level helpers — used by DreamCycle pruner and store layer
# ---------------------------------------------------------------------------

def is_prune_safe(memory) -> bool:
    """
    Return True if *memory* may be safely pruned.

    Accepts Memory dataclass instances or plain dicts.
    When in doubt (missing policy field) the memory is considered prunable
    to avoid permanent hoarding of unlabelled memories.
    """
    rp_raw = getattr(memory, "retention_policy", None)

    if rp_raw is None:
        return True  # no policy → prunable

    if isinstance(rp_raw, RetentionPolicy):
        policy = rp_raw
    elif isinstance(rp_raw, dict):
        policy = RetentionPolicy.from_dict(rp_raw)
    else:
        return True  # unrecognised type → prunable

    return not policy.is_prune_immune()


def make_policy_dict(
    class_:  str            = RetentionClass.VOLATILE,
    source:  str            = RetentionSource.SYSTEM,
    reason:  str            = "policy_default",
    expires: Optional[float] = None,
) -> dict:
    """
    Return a plain dict suitable for storage in ``Memory.retention_policy``.

    Prefer the RetentionPolicy factories for new code; use this helper when
    you need a raw dict (e.g. constructing a memory inline without importing
    the dataclass).
    """
    return {
        "class":   class_,
        "source":  source,
        "reason":  reason,
        "expires": expires,
    }
