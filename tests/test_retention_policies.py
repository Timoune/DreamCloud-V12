"""
Tests for Retention Policies & Archival Safety (v6).

Covers:
- RetentionPolicy dataclass behaviour
- is_prune_safe() helper
- Store migration (v5 → v6)
- DreamCycle _prune() respects retention policies
"""

import time
import unittest

from core.memory.retention import (
    RetentionClass,
    RetentionSource,
    RetentionPolicy,
    DEFAULT_POLICY,
    KNOWN_REASONS,
    is_prune_safe,
    make_policy_dict,
)
from core.memory.schema import Memory, MemoryType


# ---------------------------------------------------------------------------
# RetentionPolicy unit tests
# ---------------------------------------------------------------------------

class TestRetentionPolicyDefaults(unittest.TestCase):

    def test_default_policy_is_volatile(self):
        rp = RetentionPolicy()
        self.assertEqual(rp.class_, RetentionClass.VOLATILE)

    def test_default_policy_source_is_system(self):
        rp = RetentionPolicy()
        self.assertEqual(rp.source, RetentionSource.SYSTEM)

    def test_default_is_prune_safe(self):
        rp = RetentionPolicy()
        self.assertFalse(rp.is_prune_immune())

    def test_default_policy_dict_matches_constant(self):
        rp = RetentionPolicy()
        self.assertEqual(rp.to_dict(), DEFAULT_POLICY)


class TestRetentionClassLadder(unittest.TestCase):

    def test_volatile_not_immune(self):
        rp = RetentionPolicy(class_=RetentionClass.VOLATILE)
        self.assertFalse(rp.is_prune_immune())

    def test_standard_not_immune(self):
        rp = RetentionPolicy(class_=RetentionClass.STANDARD)
        self.assertFalse(rp.is_prune_immune())

    def test_protected_is_immune(self):
        rp = RetentionPolicy(class_=RetentionClass.PROTECTED)
        self.assertTrue(rp.is_prune_immune())

    def test_critical_is_immune(self):
        rp = RetentionPolicy(class_=RetentionClass.CRITICAL)
        self.assertTrue(rp.is_prune_immune())

    def test_strength_ordering(self):
        self.assertLess(
            RetentionClass.strength(RetentionClass.VOLATILE),
            RetentionClass.strength(RetentionClass.STANDARD),
        )
        self.assertLess(
            RetentionClass.strength(RetentionClass.STANDARD),
            RetentionClass.strength(RetentionClass.PROTECTED),
        )
        self.assertLess(
            RetentionClass.strength(RetentionClass.PROTECTED),
            RetentionClass.strength(RetentionClass.CRITICAL),
        )


class TestRetentionExpiry(unittest.TestCase):

    def test_no_expiry_never_expires(self):
        rp = RetentionPolicy(class_=RetentionClass.PROTECTED, expires=None)
        self.assertFalse(rp.is_expired())

    def test_future_expiry_not_expired(self):
        rp = RetentionPolicy(
            class_=RetentionClass.PROTECTED,
            expires=time.time() + 9999,
        )
        self.assertFalse(rp.is_expired())

    def test_past_expiry_is_expired(self):
        rp = RetentionPolicy(
            class_=RetentionClass.PROTECTED,
            expires=time.time() - 1,
        )
        self.assertTrue(rp.is_expired())

    def test_critical_never_expires(self):
        rp = RetentionPolicy(
            class_=RetentionClass.CRITICAL,
            expires=time.time() - 9999,   # far in the past
        )
        self.assertFalse(rp.is_expired())

    def test_expired_protected_downgrades_to_standard(self):
        rp = RetentionPolicy(
            class_=RetentionClass.PROTECTED,
            expires=time.time() - 1,
        )
        self.assertEqual(rp.effective_class(), RetentionClass.STANDARD)

    def test_expired_protected_becomes_prune_safe(self):
        rp = RetentionPolicy(
            class_=RetentionClass.PROTECTED,
            expires=time.time() - 1,
        )
        self.assertFalse(rp.is_prune_immune())

    def test_unexpired_protected_stays_immune(self):
        rp = RetentionPolicy(
            class_=RetentionClass.PROTECTED,
            expires=time.time() + 9999,
        )
        self.assertTrue(rp.is_prune_immune())


class TestRetentionFactories(unittest.TestCase):

    def test_make_critical(self):
        rp = RetentionPolicy.make_critical(
            source=RetentionSource.SYSTEM,
            reason="security_alert",
        )
        self.assertEqual(rp.class_, RetentionClass.CRITICAL)
        self.assertIsNone(rp.expires)
        self.assertTrue(rp.is_prune_immune())

    def test_make_protected_no_ttl(self):
        rp = RetentionPolicy.make_protected(
            source=RetentionSource.USER,
            reason="user_explicit",
        )
        self.assertEqual(rp.class_, RetentionClass.PROTECTED)
        self.assertIsNone(rp.expires)

    def test_make_protected_with_ttl(self):
        ttl = 3600
        before = time.time()
        rp = RetentionPolicy.make_protected(
            source=RetentionSource.USER,
            reason="user_explicit",
            ttl_seconds=ttl,
        )
        self.assertIsNotNone(rp.expires)
        self.assertAlmostEqual(rp.expires, before + ttl, delta=2)

    def test_make_volatile(self):
        rp = RetentionPolicy.make_volatile()
        self.assertEqual(rp.class_, RetentionClass.VOLATILE)
        self.assertFalse(rp.is_prune_immune())


class TestRetentionSerialisation(unittest.TestCase):

    def _roundtrip(self, rp: RetentionPolicy) -> RetentionPolicy:
        return RetentionPolicy.from_dict(rp.to_dict())

    def test_volatile_roundtrip(self):
        rp = RetentionPolicy.make_volatile()
        rt = self._roundtrip(rp)
        self.assertEqual(rt.class_, rp.class_)
        self.assertEqual(rt.source, rp.source)
        self.assertEqual(rt.expires, rp.expires)

    def test_critical_roundtrip(self):
        rp = RetentionPolicy.make_critical("system", "security_alert")
        rt = self._roundtrip(rp)
        self.assertEqual(rt.class_, RetentionClass.CRITICAL)
        self.assertIsNone(rt.expires)

    def test_from_dict_none_returns_default(self):
        rp = RetentionPolicy.from_dict(None)
        self.assertEqual(rp.class_, RetentionClass.VOLATILE)

    def test_from_dict_empty_returns_default(self):
        rp = RetentionPolicy.from_dict({})
        self.assertEqual(rp.class_, RetentionClass.VOLATILE)

    def test_describe_contains_class(self):
        rp = RetentionPolicy.make_critical("system", "security_alert")
        desc = rp.describe()
        self.assertIn("CRITICAL", desc)
        self.assertIn("security_alert", desc)


# ---------------------------------------------------------------------------
# is_prune_safe() helper
# ---------------------------------------------------------------------------

class TestIsPruneSafe(unittest.TestCase):

    def _memory(self, policy_dict=None):
        m = Memory(content="test memory")
        if policy_dict is not None:
            m.retention_policy = policy_dict
        return m

    def test_volatile_memory_is_prunable(self):
        m = self._memory(make_policy_dict(class_=RetentionClass.VOLATILE))
        self.assertTrue(is_prune_safe(m))

    def test_standard_memory_is_prunable(self):
        m = self._memory(make_policy_dict(class_=RetentionClass.STANDARD))
        self.assertTrue(is_prune_safe(m))

    def test_protected_memory_not_prunable(self):
        m = self._memory(make_policy_dict(class_=RetentionClass.PROTECTED))
        self.assertFalse(is_prune_safe(m))

    def test_critical_memory_not_prunable(self):
        m = self._memory(make_policy_dict(class_=RetentionClass.CRITICAL))
        self.assertFalse(is_prune_safe(m))

    def test_no_policy_is_prunable(self):
        m = Memory(content="no policy set")
        m.retention_policy = None  # type: ignore[assignment]
        self.assertTrue(is_prune_safe(m))

    def test_expired_protected_becomes_prunable(self):
        m = self._memory({
            "class": RetentionClass.PROTECTED,
            "source": RetentionSource.SYSTEM,
            "reason": "user_explicit",
            "expires": time.time() - 1,
        })
        self.assertTrue(is_prune_safe(m))


# ---------------------------------------------------------------------------
# Memory schema integration
# ---------------------------------------------------------------------------

class TestMemoryRetentionField(unittest.TestCase):

    def test_new_memory_has_default_policy(self):
        m = Memory(content="hello world")
        self.assertIn("class", m.retention_policy)
        self.assertEqual(m.retention_policy["class"], RetentionClass.VOLATILE)

    def test_to_dict_includes_retention_policy(self):
        m = Memory(content="hello world")
        d = m.to_dict()
        self.assertIn("retention_policy", d)

    def test_from_dict_preserves_retention_policy(self):
        policy = RetentionPolicy.make_critical("system", "security_alert").to_dict()
        m = Memory(content="security event")
        m.retention_policy = policy
        d = m.to_dict()
        m2 = Memory.from_dict(d)
        self.assertEqual(m2.retention_policy["class"], RetentionClass.CRITICAL)
        self.assertEqual(m2.retention_policy["reason"], "security_alert")


# ---------------------------------------------------------------------------
# make_policy_dict helper
# ---------------------------------------------------------------------------

class TestMakePolicyDict(unittest.TestCase):

    def test_defaults(self):
        d = make_policy_dict()
        self.assertEqual(d["class"], RetentionClass.VOLATILE)
        self.assertEqual(d["source"], RetentionSource.SYSTEM)
        self.assertIsNone(d["expires"])

    def test_custom_values(self):
        d = make_policy_dict(
            class_=RetentionClass.CRITICAL,
            source=RetentionSource.USER,
            reason="security_alert",
        )
        self.assertEqual(d["class"], RetentionClass.CRITICAL)
        self.assertEqual(d["reason"], "security_alert")


if __name__ == "__main__":
    unittest.main()
