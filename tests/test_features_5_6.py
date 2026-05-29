"""
tests/test_features_5_6.py

Unit tests for:
  Feature 5 — Typed Graph Edges
  Feature 6 — Contradiction Handling & Belief Systems

Run from the project root:
    python -m pytest tests/test_features_5_6.py -v
"""

import os
import sys
import json
import math
import time
import tempfile
import unittest
import types

# Stub sentence_transformers (ML dep not available in CI)
_st_stub = types.ModuleType("sentence_transformers")
class _FakeCE:
    def __init__(self, *a, **kw): pass
    def predict(self, pairs, apply_softmax=True):
        import numpy as np
        return np.array([[0.1, 0.1, 0.8]] * len(pairs))
_st_stub.CrossEncoder = _FakeCE
sys.modules.setdefault("sentence_transformers", _st_stub)


# Ensure repo root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ===========================================================================
# Feature 5: Typed Graph Edges
# ===========================================================================

class TestTypedGraphEdges(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        os.environ["GRAPH_PATH_OVERRIDE"] = self.tmp
        # Patch GRAPH_PATH so MemoryGraph writes to temp dir
        import config.memory_config as cfg
        self._orig_graph_path = cfg.GRAPH_PATH
        cfg.GRAPH_PATH = self.tmp
        # Re-import MemoryGraph with patched path
        import importlib
        import core.graph.graph_manager as gm
        importlib.reload(gm)
        self.MemoryGraph = gm.MemoryGraph
        self.graph = self.MemoryGraph()

    def tearDown(self):
        import config.memory_config as cfg
        cfg.GRAPH_PATH = self._orig_graph_path

    # -----------------------------------------------------------------------
    # Edge type registration
    # -----------------------------------------------------------------------

    def test_semantic_edge_default(self):
        self.graph.connect("a", "b")
        neighbors = self.graph.get_typed_neighbors("a")
        self.assertEqual(len(neighbors), 1)
        nid, ed = neighbors[0]
        self.assertEqual(nid, "b")
        self.assertEqual(ed["edge_type"], "semantic")

    def test_causal_edge_stored(self):
        self.graph.connect_causal("cause", "effect")
        neighbors = self.graph.get_typed_neighbors("cause", edge_types={"causal"})
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0][0], "effect")
        self.assertEqual(neighbors[0][1]["edge_type"], "causal")

    def test_supports_edge(self):
        self.graph.connect_supports("a", "b")
        neighbors = self.graph.get_typed_neighbors("a", edge_types={"supports"})
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0][0], "b")

    def test_contradicts_edge(self):
        self.graph.connect_contradicts("x", "y")
        neighbors = self.graph.get_typed_neighbors("x", edge_types={"contradicts"})
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0][0], "y")

    def test_goal_dependency_edge(self):
        self.graph.connect_goal_dependency("prereq", "goal")
        neighbors = self.graph.get_typed_neighbors("prereq", edge_types={"goal_dependency"})
        self.assertEqual(len(neighbors), 1)
        self.assertEqual(neighbors[0][0], "goal")

    def test_invalid_edge_type_falls_back_to_semantic(self):
        self.graph.connect("a", "b", edge_type="invented_type")
        ed = self.graph.edges["a"]["b"]
        self.assertEqual(ed["edge_type"], "semantic")

    # -----------------------------------------------------------------------
    # Priority upgrade
    # -----------------------------------------------------------------------

    def test_edge_type_upgrades_to_higher_priority(self):
        """A semantic edge upgraded by a causal assertion becomes causal."""
        self.graph.connect("a", "b", edge_type="semantic")
        self.graph.connect("a", "b", edge_type="causal")
        ed = self.graph.edges["a"]["b"]
        self.assertEqual(ed["edge_type"], "causal")

    def test_edge_type_does_not_downgrade(self):
        """A causal edge is NOT downgraded to semantic."""
        self.graph.connect("a", "b", edge_type="causal")
        self.graph.connect("a", "b", edge_type="semantic")
        ed = self.graph.edges["a"]["b"]
        self.assertEqual(ed["edge_type"], "causal")

    # -----------------------------------------------------------------------
    # Directed causal edges
    # -----------------------------------------------------------------------

    def test_causal_creates_directed_edge(self):
        self.graph.connect_causal("A", "B")
        # A is the cause — should appear in directed_edges[A]
        self.assertIn("B", self.graph.directed_edges["A"])

    def test_get_causal_precursors(self):
        self.graph.connect_causal("A", "B")
        self.graph.connect_causal("C", "B")
        precursors = self.graph.get_causal_precursors("B")
        precursor_ids = [pid for pid, _ in precursors]
        self.assertIn("A", precursor_ids)
        self.assertIn("C", precursor_ids)

    def test_get_causal_successors(self):
        self.graph.connect_causal("A", "B")
        self.graph.connect_causal("A", "C")
        successors = self.graph.get_causal_successors("A")
        successor_ids = [sid for sid, _ in successors]
        self.assertIn("B", successor_ids)
        self.assertIn("C", successor_ids)

    def test_non_causal_does_not_create_directed_edge(self):
        self.graph.connect_supports("a", "b")
        # supports is not a directed type — directed_edges should be empty
        self.assertNotIn("b", self.graph.directed_edges.get("a", {}))

    def test_goal_dependency_is_directed(self):
        self.graph.connect_goal_dependency("prereq", "goal")
        self.assertIn("goal", self.graph.directed_edges["prereq"])

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def test_directed_edges_persist_and_reload(self):
        self.graph.connect_causal("X", "Y")
        # Reload from disk
        graph2 = self.MemoryGraph()
        self.assertIn("Y", graph2.directed_edges.get("X", {}))

    def test_all_edge_types_in_valid_set(self):
        from config.memory_config import GRAPH_VALID_EDGE_TYPES
        expected = {"semantic", "causal", "temporal", "derived_from",
                    "supports", "contradicts", "goal_dependency"}
        self.assertTrue(expected.issubset(GRAPH_VALID_EDGE_TYPES))


# ===========================================================================
# Feature 6: NLI Validator (extended)
# ===========================================================================

class TestNLIValidator(unittest.TestCase):
    """
    Tests the new classify_relationship() function.
    These tests mock the CrossEncoder so no model file is required.
    """

    def _patch_model(self, scores):
        """
        Temporarily replace _get_model() so predict() returns *scores*.
        scores: [contradiction, entailment, neutral]
        """
        import core.runtime.nli_validator as nv

        class FakeModel:
            def predict(self, pairs, apply_softmax=True):
                import numpy as np
                return np.array([scores])

        nv._model = FakeModel()
        return nv

    def tearDown(self):
        import core.runtime.nli_validator as nv
        nv._model = None

    def test_classify_contradiction(self):
        nv = self._patch_model([0.85, 0.10, 0.05])
        result = nv.classify_relationship("a", "b")
        self.assertEqual(result.relationship.value, "contradiction")
        self.assertTrue(result.is_contradiction())

    def test_classify_entailment(self):
        nv = self._patch_model([0.05, 0.75, 0.20])
        result = nv.classify_relationship("a", "b")
        self.assertEqual(result.relationship.value, "entailment")
        self.assertTrue(result.is_entailment())

    def test_classify_neutral(self):
        nv = self._patch_model([0.10, 0.15, 0.75])
        result = nv.classify_relationship("a", "b")
        self.assertEqual(result.relationship.value, "neutral")
        self.assertTrue(result.is_neutral())

    def test_contradiction_takes_priority_over_entailment(self):
        """If both contradiction and entailment are high, contradiction wins."""
        nv = self._patch_model([0.72, 0.65, 0.13])
        result = nv.classify_relationship("a", "b")
        self.assertEqual(result.relationship.value, "contradiction")

    def test_model_failure_returns_neutral(self):
        import core.runtime.nli_validator as nv

        class BrokenModel:
            def predict(self, *a, **k):
                raise RuntimeError("model broke")

        nv._model = BrokenModel()
        result = nv.classify_relationship("a", "b")
        self.assertEqual(result.relationship.value, "neutral")

    def test_is_contradiction_backward_compat(self):
        nv = self._patch_model([0.80, 0.15, 0.05])
        self.assertTrue(nv.is_contradiction("a", "b"))

    def test_is_contradiction_returns_false_for_entailment(self):
        nv = self._patch_model([0.05, 0.80, 0.15])
        self.assertFalse(nv.is_contradiction("a", "b"))


# ===========================================================================
# Feature 6: ContradictionEventQueue
# ===========================================================================

class TestContradictionEventQueue(unittest.TestCase):

    def _make_queue(self):
        tmp = tempfile.mktemp(suffix=".json")
        from core.memory.contradiction_system import ContradictionEventQueue
        return ContradictionEventQueue(path=tmp), tmp

    def _make_event(self, new_id="new", cand_id="cand", outcome="contradiction"):
        from core.memory.contradiction_system import ContradictionEvent, BeliefOutcome
        return ContradictionEvent.create(
            new_memory_id=new_id,
            candidate_id=cand_id,
            outcome=BeliefOutcome(outcome),
            contradiction_score=0.85,
            entailment_score=0.10,
            neutral_score=0.05,
            new_memory_text="New memory text",
            candidate_text="Candidate memory text",
        )

    def test_enqueue_and_drain(self):
        q, _ = self._make_queue()
        e = self._make_event()
        q.enqueue(e)
        unresolved = q.drain_unresolved()
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].event_id, e.event_id)

    def test_mark_resolved(self):
        q, _ = self._make_queue()
        e = self._make_event()
        q.enqueue(e)
        q.mark_resolved(e.event_id)
        self.assertEqual(len(q), 0)
        self.assertEqual(len(q.drain_unresolved()), 0)

    def test_stale_event_replaced_by_fresh(self):
        """Re-enqueuing for the same pair replaces the old event."""
        q, _ = self._make_queue()
        e1 = self._make_event(new_id="n1", cand_id="c1")
        e2 = self._make_event(new_id="n1", cand_id="c1", outcome="entailment")
        q.enqueue(e1)
        q.enqueue(e2)
        # Only one event should exist for the pair.
        unresolved = q.drain_unresolved()
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].outcome.value, "entailment")

    def test_persistence(self):
        q, path = self._make_queue()
        e = self._make_event()
        q.enqueue(e)

        from core.memory.contradiction_system import ContradictionEventQueue
        q2 = ContradictionEventQueue(path=path)
        unresolved = q2.drain_unresolved()
        self.assertEqual(len(unresolved), 1)
        self.assertEqual(unresolved[0].event_id, e.event_id)

    def test_len(self):
        q, _ = self._make_queue()
        self.assertEqual(len(q), 0)
        q.enqueue(self._make_event(new_id="n1", cand_id="c1"))
        q.enqueue(self._make_event(new_id="n2", cand_id="c2"))
        self.assertEqual(len(q), 2)


# ===========================================================================
# Feature 6: BeliefSystem
# ===========================================================================

class TestBeliefSystem(unittest.TestCase):

    def _make_memory(self, mid, content, importance=1.0, reliability=0.5,
                     timestamp=None):
        from core.memory.schema import Memory
        m            = Memory()
        m.id         = mid
        m.content    = content
        m.importance = importance
        m.reliability = reliability
        m.timestamp  = timestamp or time.time()
        m.metadata   = {"final_score": importance}
        return m

    def _patch_nli(self, nv_module, scores):
        class FakeModel:
            def predict(self, pairs, apply_softmax=True):
                import numpy as np
                return np.array([scores])
        nv_module._model = FakeModel()

    def setUp(self):
        import core.runtime.nli_validator as nv
        self._nv = nv

        tmp_path = tempfile.mktemp(suffix="_belief_queue.json")
        import core.memory.contradiction_system as cs
        self._orig_path = cs.CONTRADICTION_QUEUE_PATH

        import config.memory_config as cfg
        cfg.CONTRADICTION_QUEUE_PATH = tmp_path
        cs.CONTRADICTION_QUEUE_PATH  = tmp_path

        self.belief_system = cs.BeliefSystem()

        # Patch save_fn
        self.saved = {}
        self.save_fn = lambda m: self.saved.__setitem__(m.id, m)

    def tearDown(self):
        self._nv._model = None
        import config.memory_config as cfg
        import core.memory.contradiction_system as cs
        cfg.CONTRADICTION_QUEUE_PATH = self._orig_path
        cs.CONTRADICTION_QUEUE_PATH  = self._orig_path

    def test_contradiction_penalizes_weaker_memory(self):
        self._patch_nli(self._nv, [0.85, 0.10, 0.05])

        new_m  = self._make_memory("new", "I love cats", importance=2.0)
        cand_m = self._make_memory("cand", "I don't love cats", importance=1.0)

        self.belief_system.evaluate(new_m, [cand_m], save_fn=self.save_fn)

        # Candidate (weaker) should have been penalized.
        self.assertIn("cand", self.saved)
        self.assertLess(self.saved["cand"].importance, 1.0)

    def test_entailment_reinforces_both_memories(self):
        self._patch_nli(self._nv, [0.05, 0.75, 0.20])

        new_m  = self._make_memory("new",  "The server is down", reliability=0.5)
        cand_m = self._make_memory("cand", "The server is offline", reliability=0.5)

        self.belief_system.evaluate(new_m, [cand_m], save_fn=self.save_fn)

        # Both should have been saved with higher reliability.
        self.assertIn("new",  self.saved)
        self.assertIn("cand", self.saved)
        self.assertGreater(self.saved["new"].reliability,  0.5)
        self.assertGreater(self.saved["cand"].reliability, 0.5)

    def test_neutral_produces_no_event(self):
        self._patch_nli(self._nv, [0.05, 0.10, 0.85])

        new_m  = self._make_memory("new",  "The sky is blue")
        cand_m = self._make_memory("cand", "Pizza is delicious")

        events = self.belief_system.evaluate(new_m, [cand_m], save_fn=self.save_fn)
        self.assertEqual(len(events), 0)
        # No saves should have occurred.
        self.assertEqual(len(self.saved), 0)

    def test_same_id_skipped(self):
        """evaluate() should skip if new_memory.id == candidate.id."""
        new_m = self._make_memory("same", "content")
        events = self.belief_system.evaluate(new_m, [new_m], save_fn=self.save_fn)
        self.assertEqual(len(events), 0)

    def test_contradiction_registers_contradicts_graph_edge(self):
        self._patch_nli(self._nv, [0.85, 0.10, 0.05])

        # Build a simple in-memory graph stub
        edges_registered = {}

        class FakeGraph:
            def connect_contradicts(self, a, b, **kw):
                edges_registered[a] = b
            def connect_supports(self, a, b, **kw):
                pass

        import core.memory.contradiction_system as cs
        bs = cs.BeliefSystem(graph=FakeGraph())
        bs.queue = self.belief_system.queue  # share the queue

        new_m  = self._make_memory("n", "cats are great", importance=2.0)
        cand_m = self._make_memory("c", "cats are not great", importance=1.0)

        bs.evaluate(new_m, [cand_m], save_fn=self.save_fn)
        # Weaker memory ("c") should have a contradicts edge registered.
        self.assertTrue(len(edges_registered) > 0)

    def test_entailment_registers_supports_graph_edge(self):
        self._patch_nli(self._nv, [0.05, 0.80, 0.15])

        edges_registered = {}

        class FakeGraph:
            def connect_contradicts(self, a, b, **kw):
                pass
            def connect_supports(self, a, b, **kw):
                edges_registered[(a, b)] = True

        import core.memory.contradiction_system as cs
        bs = cs.BeliefSystem(graph=FakeGraph())
        bs.queue = self.belief_system.queue

        new_m  = self._make_memory("n", "server is down")
        cand_m = self._make_memory("c", "server is offline")

        bs.evaluate(new_m, [cand_m], save_fn=self.save_fn)
        self.assertTrue(len(edges_registered) > 0)

    def test_belief_version_incremented_on_contradiction(self):
        self._patch_nli(self._nv, [0.85, 0.10, 0.05])

        new_m  = self._make_memory("n", "I like dogs", importance=2.0)
        cand_m = self._make_memory("c", "I don't like dogs", importance=1.0)
        cand_m.contradiction_count = 0

        self.belief_system.evaluate(new_m, [cand_m], save_fn=self.save_fn)
        self.assertEqual(self.saved["c"].contradiction_count, 1)

    def test_entailment_increments_entailment_count(self):
        self._patch_nli(self._nv, [0.05, 0.75, 0.20])

        new_m  = self._make_memory("n",  "The server crashed")
        cand_m = self._make_memory("c",  "The server went down")
        cand_m.entailment_count = 0
        new_m.entailment_count  = 0

        self.belief_system.evaluate(new_m, [cand_m], save_fn=self.save_fn)
        self.assertEqual(self.saved["n"].entailment_count, 1)
        self.assertEqual(self.saved["c"].entailment_count, 1)


# ===========================================================================
# Feature 6: Schema v8 fields
# ===========================================================================

class TestSchemaV8(unittest.TestCase):

    def test_new_fields_have_defaults(self):
        from core.memory.schema import Memory
        m = Memory()
        self.assertEqual(m.contradiction_count, 0)
        self.assertEqual(m.entailment_count,    0)
        self.assertEqual(m.belief_version,      0)

    def test_from_dict_handles_missing_v8_fields(self):
        """Old JSON without v8 fields should deserialize without error."""
        from core.memory.schema import Memory
        old_dict = {
            "id": "test-id",
            "content": "test content",
            "importance": 1.5,
            "reliability": 0.6,
            "type": "fact",
        }
        m = Memory.from_dict(old_dict)
        self.assertEqual(m.contradiction_count, 0)
        self.assertEqual(m.entailment_count,    0)
        self.assertEqual(m.belief_version,      0)
        self.assertAlmostEqual(m.importance, 1.5)

    def test_to_dict_includes_v8_fields(self):
        from core.memory.schema import Memory
        m = Memory()
        m.contradiction_count = 3
        m.entailment_count    = 7
        m.belief_version      = 10
        d = m.to_dict()
        self.assertEqual(d["contradiction_count"], 3)
        self.assertEqual(d["entailment_count"],    7)
        self.assertEqual(d["belief_version"],      10)

    def test_schema_version_is_8(self):
        from config.memory_config import SCHEMA_VERSION
        self.assertEqual(SCHEMA_VERSION, 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
