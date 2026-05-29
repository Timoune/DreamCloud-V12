"""
graph_manager.py — DreamCloud Feature 5: Typed Graph Edges

Changes from v1 (plain semantic graph)
---------------------------------------
v2  — VALID_EDGE_TYPES with priority ladder; connect_causal/temporal/derived
v3  — Feature 5: full edge-type set + directed causal tracking

Typed Edge Set (v3)
-------------------
  semantic       — default; pure embedding similarity
  causal         — A caused B  (directed: A → B)
  temporal       — A happened before B  (directed: A → B)
  derived_from   — B was derived/inferred from A
  supports       — A corroborates / reinforces B
  contradicts    — A conflicts with B  (undirected contradiction link)
  goal_dependency — B depends on A to be achieved  (directed: A → B)

Directional tracking
--------------------
For edge types in GRAPH_DIRECTED_TYPES (causal, goal_dependency, temporal)
a SEPARATE directed adjacency list is maintained alongside the undirected
`edges` structure:

  directed_edges[source_id][target_id] = edge_data_snapshot

This allows callers to ask:
  "What caused this memory?"  → graph.get_causal_precursors(effect_id)
  "What does this cause?"     → graph.get_causal_successors(cause_id)

Backpropagation rationale
-------------------------
When B becomes high-importance:
  - Traverse directed causal precursors (A → B means boost A).
  - Do NOT propagate forward to C just because A also caused C.

The undirected `edges` dict is still used for general BFS traversal;
directed_edges is an overlay for precise causal backprop.
"""

import os
import json
import time
import math

from collections import defaultdict

from config.memory_config import (
    GRAPH_PATH,
    GRAPH_VALID_EDGE_TYPES,
    GRAPH_DIRECTED_TYPES,
    BACKPROP_EDGE_WEIGHTS,
)

GRAPH_FILE         = os.path.join(GRAPH_PATH, "graph.json")
DIRECTED_GRAPH_FILE = os.path.join(GRAPH_PATH, "directed_graph.json")

# Halved from 86400 (1 day) → more aggressive decay.
DECAY_CONSTANT = 43200

MAX_NEIGHBORS = 20

# Edge priority ladder for upgrade logic: higher number = stronger relationship.
_EDGE_PRIORITY = {
    "causal":          6,
    "goal_dependency": 5,
    "derived_from":    4,
    "supports":        3,
    "temporal":        2,
    "contradicts":     1,
    "semantic":        0,
}


class MemoryGraph:

    def __init__(self):
        self.edges          = defaultdict(dict)   # undirected: id → {id: edge_data}
        self.directed_edges = defaultdict(dict)   # directed:   source → {target: snap}
        self._load()

    # ------------------------------------------------------------------
    # Core connection API
    # ------------------------------------------------------------------

    def connect(
        self,
        a,
        b,
        base_weight=1.0,
        similarity=0.0,
        edge_type="semantic",
    ):
        """
        Create or reinforce an undirected edge between memories *a* and *b*.

        Parameters
        ----------
        a, b        : memory IDs (str)
        base_weight : starting / additive weight for new edges
        similarity  : cosine similarity bonus added to weight on creation
        edge_type   : one of GRAPH_VALID_EDGE_TYPES; defaults to "semantic"
                      Controls how strongly importance propagates through this
                      edge during backpropagation.

        Notes
        -----
        For edge_types in GRAPH_DIRECTED_TYPES (causal, goal_dependency,
        temporal) a directed snapshot is ALSO written to self.directed_edges
        treating *a* as the source and *b* as the target.
        Use connect_causal(cause, effect) to make the directionality explicit.
        """
        if a == b:
            return

        if edge_type not in GRAPH_VALID_EDGE_TYPES:
            edge_type = "semantic"

        now = time.time()

        def _update(src, dst):
            edge = self.edges[src].get(dst)

            if edge:
                age   = now - edge["last_updated"]
                decay = math.exp(-age / DECAY_CONSTANT)

                edge["weight"] = (
                    edge["weight"] * decay
                    + base_weight
                    + similarity
                )
                edge["count"]       += 1
                edge["last_updated"] = now

                # Upgrade edge type only to a stronger relationship.
                existing = edge.get("edge_type", "semantic")
                if _EDGE_PRIORITY.get(edge_type, 0) > _EDGE_PRIORITY.get(existing, 0):
                    edge["edge_type"] = edge_type
            else:
                edge = {
                    "weight":       base_weight + similarity,
                    "count":        1,
                    "last_updated": now,
                    "edge_type":    edge_type,
                }

            self.edges[src][dst] = edge

        _update(a, b)
        _update(b, a)

        # Record directed snapshot for directed edge types.
        if edge_type in GRAPH_DIRECTED_TYPES:
            self._record_directed(a, b, edge_type, now)

        self._save()

    # ------------------------------------------------------------------
    # Typed edge factory methods
    # ------------------------------------------------------------------

    def connect_causal(self, cause_id: str, effect_id: str, base_weight=2.0):
        """
        Record *cause_id* → *effect_id*.

        Causal edges carry the highest backprop weight (1.0).
        When the effect becomes important, its causes receive the strongest
        retroactive importance boost.
        """
        self.connect(cause_id, effect_id, base_weight=base_weight, edge_type="causal")

    def connect_temporal(self, earlier_id: str, later_id: str, base_weight=1.5):
        """
        Record a temporal ordering: *earlier_id* happened before *later_id*.

        Temporal edges carry medium backprop weight (0.70).
        """
        self.connect(earlier_id, later_id, base_weight=base_weight, edge_type="temporal")

    def connect_derived(self, source_id: str, derived_id: str, base_weight=1.8):
        """
        Record that *derived_id* was inferred / derived from *source_id*.

        Derived-from edges carry high backprop weight (0.85).
        """
        self.connect(source_id, derived_id, base_weight=base_weight, edge_type="derived_from")

    def connect_supports(self, a_id: str, b_id: str, base_weight=1.3):
        """
        Record that memory *a_id* supports / corroborates memory *b_id*.

        Supports edges are undirected (mutual reinforcement).
        Backprop weight: 0.65.

        Example
        -------
        connect_supports(sensor_log_id, fault_report_id)
        """
        self.connect(a_id, b_id, base_weight=base_weight, edge_type="supports")

    def connect_contradicts(self, a_id: str, b_id: str, base_weight=1.0):
        """
        Record that memory *a_id* contradicts memory *b_id*.

        Contradicts edges are undirected but carry minimal backprop weight
        (0.20) so that importance does not freely propagate across conflicting
        beliefs.  Used by BeliefSystem after NLI confirms a contradiction.

        Example
        -------
        connect_contradicts(old_fact_id, new_fact_id)
        """
        self.connect(a_id, b_id, base_weight=base_weight, edge_type="contradicts")

    def connect_goal_dependency(
        self,
        precondition_id: str,
        goal_id: str,
        base_weight=1.8,
    ):
        """
        Record that *goal_id* depends on *precondition_id* being achieved.
        Directed: precondition_id → goal_id.

        Goal-dependency edges carry high backprop weight (0.90): when a goal
        becomes urgent, its preconditions inherit that urgency.

        Example
        -------
        connect_goal_dependency(auth_token_id, api_access_goal_id)
        """
        self.connect(
            precondition_id,
            goal_id,
            base_weight=base_weight,
            edge_type="goal_dependency",
        )

    # ------------------------------------------------------------------
    # Directed causal query API
    # ------------------------------------------------------------------

    def get_causal_precursors(
        self,
        effect_id: str,
        top_k: int = MAX_NEIGHBORS,
    ) -> list:
        """
        Return memories that CAUSED *effect_id*, ranked by edge weight.

        Traverses the directed_edges index in reverse: finds all source nodes
        whose directed edge points to *effect_id* with a directed type.

        Returns
        -------
        list of (cause_id, edge_data) tuples, highest weight first.
        """
        results = []
        for source_id, targets in self.directed_edges.items():
            edge_data = targets.get(effect_id)
            if edge_data is not None:
                results.append((source_id, edge_data))

        results.sort(key=lambda x: x[1].get("weight", 0.0), reverse=True)
        return results[:top_k]

    def get_causal_successors(
        self,
        cause_id: str,
        top_k: int = MAX_NEIGHBORS,
    ) -> list:
        """
        Return memories that *cause_id* CAUSED or PRECEDES, ranked by weight.

        Returns
        -------
        list of (effect_id, edge_data) tuples, highest weight first.
        """
        targets = self.directed_edges.get(cause_id, {})
        results = sorted(
            targets.items(),
            key=lambda x: x[1].get("weight", 0.0),
            reverse=True,
        )
        return results[:top_k]

    def get_directed_neighbors(
        self,
        memory_id: str,
        direction: str = "successors",
        edge_types=None,
        top_k: int = MAX_NEIGHBORS,
    ) -> list:
        """
        Return directed neighbors of *memory_id*.

        Parameters
        ----------
        memory_id  : source node
        direction  : "successors" (outgoing) or "precursors" (incoming)
        edge_types : filter by edge type(s), or None for all directed types
        top_k      : max results

        Returns
        -------
        list of (neighbor_id, edge_data) tuples, highest weight first.
        """
        if direction == "successors":
            neighbors = self.directed_edges.get(memory_id, {})
            results = [
                (nid, ed) for nid, ed in neighbors.items()
                if edge_types is None or ed.get("edge_type") in edge_types
            ]
        else:  # precursors
            results = [
                (src, ed)
                for src, targets in self.directed_edges.items()
                if (ed := targets.get(memory_id)) is not None
                and (edge_types is None or ed.get("edge_type") in edge_types)
            ]

        results.sort(key=lambda x: x[1].get("weight", 0.0), reverse=True)
        return results[:top_k]

    # ------------------------------------------------------------------
    # General retrieval helpers
    # ------------------------------------------------------------------

    def get_neighbors(self, memory_id, top_k=3):
        """Return top-k undirected neighbors by edge weight."""
        neighbors = self.edges.get(memory_id, {})
        ranked = sorted(
            neighbors.items(),
            key=lambda x: x[1]["weight"],
            reverse=True,
        )
        return ranked[:top_k]

    def get_typed_neighbors(
        self,
        memory_id: str,
        edge_types=None,
        top_k: int = MAX_NEIGHBORS,
    ) -> list:
        """
        Return undirected neighbors filtered by *edge_types*, ranked by weight.

        Parameters
        ----------
        memory_id  : source memory ID
        edge_types : iterable of edge type strings to include, or None for all
        top_k      : maximum number of neighbors to return

        Returns
        -------
        list of (neighbor_id, edge_data_dict) tuples, highest weight first.
        Legacy edges without an "edge_type" key are treated as "semantic".
        """
        neighbors = self.edges.get(memory_id, {})

        result = []
        for nid, edge_data in neighbors.items():
            etype = edge_data.get("edge_type", "semantic")
            if edge_types is None or etype in edge_types:
                result.append((nid, {**edge_data, "edge_type": etype}))

        result.sort(key=lambda x: x[1]["weight"], reverse=True)
        return result[:top_k]

    def expand(self, seed_ids, depth=2, top_k=2, max_nodes=50):
        """
        BFS graph expansion from *seed_ids*.

        Parameters
        ----------
        seed_ids  : iterable of memory IDs used as BFS roots.
        depth     : maximum number of hops from any seed.
        top_k     : neighbors examined per node at each hop.
        max_nodes : hard cap on total visited nodes.

        Returns
        -------
        list of all visited memory IDs (seeds + expanded nodes).
        """
        visited  = set(seed_ids)
        frontier = set(seed_ids)

        for _ in range(depth):

            if len(visited) >= max_nodes:
                break

            new_frontier = set()

            for mid in frontier:

                if len(visited) >= max_nodes:
                    break

                for nid, _ in self.get_neighbors(mid, top_k):

                    if nid not in visited:
                        visited.add(nid)
                        new_frontier.add(nid)

                        if len(visited) >= max_nodes:
                            break

            frontier = new_frontier

            if not frontier:
                break

        return list(visited)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _record_directed(
        self,
        source_id: str,
        target_id: str,
        edge_type: str,
        now: float,
    ) -> None:
        """
        Write / update a directed edge snapshot in self.directed_edges.

        Called automatically by connect() for GRAPH_DIRECTED_TYPES.
        """
        existing = self.directed_edges[source_id].get(target_id)
        if existing:
            existing["weight"]       = self.edges[source_id].get(target_id, {}).get("weight", 1.0)
            existing["last_updated"] = now
            # Upgrade edge_type if stronger.
            if (
                _EDGE_PRIORITY.get(edge_type, 0)
                > _EDGE_PRIORITY.get(existing.get("edge_type", "semantic"), 0)
            ):
                existing["edge_type"] = edge_type
        else:
            self.directed_edges[source_id][target_id] = {
                "edge_type":    edge_type,
                "weight":       self.edges[source_id].get(target_id, {}).get("weight", 1.0),
                "last_updated": now,
            }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _save(self):
        os.makedirs(GRAPH_PATH, exist_ok=True)

        with open(GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(self.edges), f, indent=2)

        with open(DIRECTED_GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(self.directed_edges), f, indent=2)

    def _load(self):
        if os.path.exists(GRAPH_FILE):
            with open(GRAPH_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.edges = defaultdict(dict, data)

        if os.path.exists(DIRECTED_GRAPH_FILE):
            with open(DIRECTED_GRAPH_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.directed_edges = defaultdict(dict, data)
