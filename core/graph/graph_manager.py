import os
import json
import time
import math

from collections import defaultdict

from config.memory_config import GRAPH_PATH

GRAPH_FILE = os.path.join(GRAPH_PATH, "graph.json")

# Halved from 86400 (1 day) → more aggressive decay.
# Edges that aren't reinforced within ~12 h lose roughly 63 % of their weight.
DECAY_CONSTANT = 43200

MAX_NEIGHBORS = 20

# Valid edge types for Retrospective Importance Revaluation (v15).
# Legacy edges loaded from disk without an edge_type are treated as "semantic".
VALID_EDGE_TYPES = frozenset({"causal", "temporal", "derived_from", "semantic"})


class MemoryGraph:

    def __init__(self):
        self.edges = defaultdict(dict)
        self._load()

    def connect(self, a, b, base_weight=1.0, similarity=0.0, edge_type="semantic"):
        """
        Create or reinforce an undirected edge between memories *a* and *b*.

        Parameters
        ----------
        a, b        : memory IDs (str)
        base_weight : starting/additive weight for new edges
        similarity  : cosine similarity bonus added to weight on creation
        edge_type   : one of "causal", "temporal", "derived_from", "semantic"
                      Controls how much importance flows through this edge
                      during backpropagation.  Defaults to "semantic" for
                      backwards-compatible calls that omit the argument.
        """
        if a == b:
            return

        if edge_type not in VALID_EDGE_TYPES:
            edge_type = "semantic"

        now = time.time()

        def update(src, dst):
            edge = self.edges[src].get(dst)

            if edge:
                age   = now - edge["last_updated"]
                decay = math.exp(-age / DECAY_CONSTANT)

                edge["weight"] = (
                    edge["weight"] * decay
                    + base_weight
                    + similarity
                )
                edge["count"]        += 1
                edge["last_updated"]  = now

                # Upgrade edge type if a stronger relationship is asserted.
                # Priority ladder: causal(3) > derived_from(2) > temporal(1) > semantic(0)
                _priority = {"causal": 3, "derived_from": 2, "temporal": 1, "semantic": 0}
                existing = edge.get("edge_type", "semantic")
                if _priority.get(edge_type, 0) > _priority.get(existing, 0):
                    edge["edge_type"] = edge_type

            else:
                edge = {
                    "weight":       base_weight + similarity,
                    "count":        1,
                    "last_updated": now,
                    "edge_type":    edge_type,
                }

            self.edges[src][dst] = edge

        update(a, b)
        update(b, a)

        self._save()

    # ------------------------------------------------------------------
    # Convenience factories for typed edges (v15)
    # ------------------------------------------------------------------

    def connect_causal(self, cause_id: str, effect_id: str, base_weight=2.0):
        """
        Record a causal relationship: *cause_id* → *effect_id*.

        Causal edges carry the highest backpropagation weight (1.0) so that
        when an effect is revealed to be significant, its causal precursors
        receive the strongest retroactive importance boost.
        """
        self.connect(cause_id, effect_id, base_weight=base_weight, edge_type="causal")

    def connect_temporal(self, earlier_id: str, later_id: str, base_weight=1.5):
        """
        Record a temporal ordering: *earlier_id* happened before *later_id*.

        Temporal edges carry medium backpropagation weight (0.70).  They
        capture precursor relationships even when causality is uncertain.
        """
        self.connect(earlier_id, later_id, base_weight=base_weight, edge_type="temporal")

    def connect_derived(self, source_id: str, derived_id: str, base_weight=1.8):
        """
        Record a derivation: *derived_id* was inferred or derived from *source_id*.

        Derived-from edges carry high backpropagation weight (0.85) because
        the source memory directly contributed to the derived memory's content.
        """
        self.connect(source_id, derived_id, base_weight=base_weight, edge_type="derived_from")

    # ------------------------------------------------------------------
    # Retrieval helpers
    # ------------------------------------------------------------------

    def get_neighbors(self, memory_id, top_k=3):

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
        Return neighbors filtered by *edge_types*, ranked by edge weight.

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
        BFS graph expansion starting from *seed_ids*.

        Parameters
        ----------
        seed_ids  : iterable of memory IDs used as BFS roots.
        depth     : maximum number of hops from any seed.
        top_k     : neighbors examined per node at each hop.
        max_nodes : hard cap on the total number of nodes returned.
                    Once visited reaches this size the traversal stops
                    immediately, keeping retrieval time bounded even as
                    the graph grows to thousands of nodes.

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

    def _save(self):

        os.makedirs(GRAPH_PATH, exist_ok=True)

        with open(GRAPH_FILE, "w", encoding="utf-8") as f:
            json.dump(dict(self.edges), f, indent=2)

    def _load(self):

        if not os.path.exists(GRAPH_FILE):
            return

        with open(GRAPH_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.edges = defaultdict(dict, data)
