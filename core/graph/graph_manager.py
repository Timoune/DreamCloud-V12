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
 
 
class MemoryGraph:
 
    def __init__(self):
        self.edges = defaultdict(dict)
        self._load()
 
    def connect(self, a, b, base_weight=1.0, similarity=0.0):
 
        if a == b:
            return
 
        now = time.time()
 
        def update(src, dst):
 
            edge = self.edges[src].get(dst)
 
            if edge:
 
                age   = now - edge["last_updated"]
                decay = math.exp(-age / DECAY_CONSTANT)   # decays faster
 
                edge["weight"] = (
                    edge["weight"] * decay
                    + base_weight
                    + similarity
                )
 
                edge["count"]       += 1
                edge["last_updated"] = now
 
            else:
 
                edge = {
                    "weight":       base_weight + similarity,
                    "count":        1,
                    "last_updated": now,
                }
 
            self.edges[src][dst] = edge
 
        update(a, b)
        update(b, a)
 
        self._save()
 
    def get_neighbors(self, memory_id, top_k=3):
 
        neighbors = self.edges.get(memory_id, {})
 
        ranked = sorted(
            neighbors.items(),
            key=lambda x: x[1]["weight"],
            reverse=True,
        )
 
        return ranked[:top_k]
 
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
 
            # Stop early if we've already collected enough nodes.
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
 
            # Nothing new to explore.
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