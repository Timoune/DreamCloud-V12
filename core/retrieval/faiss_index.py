import os
import json
import faiss
import numpy as np
import threading
 
from config.memory_config import (
    EMBEDDING_DIM,
    FAISS_INDEX_FILE,
    FAISS_META_FILE,
    FAISS_PATH
)
 
os.makedirs(FAISS_PATH, exist_ok=True)
 
 
class FaissIndex:
    def __init__(self):
        self._lock = threading.RLock()
        self.id_map: list = []           
        self.id_to_index: dict = {}      
 
        with self._lock:
            if os.path.exists(FAISS_INDEX_FILE):
                self.index = faiss.read_index(FAISS_INDEX_FILE)
                self._load_meta()
            else:
                self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
 
    def _normalize(self, vec) -> np.ndarray:
        arr = np.array([vec], dtype="float32")
        faiss.normalize_L2(arr)
        return arr
 
    def _persist(self):
        faiss.write_index(self.index, FAISS_INDEX_FILE)
        with open(FAISS_META_FILE, "w", encoding="utf-8") as f:
            json.dump({"id_map": self.id_map}, f, indent=2)
 
    def _load_meta(self):
        if not os.path.exists(FAISS_META_FILE):
            return
        with open(FAISS_META_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.id_map = data.get("id_map", [])
        self.id_to_index = {
            mid: idx for idx, mid in enumerate(self.id_map)
        }
 
    def add(self, embedding, memory_id: str):
        with self._lock:
            if memory_id in self.id_to_index:
                return
 
            vec = self._normalize(embedding)
            self.index.add(vec)
 
            idx = len(self.id_map)
            self.id_map.append(memory_id)
            self.id_to_index[memory_id] = idx
 
            self._persist()
 
    def update(self, embedding, memory_id: str):
        with self._lock:
            if memory_id not in self.id_to_index:
                self.add(embedding, memory_id)
                return
 
            self._embedding_cache: dict
            if not hasattr(self, "_embedding_cache"):
                self._embedding_cache = {}
 
            self._embedding_cache[memory_id] = embedding
 
            new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
            new_id_map = []
            new_id_to_index = {}
 
            for pos, mid in enumerate(self.id_map):
                vec_data = self._embedding_cache.get(mid)
                if vec_data is None:
                    continue
 
                vec = self._normalize(vec_data)
                new_index.add(vec)
                new_id_to_index[mid] = len(new_id_map)
                new_id_map.append(mid)
 
            self.index = new_index
            self.id_map = new_id_map
            self.id_to_index = new_id_to_index
 
            self._persist()
 
    def rebuild(self, memories):
        with self._lock:
            if not hasattr(self, "_embedding_cache"):
                self._embedding_cache = {}
 
            self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
            self.id_map = []
            self.id_to_index = {}
 
            for m in memories:
                if not m.embedding:
                    continue
 
                self._embedding_cache[m.id] = m.embedding
 
                vec = self._normalize(m.embedding)
                self.index.add(vec)
 
                idx = len(self.id_map)
                self.id_map.append(m.id)
                self.id_to_index[m.id] = idx
 
            self._persist()
 
    def search(self, embedding, k: int = 5) -> list:
        with self._lock:
            if not self.id_map:
                return []
 
            k = min(k, len(self.id_map))
            vec = self._normalize(embedding)
            scores, indices = self.index.search(vec, k)
 
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1 or idx >= len(self.id_map):
                    continue
                results.append((self.id_map[idx], float(score)))
 
            return results