# core/memory/store.py

import json
import os
from core.memory.schema import Memory
from config.memory_config import SCHEMA_VERSION

MEMORY_DIR = "data/memories"
os.makedirs(MEMORY_DIR, exist_ok=True)

def _migrate_v1_to_v2(data: dict) -> dict:
    data["reliability"] = data.get("reliability", 1.0)
    data["schema_version"] = 2
    return data

def _migrate_v2_to_v3(data: dict) -> dict:
    data["access_count"] = data.get("access_count", 0)
    data["last_accessed"] = data.get("last_accessed", 0.0)
    data["last_reinforced"] = data.get("last_reinforced", 0.0)
    data["reinforcement_count"] = data.get("reinforcement_count", 0)
    data["schema_version"] = 3
    return data

def _migrate_v3_to_v4(data: dict) -> dict:
    """Migrates memory schema from v3 to v4 by adding activation tracking fields."""
    data["activation"] = data.get("activation", 0.0)
    data["last_activated"] = data.get("last_activated", data.get("last_accessed", data.get("timestamp", 0.0)))
    data["schema_version"] = 4
    return data

_MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,  # Registered migration to v4
}

def load_memory(memory_id: str) -> Memory:
    path = os.path.join(MEMORY_DIR, f"{memory_id}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Memory {memory_id} not found.")
    with open(path, "r") as f:
        data = json.load(f)
    
    current_version = data.get("schema_version", 1)
    while current_version < SCHEMA_VERSION:
        migration_fn = _MIGRATIONS.get(current_version)
        if not migration_fn:
            break
        data = migration_fn(data)
        current_version = data.get("schema_version", current_version)
        
    return Memory(**{k: v for k, v in data.items() if k != "schema_version"})

def save_memory(memory: Memory):
    path = os.path.join(MEMORY_DIR, f"{memory.id}.json")
    data = memory.__dict__.copy()
    data["schema_version"] = SCHEMA_VERSION
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

def load_all_memories() -> list:
    memories = []
    for filename in os.listdir(MEMORY_DIR):
        if filename.endswith(".json"):
            mid = filename.replace(".json", "")
            try:
                memories.append(load_memory(mid))
            except Exception:
                continue
    return memories