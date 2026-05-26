import os
import json
import sqlite3
import threading

from config.memory_config import MEMORY_PATH, SCHEMA_VERSION
from core.memory.schema import Memory, MemoryType

os.makedirs(MEMORY_PATH, exist_ok=True)

DB_PATH = os.path.join(MEMORY_PATH, "dreamcloud.db")

_lock = threading.RLock()


# ---------------------------------------------------------------------------
# DB init
# ---------------------------------------------------------------------------

def _get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database():
    with _lock:
        with _get_conn() as conn:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS memories (
                id        TEXT PRIMARY KEY,
                timestamp REAL,
                content   TEXT,
                payload   TEXT
            )
            """)
            conn.commit()


initialize_database()


# ---------------------------------------------------------------------------
# Migration functions
# ---------------------------------------------------------------------------

def _migrate_v1_to_v2(data: dict) -> dict:
    data["schema_version"] = 2
    return data


def _migrate_v2_to_v3(data: dict) -> dict:
    from core.memory.schema import VALID_TYPES

    data.setdefault("reliability", 0.5)
    data.setdefault("reinforcement_count", 0)

    if data.get("type") not in VALID_TYPES:
        data["type"] = MemoryType.GENERAL

    data["schema_version"] = 3
    return data


def _migrate_v3_to_v4(data: dict) -> dict:
    """Add cognitive homeostasis fields (activation, last_activated)."""
    data.setdefault("activation", 0.0)
    data.setdefault("last_activated", 0.0)
    data["schema_version"] = 4
    return data


_MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
}


# ---------------------------------------------------------------------------
# Migration runner
# ---------------------------------------------------------------------------

def _apply_migrations(data: dict) -> dict:
    version = data.get("schema_version", 1)
    while version < SCHEMA_VERSION:
        fn = _MIGRATIONS.get(version)
        if fn is None:
            version += 1
            data["schema_version"] = version
        else:
            data = fn(data)
            version = data.get("schema_version", version + 1)
    return data


def _deserialize(payload: str) -> Memory:
    data = json.loads(payload)
    data = _apply_migrations(data)
    return Memory.from_dict(data)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_memory(memory: Memory):
    payload = json.dumps(memory.to_dict())
    with _lock:
        with _get_conn() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO memories
                (id, timestamp, content, payload)
            VALUES (?, ?, ?, ?)
            """, (memory.id, memory.timestamp, memory.content, payload))
            conn.commit()


def load_memory(memory_id: str) -> Memory:
    with _lock:
        with _get_conn() as conn:
            row = conn.execute(
                "SELECT payload FROM memories WHERE id = ?",
                (memory_id,)
            ).fetchone()

        if not row:
            raise ValueError(f"Memory not found: {memory_id}")

        return _deserialize(row["payload"])


def load_all_memories() -> list:
    with _lock:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM memories ORDER BY timestamp ASC"
            ).fetchall()

        memories = []
        for row in rows:
            try:
                memories.append(_deserialize(row["payload"]))
            except Exception as e:
                print(f"[WARN] Failed memory decode: {e}")
        return memories


def delete_memory(memory_id: str):
    with _lock:
        with _get_conn() as conn:
            conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
            conn.commit()
