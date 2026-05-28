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
                id              TEXT PRIMARY KEY,
                timestamp       REAL,
                content         TEXT,
                retention_class TEXT DEFAULT 'volatile',
                payload         TEXT
            )
            """)
            # Add retention_class column to existing DBs that pre-date v6
            try:
                conn.execute(
                    "ALTER TABLE memories ADD COLUMN retention_class TEXT DEFAULT 'volatile'"
                )
            except sqlite3.OperationalError:
                pass  # column already exists
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


def _migrate_v4_to_v5(data: dict) -> dict:
    """BUG FIX #6: v5 added full decay_strategy support.
    No new fields required â this migration simply stamps the version
    so existing records are recognised as current and aren't re-migrated
    on every load."""
    data["schema_version"] = 5
    return data


def _migrate_v5_to_v6(data: dict) -> dict:
    """
    v6: add structured retention_policy.

    Existing memories without a policy get the default VOLATILE policy so
    they retain normal pruning behaviour.  Memories that previously carried
    a legacy boolean ``archived`` flag inside metadata are upgraded to
    PROTECTED so no archival intent is silently discarded.
    """
    from core.memory.retention import DEFAULT_POLICY, RetentionClass, RetentionSource

    if "retention_policy" not in data:
        metadata = data.get("metadata") or {}
        was_archived = metadata.pop("archived", False)

        if was_archived:
            data["retention_policy"] = {
                "class":   RetentionClass.PROTECTED,
                "source":  RetentionSource.SYSTEM,
                "reason":  "migrated_from_archived_flag",
                "expires": None,
            }
        else:
            data["retention_policy"] = dict(DEFAULT_POLICY)

    data["schema_version"] = 6
    return data


_MIGRATIONS = {
    1: _migrate_v1_to_v2,
    2: _migrate_v2_to_v3,
    3: _migrate_v3_to_v4,
    4: _migrate_v4_to_v5,   # BUG FIX #6: was missing, causing silent no-op skip
    5: _migrate_v5_to_v6,
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
    """Persist a memory, indexing its retention_class for fast queries."""
    from core.memory.retention import RetentionPolicy

    rp = RetentionPolicy.from_dict(memory.retention_policy)
    retention_class = rp.effective_class()

    payload = json.dumps(memory.to_dict())
    with _lock:
        with _get_conn() as conn:
            conn.execute("""
            INSERT OR REPLACE INTO memories
                (id, timestamp, content, retention_class, payload)
            VALUES (?, ?, ?, ?, ?)
            """, (memory.id, memory.timestamp, memory.content, retention_class, payload))
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


def load_critical_memories() -> list:
    """
    Return all memories whose retention class is CRITICAL.

    Uses the indexed ``retention_class`` column for an efficient query
    rather than deserialising every row.  Useful for audit sweeps and
    system health checks.
    """
    with _lock:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM memories WHERE retention_class = 'critical' ORDER BY timestamp ASC"
            ).fetchall()

        memories = []
        for row in rows:
            try:
                memories.append(_deserialize(row["payload"]))
            except Exception as e:
                print(f"[WARN] Failed critical memory decode: {e}")
        return memories


def load_memories_by_retention_class(retention_class: str) -> list:
    """
    Return all memories matching the given retention class string.

    Parameters
    ----------
    retention_class : str
        One of 'volatile', 'standard', 'protected', 'critical'.
    """
    with _lock:
        with _get_conn() as conn:
            rows = conn.execute(
                "SELECT payload FROM memories WHERE retention_class = ? ORDER BY timestamp ASC",
                (retention_class,)
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
