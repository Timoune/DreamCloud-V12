import os
import sqlite3

from config.memory_config import MEMORY_PATH

_DB_PATH = os.path.join(MEMORY_PATH, "dreamcloud.db")


def memory_count() -> int:
    """
    Return the number of memories stored in the SQLite database.

    BUG FIX: the original implementation counted .json files in MEMORY_PATH,
    but since v5/v6 memories are persisted in a single SQLite database
    (dreamcloud.db), not as individual JSON files.  That always returned 0.
    """
    if not os.path.exists(_DB_PATH):
        return 0

    try:
        conn = sqlite3.connect(_DB_PATH)
        row = conn.execute("SELECT COUNT(*) FROM memories").fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return 0