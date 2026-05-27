import json

from core.memory.schema import Memory


def serialize_memory(memory: Memory) -> str:
    # BUG FIX #11: was memory.__dict__ which exposes _importance (private field)
    # and produces a format inconsistent with store.py.  Use to_dict() instead.
    return json.dumps(memory.to_dict(), ensure_ascii=False, indent=2)


def deserialize_memory(data: str) -> Memory:
    # BUG FIX #11: was Memory(**json.loads(data)) which bypasses the importance
    # setter and fails when the payload contains 'importance' (not '_importance').
    # Use from_dict() which handles the _importance ↔ importance mapping correctly.
    return Memory.from_dict(json.loads(data))
