"""
Prompt assembly for the local LLM.

Changes from v2
---------------
* Memory block groups entries by type so the LLM sees a structured,
  labelled context rather than a flat list.  Identity and preference
  memories appear first (most useful for personalised responses).

* Type labels in the prompt help the model understand what kind of
  information it is reading (e.g. [IDENTITY] vs [FACT]).

* MEMORY_WINDOW remains 6; high-priority types (identity, preference,
  goal) get guaranteed inclusion up to their PRIORITY_SLOTS budget before
  the remaining slots are filled by score order.
"""

from core.memory.schema import MemoryType

MEMORY_WINDOW   = 6    # total memories injected into prompt
PRIORITY_SLOTS  = 3    # slots reserved for high-priority types
GRAPH_NEIGHBORS = 2    # neighbours shown per memory in graph block

_PRIORITY_TYPES = {
    MemoryType.IDENTITY,
    MemoryType.PREFERENCE,
    MemoryType.GOAL,
}

_TYPE_LABEL = {
    MemoryType.IDENTITY:   "IDENTITY",
    MemoryType.PREFERENCE: "PREFERENCE",
    MemoryType.GOAL:       "GOAL",
    MemoryType.EPISODIC:   "EPISODIC",
    MemoryType.EMOTIONAL:  "EMOTIONAL",
    MemoryType.FACT:       "FACT",
    MemoryType.CONCEPT:    "CONCEPT",
    MemoryType.GENERAL:    "MEMORY",
}


def _select_memories(memories: list) -> list:
    """
    Pick MEMORY_WINDOW memories with PRIORITY_SLOTS guaranteed for high-
    priority types, remainder filled by score order.
    """
    priority = [m for m in memories if m.type in _PRIORITY_TYPES]
    others   = [m for m in memories if m.type not in _PRIORITY_TYPES]

    selected = priority[:PRIORITY_SLOTS] + others
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for m in selected:
        if m.id not in seen:
            seen.add(m.id)
            deduped.append(m)

    return deduped[:MEMORY_WINDOW]


def build_prompt(
    user_input: str,
    memories:   list,
    session,
    graph=None,
    system: str = "MiniVon",
) -> str:

    top_memories = _select_memories(memories)

    # -------------------------
    # Memory block — typed labels
    # -------------------------
    if top_memories:
        memory_lines = []
        for m in top_memories:
            label = _TYPE_LABEL.get(m.type, "MEMORY")
            memory_lines.append(f"[{label}] {m.content}")
        memory_block = "\n".join(memory_lines)
    else:
        memory_block = "None"

    # -------------------------
    # Graph relationship block
    # -------------------------
    graph_lines = []
    if graph:
        for m in top_memories:
            neighbours = graph.get_neighbors(m.id, top_k=GRAPH_NEIGHBORS)
            for nid, _ in neighbours:
                target = next((x for x in top_memories if x.id == nid), None)
                if target:
                    graph_lines.append(f"- {m.content} → {target.content}")

    graph_block = "\n".join(graph_lines) if graph_lines else "None"

    # -------------------------
    # Session block (last 4 turns)
    # -------------------------
    session_block = "\n".join(
        f"{msg['role']}: {msg['content']}"
        for msg in session.messages[-4:]
    )

    # -------------------------
    # Final prompt
    # -------------------------
    return f"""
You are {system}, an AI with persistent memory of the user.

Use the memory entries below to answer personally and accurately.
IDENTITY and PREFERENCE memories are especially important — they define who
the user is.  Do not invent facts not present in memory.

=== MEMORY ===
{memory_block}

=== MEMORY RELATIONSHIPS ===
{graph_block}

=== RECENT CONVERSATION ===
{session_block}

User: {user_input}
Assistant:
""".strip()