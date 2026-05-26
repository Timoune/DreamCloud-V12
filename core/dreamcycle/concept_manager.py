from config.memory_config import (
    CONCEPT_MIN_SIZE,
    CONCEPT_CONFIDENCE_THRESHOLD,
)
from core.memory.schema import Memory, MemoryType
from core.memory.store import save_memory
 
 
def create_concept(summary: str, sources: list, embedder) -> "Memory | None":
    unique_sources = list({s.id: s for s in sources}.values())
 
    if len(unique_sources) < CONCEPT_MIN_SIZE:
        return None
 
    m = Memory(
        content=summary,
        is_concept=True,
        type=MemoryType.CONCEPT,
        source_ids=[s.id for s in unique_sources],
    )
 
    m.embedding = embedder.encode(summary)
 
    total_importance = sum(s.importance for s in unique_sources)
    m.confidence = total_importance / max(len(unique_sources), 1)
 
    if m.confidence < CONCEPT_CONFIDENCE_THRESHOLD:
        return None
 
    save_memory(m)
    return m