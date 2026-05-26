from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional

from core.memory.schema import Memory
from core.memory.memory_core import MemoryCore
from core.embeddings.embedder import Embedder
from core.retrieval.faiss_index import FaissIndex          # BUG FIX #1: was FAISSIndex
from core.retrieval.retrieve import retrieve_memories


app = FastAPI()


# --------------------------------------------------
# INITIALIZE DREAMCLOUD CORE
# --------------------------------------------------

embedder = Embedder()

index = FaissIndex()                                        # BUG FIX #1: was FAISSIndex()

memory_core = MemoryCore(
    embedder=embedder,
    index=index
)


# --------------------------------------------------
# REQUEST MODELS
# --------------------------------------------------

class StoreRequest(BaseModel):
    content: str
    memory_type: str = "general"
    metadata: Optional[dict] = {}


class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5


# --------------------------------------------------
# STORE MEMORY
# --------------------------------------------------

@app.post("/store")

async def store_memory(
    request: StoreRequest
):

    memory = Memory(
        content=request.content,
        type=request.memory_type,
        metadata=request.metadata
    )

    memory_core.store(memory)

    return {
        "status": "success",
        "memory_id": memory.id
    }


# --------------------------------------------------
# RETRIEVE MEMORY
# --------------------------------------------------

@app.post("/retrieve")

async def retrieve_memory(
    request: RetrieveRequest
):

    memories = retrieve_memories(
        query=request.query,
        embedder=embedder,
        index=index,
        k=request.top_k
    )

    results = []

    for memory in memories:

        results.append({
            "id": memory.id,
            "content": memory.content,
            "importance": memory.importance,
            "reliability": memory.reliability,
            "metadata": memory.metadata
        })

    return {
        "status": "success",
        "results": results
    }


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------

@app.get("/health")

async def health():

    return {
        "status": "online",
        "system": "DreamCloudV14"                          # BUG FIX #9: was V12
    }
