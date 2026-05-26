from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
 
from core.memory.schema import Memory
from core.memory.memory_core import MemoryCore
from core.embeddings.embedder import Embedder
from core.retrieval.faiss_index import FAISSIndex
from core.retrieval.retrieve import retrieve_memories
 
 
app = FastAPI()