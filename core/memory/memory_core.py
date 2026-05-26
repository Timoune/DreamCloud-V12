from core.memory.store import save_memory
 
 
class MemoryCore:
 
    def __init__(self, embedder, index):
        self.embedder = embedder
        self.index = index
 
    def store(self, memory):
        # ensure embedding exists
        memory.embedding = self.embed(memory.content)
 
        # persist
        save_memory(memory)
 
        # index
        self.index.add(memory.embedding, memory.id)
 
    def embed(self, text):
        return self.embedder.encode(text)
 
    def retrieve(self, query, k=5):
        embedding = self.embed(query)
        return self.index.search(embedding, k)