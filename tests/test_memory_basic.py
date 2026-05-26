from core.embeddings.embedder import Embedder
from core.retrieval.faiss_index import FaissIndex
from core.memory.memory_core import MemoryCore
from core.memory.schema import Memory
from core.memory.store import load_all_memories
from core.retrieval.retrieve import retrieve_memories


def run_test():
    print("=== DreamCloud Memory Test ===\n")

    embedder = Embedder()
    index = FaissIndex()
    memory_core = MemoryCore(embedder, index)

    # -------------------------
    # STEP 1: STORE MEMORY
    # -------------------------
    print("[1] Storing memory...")

    m = Memory(content="My favorite color is blue")
    memory_core.store(m)

    print(f"Stored memory ID: {m.id}\n")

    # -------------------------
    # STEP 2: REBUILD INDEX
    # -------------------------
    print("[2] Rebuilding index...")

    memories = load_all_memories()

    for mem in memories:
        if not mem.embedding:
            mem.embedding = embedder.encode(mem.content)
        index.add(mem.embedding, mem.id)

    print(f"Indexed {len(memories)} memories\n")

    # -------------------------
    # STEP 3: QUERY
    # -------------------------
    print("[3] Querying memory...")

    query = "What is my favorite color?"
    results = retrieve_memories(query, embedder, index, k=5)

    print(f"Query: {query}\n")

    # -------------------------
    # STEP 4: OUTPUT RESULTS
    # -------------------------
    print("[4] Results:")

    if not results:
        print("❌ No memories retrieved")
        return

    for i, mem in enumerate(results):
        score = mem.metadata.get("score", 0)
        print(f"{i+1}. ({score:.4f}) {mem.content}")

    # -------------------------
    # STEP 5: ASSERTION
    # -------------------------
    print("\n[5] Validation:")

    top = results[0].content.lower()

    if "blue" in top:
        print("✅ PASS: Correct memory retrieved")
    else:
        print("❌ FAIL: Memory retrieval incorrect")


if __name__ == "__main__":
    run_test()