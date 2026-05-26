import sys
import os
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.embeddings.embedder import Embedder
from core.retrieval.faiss_index import FaissIndex
from core.memory.memory_core import MemoryCore
from core.memory.schema import Memory
from core.memory.store import load_all_memories, save_memory
from core.retrieval.retrieve import retrieve_memories


def print_memory_state(memories, title):
    print(f"\n=== {title} ===")
    for m in memories:
        print(
            f"{m.content} | importance={m.importance:.4f} | access={m.access_count}"
        )


def run_test():
    print("=== DreamCloud Intelligence Test ===\n")

    embedder = Embedder()
    index = FaissIndex()
    memory_core = MemoryCore(embedder, index)

    # -------------------------
    # STEP 1: CLEAN START
    # -------------------------
    print("[1] Initializing test memories...")

    test_memories = [
        "My favorite color is blue",
        "I like blue sometimes",
        "Blue is a calming color",
        "My favorite color is not blue"
    ]

    for text in test_memories:
        m = Memory(content=text)
        memory_core.store(m)

    # rebuild index
    all_memories = load_all_memories()
    for m in all_memories:
        if not m.embedding:
            m.embedding = embedder.encode(m.content)
        index.add(m.embedding, m.id)

    print(f"Stored {len(all_memories)} memories")

    # -------------------------
    # STEP 2: QUERY LOOP
    # -------------------------
    print("\n[2] Running repeated queries...")

    query = "What is my favorite color?"

    for i in range(5):
        print(f"\n--- Query iteration {i+1} ---")

        results = retrieve_memories(query, embedder, index, k=5)

        for r in results:
            print(f"{r.content} | score={r.metadata.get('score', 0):.4f}")

        time.sleep(1)

    # -------------------------
    # STEP 3: FINAL STATE
    # -------------------------
    final_memories = load_all_memories()

    print_memory_state(final_memories, "FINAL MEMORY STATE")

    # -------------------------
    # STEP 4: VALIDATION
    # -------------------------
    print("\n[3] Validation:")

    correct = None
    contradiction = None

    for m in final_memories:
        if "favorite color is blue" in m.content.lower():
            correct = m
        if "not blue" in m.content.lower():
            contradiction = m

    if correct and contradiction:
        if correct.importance > contradiction.importance:
            print("✅ PASS: Correct memory dominates contradiction")
        else:
            print("❌ FAIL: Contradiction was reinforced incorrectly")

    # weak memory check
    weak = [m for m in final_memories if "sometimes" in m.content.lower()]

    if weak:
        if weak[0].importance < correct.importance:
            print("✅ PASS: Weak memory did not dominate")
        else:
            print("❌ FAIL: Weak memory reinforced too much")

    print("\nTest complete.")


if __name__ == "__main__":
    run_test()