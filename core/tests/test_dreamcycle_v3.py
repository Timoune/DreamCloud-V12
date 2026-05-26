import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.embeddings.embedder import Embedder
from core.retrieval.faiss_index import FaissIndex
from core.memory.memory_core import MemoryCore
from core.memory.schema import Memory
from core.memory.store import load_all_memories
from core.dreamcycle.dream_cycle import DreamCycle


def print_memories(memories, title):
    print(f"\n=== {title} ===")
    for m in memories:
        tag = "CONCEPT" if m.is_concept else "RAW"
        print(
            f"[{tag}] {m.content} | importance={m.importance:.3f} | confidence={getattr(m, 'confidence', 0):.3f}"
        )


def run_test():
    print("=== DreamCycle v3 Test ===\n")

    embedder = Embedder()

    # 🔥 FIX: Proper FAISS index
    index = FaissIndex()
    memory_core = MemoryCore(embedder, index)

    # -------------------------
    # STEP 1: SEED MEMORIES
    # -------------------------
    print("[1] Seeding related memories...")

    seed_data = [
        "I like blue",
        "Blue is calming",
        "My favorite color is blue",
        "I enjoy the color blue",
        "Sometimes I think blue is relaxing"
    ]

    for text in seed_data:
        m = Memory(content=text)
        m.importance = 1.0  # force relevance
        memory_core.store(m)

    # unrelated noise
    noise = [
        "I like pizza",
        "Dogs are cool"
    ]

    for text in noise:
        m = Memory(content=text)
        m.importance = 0.1
        memory_core.store(m)

    print("Memories stored.")

    # -------------------------
    # STEP 2: RUN DREAMCYCLE
    # -------------------------
    print("\n[2] Running DreamCycle...")

    dc = DreamCycle(embedder)
    dc.run()

    # -------------------------
    # STEP 3: LOAD RESULTS
    # -------------------------
    memories = load_all_memories()

    print_memories(memories, "POST-DREAMCYCLE STATE")

    # -------------------------
    # STEP 4: VALIDATION
    # -------------------------
    print("\n[3] Validation:\n")

    concepts = [m for m in memories if m.is_concept]
    raws = [m for m in memories if not m.is_concept]

    # ---- Concept created?
    if len(concepts) > 0:
        print("✅ PASS: Concept(s) created")
    else:
        print("❌ FAIL: No concepts created")

    # ---- Concept quality check
    valid_concept = False
    for c in concepts:
        if "blue" in c.content.lower():
            valid_concept = True
            break

    if valid_concept:
        print("✅ PASS: Concept captures correct theme")
    else:
        print("❌ FAIL: Concept missing expected meaning")

    # ---- Source linking
    linked = any(len(c.source_ids) >= 3 for c in concepts)

    if linked:
        print("✅ PASS: Concept linked to sources")
    else:
        print("❌ FAIL: Concept has weak or missing source links")

    # ---- Noise pruning
    noise_remaining = [m for m in raws if "pizza" in m.content.lower()]

    if len(noise_remaining) == 0:
        print("✅ PASS: Noise memory pruned")
    else:
        print("❌ FAIL: Noise memory not pruned")

    # ---- Raw memory survival (should not all vanish)
    if len(raws) > 0:
        print("✅ PASS: Raw memory layer preserved")
    else:
        print("❌ FAIL: All raw memories deleted (too aggressive)")

    print("\nTest complete.")


if __name__ == "__main__":
    run_test()