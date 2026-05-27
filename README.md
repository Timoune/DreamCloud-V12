# DreamCloud V14

**Cognitive Memory Subsystem for Mini Von Architecture**

A persistent, intelligent long-term memory system featuring semantic retrieval, graph associations, reinforcement dynamics, contradiction resolution (NLI-powered), autonomous "DreamCycle" concept synthesis, and advanced **Activation Decay Architectures**.

---

## ✨ Key Features (V14)

- **Activation Decay Architectures**: Global Sweep, Lazy, and Hybrid models for scalable cognitive homeostasis
- **Persistent Hybrid Storage**: SQLite + FAISS vector index
- **Semantic + Graph Retrieval**: Vector similarity + relational graph expansion
- **DreamCycle**: Background concept clustering and knowledge consolidation
- **Dynamic Reinforcement & Decay**: Importance scoring with usage-based boosting and temporal decay
- **Advanced Contradiction Detection**: Two-stage (heuristic pre-filter + NLI Cross-Encoder)
- And all previous V12 features...

## Activation Decay (New in V14)

**Global Sweep Decay**:
- Simple `activation *= decay_rate`
- Easy debugging

**Lazy Decay**:
- On-access computation: `activation * exp(-lambda * delta_time)`
- Highly scalable for large graphs

**Hybrid (Default)**:
- Lazy + periodic consolidation + anomaly detection (runaway reinforcement, over-activation, monopolization)

See `config/memory_config.py` for tuning.

## Quick Start
Same as before.

DreamCloud V14 — Now with proper long-term activation dynamics.