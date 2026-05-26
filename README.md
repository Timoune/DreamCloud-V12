 # DreamCloud V12

**Cognitive Memory Subsystem for Mini Von Architecture**

A persistent, intelligent long-term memory system featuring semantic retrieval, graph associations, reinforcement dynamics, contradiction resolution (NLI-powered), and autonomous "DreamCycle" concept synthesis.

---

## ✨ Key Features

- **Persistent Hybrid Storage**: SQLite + FAISS vector index
- **Semantic + Graph Retrieval**: Vector similarity + relational graph expansion
- **DreamCycle**: Background concept clustering and knowledge consolidation
- **Dynamic Reinforcement & Decay**: Importance scoring with usage-based boosting and temporal decay
- **Advanced Contradiction Detection**: Two-stage (heuristic pre-filter + NLI Cross-Encoder)
- **Typed Memories**: Support for different memory types (episodic, semantic, procedural, etc.)
- **Background Processing**: Bounded queue for non-blocking memory extraction
- **REST API**: FastAPI endpoints for integration
- **Robust Infrastructure**: Logging, diagnostics, event bus, recovery hooks

---

## 📁 Project Structure

DreamCloud-V12/ ├── main.py # Entry point ├── requirements.txt ├── README.md ├── api/ │ └── server.py # FastAPI REST interface ├── config/ │ ├── memory_config.py │ ├── model_config.py │ └── runtime_config.py ├── core/ │ ├── dreamcycle/ │ │ ├── dream_cycle.py │ │ └── concept_manager.py │ ├── embeddings/ │ │ └── embedder.py │ ├── graph/ │ │ └── graph_manager.py │ ├── infrastructure/ │ │ ├── logger.py │ │ ├── event_bus.py │ │ ├── diagnostics.py │ │ └── recovery.py │ ├── memory/ │ │ ├── store.py │ │ ├── schema.py │ │ ├── extractor.py │ │ ├── pipeline.py │ │ ├── memory_core.py │ │ ├── memory_filter.py │ │ ├── validator.py │ │ └── serializer.py │ ├── retrieval/ │ │ ├── faiss_index.py │ │ ├── retrieve.py │ │ └── ranking.py │ ├── runtime/ │ │ ├── engine.py # Main orchestration engine │ │ ├── prompt_builder.py │ │ ├── llama_runner.py │ │ ├── nli_validator.py │ │ ├── response_cleaner.py │ │ └── session.py │ └── utils/ │ ├── filesystem.py │ ├── hashing.py │ └── timers.py ├── models/ │ ├── embeddings/ │ └── nli/ ├── data/ # SQLite, FAISS, logs, graph (gitignored) └── tests/ ├── test_memory_basic.py ├── test_memory_intelligence.py └── test_dreamcloud_v3.py
---

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/Timoune/DreamCloud-V12.git
   cd DreamCloud-V12
	2	Install dependencies pip install -r requirements.txt
	3	
	4	Start llama.cpp server (required for inference)
	◦	Run your local llama.cpp server on http://localhost:8080
	5	Run DreamCloud python main.py
	6	
	7	(Optional) Start API server cd api
	8	uvicorn server:app --reload
	9	

🧠 How It Works
Memory Flow
	1	User input → Typed memory extraction (background thread)
	2	Embedding + FAISS indexing
	3	Graph association
	4	Retrieval on query: Vector search → Graph expansion → Re-ranking (importance, recency, reliability)
	5	Contradiction resolution via NLI
	6	LLM prompt augmentation with enriched context
	7	Reinforcement on used memories
	8	Periodic DreamCycle for concept synthesis
DreamCycle
	•	Runs every ~60 seconds in background
	•	Clusters similar memories into higher-level concepts
	•	Strengthens associations in the graph
	•	Prunes weak/low-importance entries

🔧 Configuration
Key settings in config/memory_config.py:
	•	TOP_K, embedding dimensions, reinforcement thresholds
	•	CONTRADICTION_PENALTY, NLI_CONTRADICTION_THRESHOLD
	•	DreamCycle parameters
	•	Paths for data persistence

🧪 Testing
python -m pytest tests/
Includes basic functionality, intelligence/retrieval, and integration tests.

📡 API Endpoints
	•	POST /store — Store new memory
	•	POST /retrieve — Semantic retrieval
	•	GET /health — System status

🛠️ Architecture Highlights (V12 Improvements)
	•	Efficient Contradiction Detection: Heuristic filter + NLI model (much faster than previous LLM prompting)
	•	Bounded Background Extraction: Single worker thread + queue (prevents resource exhaustion)
	•	Graph Expansion: Relational reasoning beyond pure vector search
	•	Modular & Observable: Clean separation of concerns + comprehensive logging

DreamCloud V12 — Turning conversations into structured, evolving knowledge.
Built for Mini Von.