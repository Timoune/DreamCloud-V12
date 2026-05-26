 # DreamCloud V12

**Cognitive Memory Subsystem for Mini Von Architecture**

A persistent, intelligent long-term memory system featuring semantic retrieval, graph associations, reinforcement dynamics, contradiction resolution (NLI-powered), and autonomous "DreamCycle" concept synthesis.

**IMPORTANT: This README is not up-to-date, as i am adding more features.

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

## 🚀 Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/Timoune/DreamCloud.git
   cd DreamCloud
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