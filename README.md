<div align="center">

<img src="assets/banner.png" alt="QueryBridge Banner" width="100%"/>

<br/>

# 🔗 QueryBridge — Hybrid RAG with Active Knowledge Gap Detection

**A self-aware, evaluation-driven Retrieval-Augmented Generation system that scores retrieval confidence, detects contradictions, identifies knowledge gaps, and autonomously fills them through intelligent web search — or escalates to a human with a structured report.**

<br/>

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge&logo=databricks&logoColor=white)](https://www.trychroma.com/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLM-7C3AED?style=for-the-badge&logo=meta&logoColor=white)](https://console.groq.com)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

<br/>

[🚀 Quick Start](#-quick-start) · [🧠 How It Works](#-how-it-works) · [📊 Confidence Scoring](#-confidence-scoring) · [🌐 API & Dashboard](#-api--dashboard) · [📈 Evaluation](#-evaluation)

</div>

---

## 🎯 What Makes QueryBridge Different?

Most RAG systems either answer or say *"I don't know."* QueryBridge goes further — it understands **why** it doesn't know and takes action:

| Capability | What it does |
|---|---|
| 🎯 **Multi-factor Confidence Scoring** | Combines retrieval similarity, context coverage, and source agreement into a calibrated 0–1 score |
| ⚔️ **Contradiction Detection** | Pairwise LLM comparison of retrieved chunks to catch conflicting claims — applies a penalty to confidence |
| 🔍 **Gap Classification** | Distinguishes between `missing`, `partial`, and `contradictory` gaps — not just "low confidence" |
| 🌐 **Targeted Query Generation** | Generates 3–5 diverse search queries, each targeting a different information angle |
| 🏆 **Search Ranking & Deduplication** | Scores queries for specificity, removes near-duplicates, selects the best candidate |
| 🔀 **Intelligent Routing** | Answers directly when confident, triggers web search when not, escalates when search still fails |
| 🚨 **Structured Escalation** | Logs unresolvable queries with full context: confidence, gap type, attempted queries, sources checked |
| 📊 **7-Metric Self-Evaluation** | Measures its own retrieval precision, recall, faithfulness, and gap detection accuracy |

---

## 🧠 How It Works

```
User Query
    ↓
┌───────────────────────┐
│  Document Loader      │   PDF, TXT, Markdown ingestion
│  + Text Splitter      │   Recursive chunking with overlap
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  Sentence Transformers │   all-MiniLM-L6-v2 embeddings
│  → ChromaDB            │   Persistent vector store
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  Retriever            │   Top-k chunks with cosine similarity
│  + Deduplication      │   Collapse overlapping passages
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  Contradiction        │   Pairwise LLM comparison of chunks
│  Detector             │   → penalty score (0.0–0.3)
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  Confidence Scorer    │   Multi-factor: similarity × 0.4
│                       │   + coverage × 0.4 + agreement × 0.2
│                       │   − contradiction penalty
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  Gap Detector         │   Sufficient context?
└──────┬───────┬────────┘
       │       │
   YES ↓       ↓ NO
       │   ┌───────────────────┐
       │   │  Query Generator  │   3–5 targeted search queries
       │   └────────┬──────────┘
       │            ↓
       │   ┌───────────────────┐
       │   │  Search Ranker    │   Score, deduplicate, select best
       │   └────────┬──────────┘
       │            ↓
       │   ┌───────────────────┐
       │   │  Web Search       │   Serper.dev + Tavily fallback
       │   └────────┬──────────┘
       │            ↓
       │   ┌───────────────────┐
       │   │  Re-score         │   Still low? → Escalate
       │   └────────┬──────────┘
       │            │
       ↓            ↓
┌───────────────────────┐
│  Answer Generator     │   Source-attributed response
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  Evaluator            │   7 quality metrics
└──────────┬────────────┘
           ↓
┌───────────────────────┐
│  Streamlit Dashboard  │   Full reasoning trace visualization
└───────────────────────┘
```

---

## ✨ Features at a Glance

| 🔧 Component | 📌 What it does |
|---|---|
| 🎵 **Document Ingestion** | Loads PDF, TXT, and Markdown files into structured chunks |
| 🧠 **Embedding Pipeline** | Batch sentence-transformer embeddings with local disk cache |
| 🗄️ **Vector Store** | ChromaDB with cosine similarity search and persistent storage |
| 🔍 **Smart Retrieval** | Top-k retrieval with duplicate filtering and over-fetching |
| ⚔️ **Contradiction Detection** | LLM-based pairwise chunk comparison with confidence penalty |
| 📊 **Confidence Scoring** | 4-signal calibrated score with human-readable labels |
| 🔍 **Gap Detection** | Classifies gaps as missing, partial, or contradictory |
| 🌐 **Query Generation** | 3–5 diverse, angle-specific search queries per gap |
| 🏆 **Search Ranking** | Specificity scoring and deduplication before web search |
| 🔀 **Intelligent Router** | Answer → Search → Escalate decision tree |
| 🚨 **Escalation Store** | JSON-persisted escalation records with full context |
| 📈 **Self-Evaluation** | 7 metrics against benchmark dataset |
| 🖥️ **Streamlit Dashboard** | Full reasoning trace from query to answer |
| 📡 **FastAPI Backend** | REST API for pipeline execution and escalation retrieval |

---

## 🗂️ Project Structure

```
QueryBridge/
│
├── 📂 querybridge/                   # Core intelligence layer
│   ├── loader.py                     # Document ingestion (PDF, TXT, MD)
│   ├── chunker.py                    # Recursive text splitting with overlap
│   ├── embedder.py                   # Sentence-transformer embeddings + cache
│   ├── vectorstore.py                # ChromaDB wrapper (VectorStore class)
│   ├── retriever.py                  # Top-k retrieval with deduplication
│   ├── contradiction_detector.py     # Pairwise conflict detection via LLM
│   ├── scorer.py                     # Multi-factor confidence scoring
│   ├── gap_detector.py               # Gap type classification
│   ├── query_generator.py            # 3–5 candidate search queries
│   ├── search_ranker.py              # Query scoring and selection
│   ├── search.py                     # Serper.dev + Tavily web search
│   ├── router.py                     # Answer / search / escalate routing
│   ├── escalation_store.py           # JSON-persisted escalation logs
│   ├── evaluator.py                  # 7-metric evaluation engine
│   └── pipeline.py                   # End-to-end orchestration
│
├── 📂 api/
│   └── main.py                       # FastAPI — /query and /escalations
│
├── 📂 dashboard/
│   └── app.py                        # Streamlit reasoning trace UI
│
├── 📂 data/
│   ├── raw/                          # Input documents (PDF, TXT, MD)
│   ├── chunks/                       # Auto-generated processed chunks
│   └── benchmarks/                   # QA pairs for evaluation
│
├── 📂 tests/
│   ├── test_chunker.py
│   ├── test_scorer.py
│   ├── test_contradiction_detector.py
│   ├── test_gap_detector.py
│   └── test_pipeline.py
│
├── 📂 examples/
│   ├── sample_docs/                  # Sample documents for demo
│   └── run_example.py                # Quick-start script
│
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🚀 Quick Start

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/SARWAGYASHAH/QueryBridge-RAG-with-active-knowledge-gap-detection.git
cd QueryBridge-RAG-with-active-knowledge-gap-detection
```

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2️⃣ Configure API Keys

```bash
cp .env.example .env
```

Edit `.env` with your free-tier keys:

```env
GROQ_API_KEY=your_groq_key_here
SERPER_API_KEY=your_serper_key_here      # optional — live web search
TAVILY_API_KEY=your_tavily_key_here      # optional — fallback search
```

> **All keys are free tier:**
> [Groq Console](https://console.groq.com) · [Serper.dev](https://serper.dev) · [Tavily](https://tavily.com)

### 3️⃣ Ingest Documents

```python
from querybridge.pipeline_ingest import ingest_file, ingest_directory

# Ingest a single file
result = ingest_file("data/raw/attention_paper.pdf")
print(f"Ingested {result['chunks_ingested']} chunks")

# Or ingest an entire directory
result = ingest_directory("data/raw/")
print(f"Processed {result['sources_processed']} files, {result['chunks_ingested']} chunks")
```

### 4️⃣ Run a Query

```python
from querybridge.pipeline import run

result = run("What is the Transformer architecture and how does self-attention work?")

print(result["answer"])
print(f"Confidence: {result['confidence']:.2f} ({result['confidence_label']})")
print(f"Gap detected: {result['gap_detected']} ({result['gap_type']})")
print(f"Contradictions: {result['contradictions_found']}")
print(f"Search used: {result['search_used']}")

if result["search_used"]:
    print(f"Search queries generated: {result['search_queries_generated']}")
    print(f"Selected query: {result['selected_query']}")

if result["escalation"]:
    print(f"Escalated: {result['escalation']['reason']}")
```

<details>
<summary>📋 Example Pipeline Output</summary>

```json
{
    "answer": "The Transformer architecture was introduced by Vaswani et al. in 2017...",
    "confidence": 0.82,
    "confidence_label": "high",
    "gap_detected": false,
    "gap_type": "none",
    "contradictions_found": false,
    "search_used": false,
    "search_queries_generated": [],
    "selected_query": null,
    "sources": [
        {"text": "The key innovation is multi-head attention...", "score": 0.89, "source": "attention_paper.pdf"},
        {"text": "Self-attention allows the model to attend...", "score": 0.85, "source": "attention_paper.pdf"}
    ],
    "escalation": null
}
```

</details>

---

## 📊 Confidence Scoring

QueryBridge computes a multi-factor confidence score — not just retrieval similarity:

```
confidence = (similarity × 0.4) + (coverage × 0.4) + (agreement × 0.2) − contradiction_penalty
```

| Signal | Weight | Description |
|---|---|---|
| 📐 **Retrieval Similarity** | 0.4 | Average cosine similarity of top-k retrieved chunks |
| 📋 **Context Coverage** | 0.4 | Proportion of query aspects addressed by the retrieved context |
| 🤝 **Source Agreement** | 0.2 | Consistency of information across different source documents |
| ⚠️ **Contradiction Penalty** | −0.0 to −0.3 | Applied when conflicting claims are detected between chunks |

### Confidence Labels

| Label | Score Range | System Action |
|:---:|:---:|---|
| 🟢 `high` | ≥ 0.7 | Generate answer directly with source attribution |
| 🟡 `medium` | 0.4 – 0.7 | Trigger gap detection → possible web search |
| 🔴 `low` | < 0.4 | Trigger search → escalate if search fails |

---

## ⚔️ Contradiction Detection

When retrieved chunks contain conflicting claims, QueryBridge detects them via LLM-based pairwise comparison:

```python
# Example output from contradiction detector
{
    "contradictions_found": True,
    "conflicting_pairs": [
        {
            "chunk_a": "The model was trained on 100M parameters...",
            "chunk_b": "The architecture uses 340M parameters...",
            "explanation": "Conflicting parameter counts for the same model"
        }
    ],
    "penalty": 0.15
}
```

The contradiction penalty is subtracted from the confidence score, ensuring the system flags uncertain or inconsistent information rather than confidently returning a wrong answer.

---

## 🔍 Gap Detection & Intelligent Search

When confidence is insufficient, QueryBridge classifies the gap and takes action:

| Gap Type | Description | System Action |
|:---:|---|---|
| `missing` | Required information not found in any retrieved chunk | Generate targeted search queries |
| `partial` | Some aspects of the query are covered, others are not | Generate queries for missing aspects |
| `contradictory` | Retrieved chunks contain conflicting information | Search for authoritative resolution |
| `none` | All query aspects covered with consistent information | Answer directly |

### Query Generation & Ranking

```python
# QueryBridge generates 3–5 diverse search queries, not restatements
{
    "ranked_queries": [
        "transformer architecture parameter count official paper",
        "vaswani et al 2017 model size specifications",
        "attention is all you need model dimensions",
        "transformer base model vs large model parameters"
    ],
    "selected_query": "transformer architecture parameter count official paper"
}
```

Each query targets a **different information angle** — the search ranker scores them for specificity, removes near-duplicates, and selects the best candidate before executing the web search.

---

## 🌐 API & Dashboard

### FastAPI Backend

```bash
uvicorn api.main:app --reload
```

| Method | Endpoint | Description |
|:---:|---|---|
| `POST` | `/query` | Run the full pipeline on a query |
| `GET` | `/escalations` | Retrieve all stored escalation events |

#### Example Request

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France and when was the Eiffel Tower built?"}'
```

<details>
<summary>📋 Example Response</summary>

```json
{
    "answer": "The capital of France is Paris. The Eiffel Tower was built between 1887 and 1889...",
    "confidence": 0.91,
    "gap_detected": false,
    "gap_type": "none",
    "contradictions_found": false,
    "search_used": false,
    "sources": ["france_guide.pdf", "european_landmarks.txt"]
}
```

</details>

### Streamlit Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard visualizes the **full reasoning trace** for every query:

- 📥 Retrieved chunks with similarity scores
- ⚔️ Contradiction detection results
- 📊 Confidence score breakdown (all 4 signals)
- 🔍 Gap classification and type
- 🌐 Generated search queries with rankings
- 🔎 Web search results (if triggered)
- 💬 Final answer with source attribution
- 🚨 Escalation status and report

---

## 📈 Evaluation

QueryBridge measures itself across **7 quality metrics** against a benchmark dataset:

| # | Metric | What it measures |
|:---:|---|---|
| 1 | **Retrieval Precision** | Proportion of retrieved chunks that are relevant |
| 2 | **Retrieval Recall** | Proportion of relevant chunks that were retrieved |
| 3 | **Gap Detection Accuracy** | Correct identification of knowledge gaps |
| 4 | **Search Trigger Precision** | Searches triggered correctly (not false alarms) |
| 5 | **Search Trigger Recall** | Gaps that correctly triggered a search |
| 6 | **Answer Faithfulness** | Answers grounded in retrieved evidence |
| 7 | **Contradiction Detection Accuracy** | Correct identification of conflicting claims |

```bash
python -m querybridge.evaluator --benchmark data/benchmarks/benchmark.json
```

<details>
<summary>📋 Example Evaluation Output</summary>

```json
{
    "retrieval_precision": 0.84,
    "retrieval_recall": 0.79,
    "gap_detection_accuracy": 0.88,
    "search_trigger_precision": 0.91,
    "search_trigger_recall": 0.76,
    "answer_faithfulness": 0.93,
    "contradiction_detection_accuracy": 0.85,
    "total_queries_evaluated": 50
}
```

</details>

---

## 🚨 Escalation System

When both retrieval and web search fail to provide sufficient confidence, QueryBridge escalates to a human with a structured, actionable report:

```json
{
    "timestamp": "2026-06-15T14:32:00+05:30",
    "query": "What is the exact energy consumption of GPT-4 during training?",
    "confidence": 0.22,
    "reason": "No authoritative source found; available data is speculative",
    "missing_info": "Official energy consumption figures from OpenAI",
    "suggested_queries": [
        "GPT-4 training energy consumption official report",
        "openai GPT-4 compute requirements published data"
    ],
    "sources_checked": ["arxiv_papers.pdf", "ai_news.txt", "serper_web_results"]
}
```

All escalation events are persisted to JSON and retrievable via the `/escalations` API endpoint.

---

## 🧩 Intelligence Layer — Module Reference

These are the custom-built modules that make QueryBridge work. No external orchestration frameworks — pure Python logic:

| Module | Responsibility |
|---|---|
| `scorer.py` | Multi-factor confidence scoring using similarity, coverage, agreement, and contradiction signals |
| `contradiction_detector.py` | LLM-based pairwise chunk comparison for conflicting claims |
| `gap_detector.py` | Classify gap type (missing / partial / contradictory / none) and decide action |
| `query_generator.py` | Generate 3–5 candidate search queries, each targeting a different angle |
| `search_ranker.py` | Score queries for specificity, deduplicate, select best |
| `search.py` | Web search via Serper.dev with Tavily fallback and rate limit handling |
| `router.py` | Route to answer, search, or escalation based on confidence and gap type |
| `escalation_store.py` | Persist escalation events with full context to JSON |
| `evaluator.py` | Measure system quality across 7 metrics against benchmark data |
| `pipeline.py` | End-to-end orchestration — single `run(query)` entry point |

---

## 🔧 Tech Stack

<div align="center">

| Layer | Technology |
|---|---|
| 🧠 **Embeddings** | sentence-transformers/all-MiniLM-L6-v2 |
| 🗄️ **Vector Store** | ChromaDB (persistent, cosine similarity) |
| 🤖 **LLM** | Groq (llama3-8b-8192) / Ollama (local) |
| 🌐 **Web Search** | Serper.dev (primary) / Tavily (fallback) |
| 📡 **Backend** | FastAPI |
| 🖥️ **Dashboard** | Streamlit |
| 🐍 **Language** | Python 3.10+ |

</div>

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suite
pytest tests/test_scorer.py -v
pytest tests/test_pipeline.py -v

# With coverage report
pytest tests/ --cov=querybridge --cov-report=term-missing
```

---

## 📦 Dependencies

<div align="center">

| Category | Libraries |
|---|---|
| 🧠 Embeddings | `sentence-transformers` |
| 🗄️ Vector Store | `chromadb` |
| 🤖 LLM | `groq` · `langchain-groq` |
| 🌐 Search | `requests` |
| 📡 API | `fastapi` · `uvicorn` |
| 🖥️ Dashboard | `streamlit` |
| 📄 Document Loading | `pypdf` · `unstructured` |
| 🧪 Testing | `pytest` |
| 🔧 Utilities | `python-dotenv` · `pydantic` |

</div>

---

## 🗺️ Architecture Overview

```
                          ┌─────────────────────────────────────┐
                          │           User Query                │
                          └─────────────────┬───────────────────┘
                                            │
                                            ▼
                   ┌────────────────────────────────────────────┐
                   │         Document Ingestion Layer           │
                   │  loader.py → chunker.py → embedder.py     │
                   │              → vectorstore.py              │
                   └────────────────────────┬───────────────────┘
                                            │
                                            ▼
                   ┌────────────────────────────────────────────┐
                   │           Retrieval Layer                  │
                   │  retriever.py — top-k + deduplication      │
                   └────────────────────────┬───────────────────┘
                                            │
                          ┌─────────────────┼──────────────────┐
                          ▼                                    ▼
              ┌───────────────────┐                ┌───────────────────┐
              │  Contradiction    │                │  Confidence       │
              │  Detector         │───penalty──→   │  Scorer           │
              └───────────────────┘                └─────────┬─────────┘
                                                             │
                                                             ▼
                                                   ┌─────────────────┐
                                                   │  Gap Detector   │
                                                   └──┬──────────┬───┘
                                              answer  │          │ search
                                                      ▼          ▼
                                              ┌──────────┐  ┌──────────────┐
                                              │ Generate  │  │ Query Gen    │
                                              │ Answer    │  │ → Ranker     │
                                              └──────────┘  │ → Web Search │
                                                            │ → Re-score   │
                                                            └──────┬───────┘
                                                                   │
                                                         ┌─────────┴─────────┐
                                                    still low?           sufficient?
                                                         ▼                    ▼
                                                   ┌──────────┐       ┌──────────┐
                                                   │ Escalate │       │ Answer   │
                                                   └──────────┘       └──────────┘
```

---

<div align="center">

**Built with ❤️ by [Sarwagya Shah](https://github.com/SARWAGYASHAH)**

*If this project helped you, consider giving it a ⭐ on GitHub!*

</div>