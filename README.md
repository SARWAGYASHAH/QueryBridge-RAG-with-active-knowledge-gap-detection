# QueryBridge

**Self-aware RAG with active knowledge gap detection.**

QueryBridge is an evaluation-driven Retrieval-Augmented Generation system that goes beyond simple retrieval. It scores retrieval confidence using multiple signals, detects contradictions across retrieved chunks, identifies exactly what information is missing, generates and ranks targeted search queries to fill that gap, and either retrieves additional information automatically or escalates to a human with a structured, actionable report.

---

## Why QueryBridge?

Most RAG systems either answer or say "I don't know." QueryBridge does something smarter:

- **Multi-factor confidence scoring** — combines retrieval similarity, context coverage, and source agreement into a single calibrated score
- **Contradiction detection** — identifies conflicting claims across retrieved chunks and penalizes confidence accordingly
- **Gap classification** — distinguishes between missing, partial, and contradictory information gaps
- **Targeted query generation** — generates 3–5 distinct search queries, each targeting a different angle, not restatements
- **Intelligent routing** — answers directly when confident, triggers live search when not, escalates when search still fails

---

## System Flow

```
User Query
    ↓
Document Loader + Text Splitter
    ↓
Sentence Transformers → Vector Store
    ↓
Retriever — top-k chunks with similarity scores
    ↓
Contradiction Detector — compare chunks for conflicts
    ↓
Confidence Scorer — multi-factor score
    ↓
Gap Detector — sufficient context?
    ├── YES → Generate answer with source attribution
    └── NO  → Query Generator (3–5 candidates)
                ↓
            Search Ranker — select best query
                ↓
            Web Search
                ↓
            Re-score confidence
                ├── Sufficient → Answer
                └── Still low  → Escalate to human
                                    ↓
                                Escalation Store
    ↓
Evaluator — 7 quality metrics
    ↓
Dashboard — full reasoning trace
```

---

## Intelligence Layer

These are the core custom-built modules that make QueryBridge work:

| Module | Responsibility |
|--------|---------------|
| `scorer.py` | Multi-factor confidence scoring using similarity, coverage, and agreement signals |
| `contradiction_detector.py` | Pairwise chunk comparison to detect conflicting claims |
| `gap_detector.py` | Classify gap type and decide: answer, search, or escalate |
| `query_generator.py` | Generate 3–5 candidate search queries targeting different information angles |
| `search_ranker.py` | Score, deduplicate, and select the best query |
| `router.py` | Route query to the correct action path |
| `evaluator.py` | Measure system quality across 7 metrics |
| `escalation_store.py` | Persist escalation events with structured context |

---

## Project Structure

```
querybridge/
├── README.md
├── requirements.txt
├── .gitignore
├── .env.example
│
├── data/
│   ├── raw/                # input documents (PDF, txt, md)
│   ├── chunks/             # processed chunks (auto-generated)
│   └── benchmarks/         # QA pairs for evaluation
│
├── querybridge/
│   ├── loader.py
│   ├── chunker.py
│   ├── embedder.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── contradiction_detector.py
│   ├── scorer.py
│   ├── gap_detector.py
│   ├── query_generator.py
│   ├── search_ranker.py
│   ├── search.py
│   ├── router.py
│   ├── escalation_store.py
│   ├── evaluator.py
│   └── pipeline.py
│
├── api/
│   └── main.py             # FastAPI — /query and /escalations endpoints
│
├── dashboard/
│   └── app.py              # Streamlit — full reasoning trace UI
│
├── tests/
│   ├── test_chunker.py
│   ├── test_scorer.py
│   ├── test_contradiction_detector.py
│   ├── test_gap_detector.py
│   └── test_pipeline.py
│
└── examples/
    ├── sample_docs/
    └── run_example.py
```

---

## Setup

**1. Clone the repository**

```bash
git clone https://github.com/SARWAGYASHAH/QueryBridge-RAG-with-active-knowledge-gap-detection.git
cd QueryBridge-RAG-with-active-knowledge-gap-detection
```

**2. Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure environment variables**

```bash
cp .env.example .env
```

Edit `.env` and fill in your API keys:

```
GROQ_API_KEY=your_groq_key_here
SERPER_API_KEY=your_serper_key_here      # optional — for live web search
TAVILY_API_KEY=your_tavily_key_here      # optional — fallback search
```

All keys are free tier. Groq: [console.groq.com](https://console.groq.com) | Serper: [serper.dev](https://serper.dev) | Tavily: [tavily.com](https://tavily.com)

---

## Usage

**Run the full pipeline from the command line:**

```python
from querybridge.pipeline import run

result = run("What is the capital of France and when was the Eiffel Tower built?")

print(result["answer"])
print(f"Confidence: {result['confidence']:.2f}")
print(f"Gap detected: {result['gap_detected']} ({result['gap_type']})")
print(f"Search used: {result['search_used']}")
```

**Start the API server:**

```bash
uvicorn api.main:app --reload
```

Endpoints:
- `POST /query` — run the full pipeline on a query
- `GET /escalations` — retrieve stored escalation events

**Launch the dashboard:**

```bash
streamlit run dashboard/app.py
```

The dashboard shows the full reasoning trace: retrieved chunks → contradiction result → confidence breakdown → gap classification → search queries → search results → final answer → escalation status.

---

## Confidence Scoring

The confidence score is computed as:

```
confidence = (similarity × 0.4) + (coverage × 0.4) + (agreement × 0.2) − contradiction_penalty
```

| Signal | Weight | Description |
|--------|--------|-------------|
| Retrieval similarity | 0.4 | Average cosine similarity of top-k chunks |
| Context coverage | 0.4 | Proportion of query aspects addressed by retrieved chunks |
| Source agreement | 0.2 | Consistency across chunk sources |
| Contradiction penalty | −0.0 to −0.3 | Applied when conflicting claims are detected |

Scores map to labels: `high` (≥ 0.7) · `medium` (0.4–0.7) · `low` (< 0.4)

---

## Evaluation

QueryBridge measures itself across 7 metrics against a benchmark dataset:

1. Retrieval Precision
2. Retrieval Recall
3. Gap Detection Accuracy
4. Search Trigger Precision
5. Search Trigger Recall
6. Answer Faithfulness
7. Contradiction Detection Accuracy

```bash
python -m querybridge.evaluator --benchmark data/benchmarks/benchmark.json
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Store | ChromaDB |
| LLM | Groq (llama3-8b-8192) / Ollama (local) |
| Web Search | Serper.dev / Tavily |
| API | FastAPI |
| Dashboard | Streamlit |
| Language | Python 3.10+ |

---

## License

MIT