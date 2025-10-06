# 🧠 SemanticCache: Context-Aware Caching for Multi-Turn LLM Conversations

This project implements a **context-aware semantic caching system** for multi-turn conversations powered by the **Google Gemini API**.
It reuses previous responses that are semantically similar to new queries, reducing API calls, latency, and cost — while maintaining contextual accuracy through multi-stage verification.

---

## ⚙️ 1. Overview

The system simulates a multi-turn chat session where each user query is:

1. Embedded into a semantic vector using **Gemini Embeddings API**
2. Compared against previous queries in a **vector store**
3. Validated through lexical filters
4. Either reuses a cached response (if similar) or calls Gemini again (if new)

This structure mimics real-world retrieval logic for **LLM-based memory systems**, enabling efficient reuse of semantically close content.

---

## 🧩 2. Architecture (File Structure)

```
semantic_cache/
├── cache/
│   ├── __init__.py
│   ├── embedding_cache.json           # Stores text embeddings locally
│   ├── responses_cache.json           # Stores raw LLM responses (dev use only)
│   ├── semantic_cache.py              # Core caching logic
│   └── session_manager.py             # Tracks conversation sessions
│
├── embedding/
│   ├── __init__.py
│   └── gemini_embedder.py             # Generates embeddings using Gemini API
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                     # Logs performance metrics (hit rate, latency)
│   └── threshold_sweep.py             # Sweeps similarity thresholds for tuning
│
├── logs/
│   └── session_log.json               # Stores full session activity logs
│
├── storage/
│   ├── __init__.py
│   └── vector_store.py                # Stores and retrieves embedding vectors
│
├── .env                               # Environment variables (Gemini API key)
├── .env.example                       # Example environment configuration
├── .gitignore                         # Git ignore settings
├── main.py                            # Entry point for running the system
├── README.md                          # Project documentation
└── requirements.txt                   # Python dependencies
```

---

## 🔧 3. Configuration

| Parameter        | Description                                     | Default |
| ---------------- | ----------------------------------------------- | ------- |
| `threshold`      | Maximum vector distance for semantic similarity | `0.3`   |
| `min_overlap`    | Minimum lexical overlap between queries         | `0.10`  |
| `topk`           | Number of nearest candidates to consider        | `3`     |
| `history_window` | Number of recent turns used for context         | `6`     |

Set your Gemini API key in a `.env` file:

```bash
GEMINI_API_KEY=your_actual_key_here
```

---

## 🔍 4. LLM Responses (Real + Locally Cached)

Responses are generated using **`models/gemini-2.5-flash-lite`**.
To minimize development cost, each exact query result is stored locally in:

```
cache/responses_cache.json
```

If the same query appears again, the system retrieves the stored response instead of re-calling Gemini — ensuring **zero-cost reuse during debugging**.

This local cache is for **development efficiency only**, not part of the semantic caching mechanism.

---

## 🧠 5. Embeddings with Gemini API

Embeddings are generated using **`models/embedding-001`**, via:

```python
genai.embed_content(model="models/embedding-001", content=text)
```

To avoid redundant embedding calls, each text’s embedding is cached locally in:

```
cache/embedding_cache.json
```

✅ If the same text is embedded again, the system automatically loads its vector from the local cache instead of re-calling Gemini.
When additional conversation history is included, a new contextual embedding is generated to reflect the updated dialogue context.

---

## 🚀 6. SemanticCache Logic

SemanticCache combines **two verification layers** before confirming a cache hit:

1. **Vector Distance Filter** — Check if semantic distance < `threshold`
2. **Lexical Overlap Filter** — Require minimum token overlap > `min_overlap`

If both pass, a cached answer is reused. Otherwise, a fresh Gemini API call is made.

---

## 📊 7. Metrics & Logging

All runs automatically generate:

```
logs/session_log.json
```

Each record includes:

```json
{
  "query": "Effect of global warming on corn productivity",
  "status": "HIT",
  "matched_query": "What is the impact of climate change on corn yields?",
  "distance": 0.14,
  "overlap": 0.42,
  "latency_sec": 0.62
}
```

At the end of each session:

```
📊 System Results:
{'total_calls': 8, 'total_hits': 5, 'hit_rate': 0.625, 'avg_latency': 0.48}
🗂️ Logs saved to: logs/session_log.json
```

---

## 🧪 8. Example Run

```
💬 Simulating multi-turn conversation with Semantic Cache

[API CALL #1] What is the impact of climate change on corn yields?
→ LLM Response: Climate change is having a significant impact...

[CACHE HIT #2] Effect of global warming on corn productivity
→ Return cached content: Climate change is having a signific
```
