# 🧠 SemanticCache: Context-Aware Caching for Multi-Turn LLM Conversations

This project implements a **context-aware semantic caching system** for multi-turn conversations powered by the **Google Gemini API**.
It reuses previous responses that are semantically similar to new queries, reducing API calls, latency, and cost —
while maintaining contextual accuracy through multi-stage verification and topic drift detection.

---

## ⚙️ 1. Overview

The system simulates a multi-turn chat session where each user query is:

1. Embedded into a semantic vector using **Gemini Embeddings API**
2. Compared against previous queries in a **vector store**
3. Validated through lexical and topic-level filters
4. Either reuses a cached response (if similar) or calls Gemini again (if new)

This structure mimics real-world retrieval logic for **LLM-based memory systems**, enabling efficient reuse of semantically close content.

---

## 🧩 2. Architecture

```
User Query ─┬─> Context Builder ─┬─> Gemini Embedder ─┬─> Vector Store
            │                    │                    │
            │                    └──── Distance + Token Filter ───┐
            │                                                     │
            └────────────> Local Cache (exact match, dev only) ───┘
```

* **SessionManager** — Tracks conversation history per user
* **GeminiEmbedder** — Generates embeddings using `models/embedding-001`, with optional local embedding cache
* **VectorStore** — Stores and retrieves nearest embeddings
* **SemanticCache** — Core logic for context-aware reuse
* **Metrics** — Evaluates hit rates and latency
* **Local Response Cache (`responses_cache.json`)** — Development-time cost optimizer, *not part of semantic logic*

---

## 🔧 3. Configuration

| Parameter               | Description                                      | Default |
| ----------------------- | ------------------------------------------------ | ------- |
| `threshold`             | Maximum vector distance for semantic similarity  | `0.3`  |
| `min_overlap`           | Minimum lexical overlap between queries          | `0.10`  |
| `topk`                  | Number of nearest candidates to consider         | `3`     |
| `history_window`        | Number of recent turns used for context          | `6`     |
| `topic_drift_threshold` | Minimum noun overlap ratio to confirm same topic | `0.2`   |

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

SemanticCache combines **three verification layers** before confirming a cache hit:

1. **Vector Distance Filter** — Check if semantic distance < `threshold`
2. **Lexical Overlap Filter** — Require minimum token overlap > `min_overlap`
3. **Topic Drift Detection** — Ensure noun overlap > `topic_drift_threshold`

If all pass, a cached answer is reused. Otherwise, a fresh Gemini API call is made.

---

## 🧭 7. Topic Drift Detection (v1.1 Update)

### 🧩 Problem

Earlier versions might reuse answers across **different domains** (e.g., *corn* vs. *wheat*) because embeddings were semantically close but contextually wrong.

### ✅ Solution

Added a **topic drift detection** layer using **noun overlap analysis**:

* Extracts nouns via `nltk.pos_tag()`
* Computes overlap ratio between query nouns
* Rejects hits below `topic_drift_threshold` (default 0.2)

**Example:**

| Query                                           | Matched Query                                          | Distance | Overlap | Result                    |
| ----------------------------------------------- | ------------------------------------------------------ | -------- | ------- | ------------------------- |
| “Effect of global warming on corn productivity” | “What is the impact of climate change on corn yields?” | 0.14     | 0.42    | ✅ HIT                     |
| “How does climate change affect wheat yields?”  | “What is the impact of climate change on corn yields?” | 0.15     | 0.43    | 🚫 Rejected (topic drift) |

**Log Output Example:**

```
🚫 Topic drift detected between: 'wheat yields' ↔ 'corn yields'
```

### Installation

```bash
pip install nltk
```

---

## 📊 8. Metrics & Logging

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

## 🧪 9. Example Run

```
💬 Simulating multi-turn conversation with Semantic Cache

[API CALL #1] What is the impact of climate change on corn yields?
→ LLM Response: Climate change is having a significant impact...

[CACHE HIT #2] Effect of global warming on corn productivity
→ Return cached content: Climate change is having a significant...

🚫 Topic drift detected between: 'wheat' ↔ 'corn'
[API CALL #3] How does climate change affect wheat yields?
→ LLM Response: Wheat production is significantly impacted...
```

---

## 💡 10. Summary

| Component                 | Purpose                             |
| ------------------------- | ----------------------------------- |
| **GeminiEmbedder**        | Converts text into semantic vectors |
| **SemanticCache**         | Performs context-aware reuse        |
| **Topic Drift Detection** | Prevents cross-domain misreuse      |
| **VectorStore**           | Stores embedding records            |
| **SessionManager**        | Tracks per-user history             |
| **Metrics + Logs**        | Evaluates hit performance           |


