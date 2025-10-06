# =============================================================================
# SemanticCache (context-aware semantic caching)
#
# Simulates a multi-turn chat session with a semantic cache layer.
# Reuses semantically similar responses to reduce Gemini API calls, latency, and cost.
# Also records each step to logs/session_log.json for later inspection.
# =============================================================================

import os
import json
import time
from google import generativeai as genai

from cache.session_manager import SessionManager
from cache.semantic_cache import SemanticCache
from embedding.gemini_embedder import GeminiEmbedder
from storage.vector_store import VectorStore
from evaluation.metrics import Metrics

# ========= 🔧 Configuration =========
CACHE_PATH = "cache/responses_cache.json"
LOG_PATH = "logs/session_log.json"
MODEL_NAME = "models/gemini-2.5-flash-lite"
# NOTE: CACHE_PATH is for development-time local caching (not part of semantic cache)

# ========= 💬 Gemini API Wrapper =========
def cached_gemini_api(query: str) -> str:
    """
    Lightweight Gemini API wrapper with a local JSON cache
    (to avoid repeated API calls during development).
    """
    # Initialize local cache file
    if not os.path.isfile(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)

    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        local_cache = json.load(f)

    # Reuse previous response if available
    if query in local_cache:
        print(f"[LOCAL DEV CACHE ✓] {query[:50]}...")
        return local_cache[query]

    # Otherwise call Gemini API
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(query, generation_config={"temperature": 0})
        text = response.text.strip()
    except Exception as e:
        print(f"⚠️ Gemini API call failed: {e}")
        text = "Temporary error. Please try again later."

    # Save to local cache for next time reuse
    local_cache[query] = text
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(local_cache, f, indent=2, ensure_ascii=False)

    print(f"[API CALL → Gemini] {query[:50]}...")
    return text


# ========= 🧾 Log Helper =========
def append_log(entry: dict):
    """
    Append a log record to logs/session_log.json.
    Each record includes query, hit/miss, distance, overlap, and short response.
    """
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logs = []

    # If log file exists, load old logs first
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append(entry)

    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


# ========= 🚀 Main Flow =========
def main():
    print("💬 Simulating multi-turn conversation with Semantic Cache\n")

    # 🧹 Cleared previous logs (if exist)
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2, ensure_ascii=False)
        print(f"🧹 Cleared previous logs: {LOG_PATH}\n")

    session = SessionManager()
    embedder = GeminiEmbedder()
    vector_store = VectorStore()
    cache = SemanticCache(
        embedder=embedder,
        vector_store=vector_store,
        threshold=0.3,
        topk=3
    )
    metrics = Metrics()

    session_id = "user_1"

    queries = [
        "What is the impact of climate change on corn yields?",
        "Effect of global warming on corn productivity",
        "What is GDP?",
        "Tell me about wheat production in global warming",
        "What is the impact of climate change on wheat production?",
        "How does global warming affect wheat production?",
        "Write a poem for me.",
        "Tell me a famous poem written by Shakespeare."
    ]

    for i, q in enumerate(queries, 1):
        start = time.time()
        history = session.get_history(session_id)
        res = cache.get(session_id, q, history)

        if res["hit"]:
            print(f"[CACHE HIT #{i}] {q}")
            preview = "\n".join(res["response"].splitlines()[:3])
            print(f"→ Return cached content:\n{preview}\n")
            entry = {
                "query": q,
                "status": "HIT",
                "matched_query": res["matched_query"],
                "distance": res["distance"],
                "overlap": res["overlap"],
                "response_preview": preview
            }
        else:
            # Real Gemini call
            answer = cached_gemini_api(q)
            preview = "\n".join(answer.splitlines()[:3])
            print(f"[API CALL #{i}] {q}")
            print(f"→ LLM Response:\n{preview}\n")

            # Update cache and session
            cache.add(session_id, q, answer, history)
            session.add_message(session_id, "user", q)
            session.add_message(session_id, "ai", answer)

            entry = {
                "query": q,
                "status": "MISS",
                "response_preview": preview
            }

        latency = time.time() - start
        metrics.record_call(res["hit"], latency)
        entry["latency_sec"] = round(latency, 3)
        append_log(entry)

    print("📊 System Results:")
    print(metrics.summary())
    print(f"🗂️ Logs saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
