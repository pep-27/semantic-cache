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
import pandas as pd
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

# Official Gemini pricing (as of 2025-10)
# https://ai.google.dev/pricing
# gemini-2.5-flash-lite : $0.10 / 1M input tokens, $0.40 / 1M output tokens
PRICE_PER_MILLION_INPUT = 0.10
PRICE_PER_MILLION_OUTPUT = 0.40


# ========= 💬 Gemini API Wrapper =========
def cached_gemini_api(query: str) -> (str, int, int):
    """
    Lightweight Gemini API wrapper with a local JSON cache
    (to avoid repeated API calls during development).
    Automatically extracts token usage if available from API response.

    Returns:
        text (str): Model output text
        input_tokens (int): Token count for prompt
        output_tokens (int): Token count for completion
    """
    # Initialize local cache file
    if not os.path.isfile(CACHE_PATH):
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump({}, f, indent=2, ensure_ascii=False)

    # Read from local cache
    with open(CACHE_PATH, "r", encoding="utf-8") as f:
        local_cache = json.load(f)

    # Reuse previous response if available
    if query in local_cache:
        print(f"[LOCAL DEV CACHE ✓] {query[:50]}...")
        return local_cache[query], 0, 0

    # Otherwise call Gemini API
    try:
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(query, generation_config={"temperature": 0})
        text = response.text.strip()

        # --- Extract token usage (if supported by SDK) ---
        usage = getattr(response, "usage_metadata", None)
        if usage:
            input_tokens = usage.get("prompt_token_count", 0)
            output_tokens = usage.get("candidates_token_count", 0)
        else:
            input_tokens = 0
            output_tokens = 0

    except Exception as e:
        print(f"⚠️ Gemini API call failed: {e}")
        text = "Temporary error. Please try again later."
        input_tokens = 0
        output_tokens = 0

    # Save response for local reuse
    local_cache[query] = text
    with open(CACHE_PATH, "w", encoding="utf-8") as f:
        json.dump(local_cache, f, indent=2, ensure_ascii=False)

    print(f"[API CALL → Gemini] {query[:50]}...")
    return text, input_tokens, output_tokens


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
    results = []

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
                "response_preview": preview,
                "input_tokens": 0,
                "output_tokens": 0
            }
        else:
            # Real Gemini call
            answer, inp_toks, out_toks = cached_gemini_api(q)
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
                "response_preview": preview,
                "input_tokens": inp_toks,
                "output_tokens": out_toks
            }

        latency = time.time() - start
        metrics.record_call(res["hit"], latency)
        entry["latency_sec"] = round(latency, 3)
        append_log(entry)
        results.append(entry)

    # ========= 📊 Summary & Cost Estimation =========
    df = pd.DataFrame(results)

    hit_rate = (df["status"] == "HIT").mean()
    avg_latency = df["latency_sec"].mean()
    miss_latency = df[df["status"] == "MISS"]["latency_sec"].mean()
    hit_latency = df[df["status"] == "HIT"]["latency_sec"].mean()

    print("\n📈 Summary Statistics")
    print(f"Cache Hit Rate: {hit_rate:.2%}")
    print(f"Avg Latency: {avg_latency:.3f} s")
    print(f"Hit Latency: {hit_latency:.3f} s, Miss Latency: {miss_latency:.3f} s")

    # Calculate API cost using official Gemini pricing
    df["cost_usd"] = (
        df["input_tokens"] / 1_000_000 * PRICE_PER_MILLION_INPUT +
        df["output_tokens"] / 1_000_000 * PRICE_PER_MILLION_OUTPUT
    )

    total_cost = df["cost_usd"].sum()
    cost_saved = df[df["status"] == "HIT"]["cost_usd"].sum()
    llm_calls_saved = (df["status"] == "HIT").sum()

    print(f"LLM Calls Saved: {llm_calls_saved}")
    print(f"Estimated Cost Saved: ${cost_saved:.6f}")
    print(f"Total Estimated API Cost (MISS only): ${total_cost:.6f}")

    # Save summary with formatted decimal precision
    summary_entry = {
        "summary": {
            "cache_hit_rate": round(hit_rate, 4),
            "avg_latency_s": round(avg_latency, 3),
            "hit_latency_s": round(hit_latency, 3),
            "miss_latency_s": round(miss_latency, 3),
            "llm_calls_saved": int(llm_calls_saved),
            "estimated_cost_saved_usd": round(cost_saved, 6),
            "total_estimated_cost_usd": round(total_cost, 6)
        }
    }

    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(",\n")
        json.dump(summary_entry, f, indent=2, ensure_ascii=False)

    print(f"\n🗂️ Detailed logs + summary saved to: {LOG_PATH}")


if __name__ == "__main__":
    main()
