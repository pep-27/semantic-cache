from cache.session_manager import SessionManager
from cache.semantic_cache import SemanticCache
from embedding.gemini_embedder import GeminiEmbedder
from storage.vector_store import VectorStore
import matplotlib.pyplot as plt


def run_experiment(thresholds):
    embedder = GeminiEmbedder()
    store = VectorStore()
    session = SessionManager()

    # construct a minimum knowledge base 
    kb = [
        ("What is the impact of climate change on corn yields?",
         "Climate change reduces corn productivity in many regions."),
        ("How does global warming affect maize productivity?",
         "Global warming affects maize yields similarly to corn."),
        ("How does climate change affect wheat yields?",
         "Wheat yields are also impacted, but with regional variability."),
        ("What is GDP?",
         "GDP is the total monetary value of goods and services produced.")
    ]

    # write to knowledge base
    for q, r in kb:
        store.insert(embedder.embed(q), q, r)

    # a set of test queries
    tests = [
        ("Effect of climate change on corn production", True),
        ("Wheat productivity under global warming", True),
        ("GDP meaning", True),
        ("How to bake a cake", False),
    ]

    for th in thresholds:
        cache = SemanticCache(store, embedder, threshold=th, history_window=6, topk=3, min_overlap=0.10)
        hits = 0
        total = len(tests)
        for q, expect_hit in tests:
            res = cache.get(session_id="demo", query=q, history=session.get_history("demo"))
            if res["hit"]:
                hits += 1
        print(f"threshold={th:.2f}  hit_rate={hits/total:.2f}  hits={hits}/{total}")

if __name__ == "__main__":
    run_experiment([0.20, 0.25, 0.30, 0.35, 0.40])

