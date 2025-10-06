import re
from embedding.gemini_embedder import GeminiEmbedder
from storage.vector_store import VectorStore

# stopwords (for removing meaningless overlap)
STOPWORDS = {
    "the","is","are","a","an","of","on","in","at","to","for","and","or","by","with","about",
    "what","how","why","does","do","did","can","could","would","should","tell","me","please",
    "this","that","it","its","from","into","as"
}


class SemanticCache:
    """
    Context-aware semantic cache for conversational AI.

    - Combines recent dialog turns into a context-aware embedding.
    - Searches top-k nearest vectors from vector store.
    - Uses two-step filtering (vector distance + lexical overlap) to confirm cache hits.

    Parameters:
        embedder: GeminiEmbedder instance (for generating embeddings)
        vector_store: VectorStore instance (for storing and searching vectors)
        threshold: maximum distance to be considered "semantically similar"
        topk: number of top candidates to check
        min_overlap: minimum lexical overlap ratio for second-stage validation
        history_window: number of most recent messages to include in context
        context_min_overlap: if query vs. history overlap is below this, treat as topic drift
    """

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: GeminiEmbedder,
        threshold: float = 0.3,
        topk: int = 3,
        min_overlap: float = 0.10,
        history_window: int = 6,
        context_min_overlap: float = 0.08,
    ):
        # basic attributes
        self.vector_store = vector_store
        self.embedder = embedder
        self.threshold = threshold
        self.topk = topk
        self.min_overlap = min_overlap
        self.history_window = history_window
        self.context_min_overlap = context_min_overlap

        # initialize prompt (for debugging or presentation)
        print(
            f"✅ SemanticCache initialized | "
            f"threshold={self.threshold}, topk={self.topk}, "
            f"min_overlap={self.min_overlap}, history_window={self.history_window}"
        )

    # ================== 🔍 Cache Lookup ==================
    def get(self, session_id, query, history):
        """
        Retrieve from semantic cache if a semantically similar query exists.
        Steps:
            1) Build context-aware query embedding
            2) Retrieve top-k candidates
            3) Filter by distance threshold
            4) Confirm hit with lexical overlap
        """
        # combine context (with User label)
        context_query = self._combine_context(query, history)
        query_embedding = self.embedder.embed(context_query)

        # Step 1: get top-k candidates from vector store
        candidates = self.vector_store.search_topk(query_embedding, k=self.topk)
        if not candidates:
            return {"hit": False}

        # Step 2: distance initial screening
        close_ones = [c for c in candidates if c["distance"] < self.threshold]
        if not close_ones:
            return {"hit": False}

        # Step 3: lexical overlap discrimination
        best = max(close_ones, key=lambda c: self._token_overlap(query, c["query"]))
        best_overlap = self._token_overlap(query, best["query"])
        if best_overlap < self.min_overlap:
            return {"hit": False}

        return {
            "hit": True,
            "matched_query": best["query"],
            "response": best["response"],
            "distance": best["distance"],
            "overlap": best_overlap,
        }

    # ================== 💾 Cache Insert ==================
    def add(self, session_id, query, response, history):
        """
        Insert new context-aware embedding into vector store.
        Each entry stores:
            - combined context query (for embedding)
            - raw user query (for readability)
            - response text (for cache reuse)
        """
        context_query = self._combine_context(query, history)
        query_embedding = self.embedder.embed(context_query)
        self.vector_store.insert(query_embedding, query, response)

    # ================== 🧠 Context Combination ==================
    def _combine_context(self, query, history):
        """
        Combine the last N turns into a context-aware input.
        Includes role labels (USER/AI) for better embedding consistency.
        """
        if not history:
            return f"USER: {query}"

        # get the last N history
        window = history[-self.history_window:]
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in window)

        # check if there is topic drift (low overlap with history)
        hist_overlap = self._token_overlap(query, history_text)
        if hist_overlap < self.context_min_overlap:
            return f"USER: {query}"

        return f"{history_text}\nUSER: {query}"

    # ================== 🧮 Lexical Overlap ==================
    def _token_overlap(self, a: str, b: str) -> float:
        ta = self._tokenize(a)
        tb = self._tokenize(b)
        if not ta or not tb:
            return 0.0
        inter = len(ta & tb)
        union = len(ta | tb)
        return inter / union if union else 0.0

    def _tokenize(self, s: str) -> set:
        tokens = re.split(r"[^a-z0-9]+", s.lower())
        return {t for t in tokens if t and t not in STOPWORDS}
