# =============================================================================
# SemanticCache (Context-Aware Semantic Caching)
#
# Reuses semantically similar responses while preventing topic drift errors.
# Now includes automatic topic conflict detection via NLTK noun overlap check.
# =============================================================================

import re
import nltk
from collections import Counter
from embedding.gemini_embedder import GeminiEmbedder
from storage.vector_store import VectorStore


# Auto-download missing NLTK resources if not already present
for pkg in ["punkt", "punkt_tab", "averaged_perceptron_tagger_eng"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)


# --- Stopwords for overlap filtering ---
STOPWORDS = {
    "the","is","are","a","an","of","on","in","at","to","for","and","or","by","with","about",
    "what","how","why","does","do","did","can","could","would","should","tell","me","please",
    "this","that","it","its","from","into","as"
}

# --- Ensure NLTK dependencies (only runs once) ---
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)
try:
    nltk.data.find("taggers/averaged_perceptron_tagger_eng")
except LookupError:
    nltk.download("averaged_perceptron_tagger_eng", quiet=True)


class SemanticCache:
    def __init__(
        self,
        embedder: GeminiEmbedder,
        vector_store: VectorStore,
        threshold: float = 0.3,
        history_window: int = 6,
        topk: int = 3,
        min_overlap: float = 0.10,
        context_min_overlap: float = 0.08,
        topic_drift_threshold: float = 0.2,
    ):
        """
        Context-aware semantic cache controller.
        Includes a topic drift filter based on noun overlap.
        """
        self.vector_store = vector_store
        self.embedder = embedder
        self.threshold = threshold
        self.history_window = history_window
        self.topk = topk
        self.min_overlap = min_overlap
        self.context_min_overlap = context_min_overlap
        self.topic_drift_threshold = topic_drift_threshold

        print(
            f"✅ SemanticCache initialized | "
            f"threshold={threshold}, topk={topk}, min_overlap={min_overlap}, history_window={history_window}"
        )

    # ============================================================
    # 🔍 Query Retrieval
    # ============================================================
    def get(self, session_id: str, query: str, history: list[dict]):
        """
        Main retrieval logic:
        1. Combine query + recent history for context embedding
        2. Search top-k in vector store
        3. Filter by distance and lexical overlap
        4. Reject if topic drift detected
        """
        context_query = self._combine_context(query, history)
        query_embedding = self.embedder.embed(context_query)

        candidates = self.vector_store.search_topk(query_embedding, k=self.topk)
        if not candidates:
            return {"hit": False}

        # --- Step 1: Distance filter ---
        close_ones = [c for c in candidates if c["distance"] < self.threshold]
        if not close_ones:
            return {"hit": False}

        # --- Step 2: Lexical overlap filter ---
        best = max(close_ones, key=lambda c: self._token_overlap(query, c["query"]))
        best_overlap = self._token_overlap(query, best["query"])
        if best_overlap < self.min_overlap:
            return {"hit": False}

        # --- Step 3: Topic conflict detection ---
        if self._topic_conflict(query, best["query"]):
            print(f"🚫 Topic drift detected between: '{query}'  ↔  '{best['query']}'")
            return {"hit": False}

        # --- Pass all filters → return hit ---
        return {
            "hit": True,
            "matched_query": best["query"],
            "response": best["response"],
            "distance": best["distance"],
            "overlap": best_overlap,
        }

    # ============================================================
    # 🧠 Add to cache
    # ============================================================
    def add(self, session_id: str, query: str, response: str, history: list[dict]):
        """
        Store new (query + response) pair in vector store.
        Embedding is generated using contextualized query.
        """
        context_query = self._combine_context(query, history)
        query_embedding = self.embedder.embed(context_query)
        self.vector_store.insert(query_embedding, query, response)

    # ============================================================
    # 🧩 Helper: Combine user history into one context string
    # ============================================================
    def _combine_context(self, query: str, history: list[dict]) -> str:
        """Merge last N history messages into a contextualized query."""
        if not history:
            return f"USER: {query}"
        window = history[-self.history_window:]
        history_text = "\n".join(f"{m['role'].upper()}: {m['content']}" for m in window)
        hist_overlap = self._token_overlap(query, history_text)
        if hist_overlap < self.context_min_overlap:
            return f"USER: {query}"
        return f"{history_text}\nUSER: {query}"

    # ============================================================
    # 🧮 Helper: Token overlap (Jaccard)
    # ============================================================
    def _token_overlap(self, a: str, b: str) -> float:
        """Compute Jaccard similarity between token sets."""
        ta, tb = self._tokenize(a), self._tokenize(b)
        if not ta or not tb:
            return 0.0
        inter, union = len(ta & tb), len(ta | tb)
        return inter / union if union else 0.0

    def _tokenize(self, s: str) -> set[str]:
        """Lowercase, alphanumeric split, remove stopwords."""
        tokens = re.split(r"[^a-z0-9]+", s.lower())
        return {t for t in tokens if t and t not in STOPWORDS}

    # ============================================================
    # 🧭 Topic Conflict Detection
    # ============================================================
    def _topic_conflict(self, q1: str, q2: str) -> bool:
        """
        Automatically detect topic mismatch between two queries.
        Uses POS tagging to extract main nouns, then measures overlap.
        If noun overlap ratio < threshold → consider as topic drift.
        """
        nouns1 = self._extract_nouns(q1)
        nouns2 = self._extract_nouns(q2)
        if not nouns1 or not nouns2:
            return False
        overlap = len(nouns1 & nouns2) / len(nouns1 | nouns2)
        return overlap < self.topic_drift_threshold

    def _extract_nouns(self, text: str) -> set[str]:
        """Extract nouns (NN, NNP, etc.) using NLTK POS tagging."""
        words = nltk.word_tokenize(text.lower())
        tagged = nltk.pos_tag(words)
        nouns = {w for w, pos in tagged if pos.startswith("NN") and len(w) > 2}
        return nouns
