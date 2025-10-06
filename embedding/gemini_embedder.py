import os
import json
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables from .env
load_dotenv()


class GeminiEmbedder:
    """
    Wrapper around Google's Gemini Embeddings API.
    Converts text into semantic vectors for similarity search.
    Includes a lightweight local cache to avoid repeated API calls
    for identical input strings.
    """

    def __init__(
        self,
        model: str = "models/embedding-001",
        api_key: str | None = None,
        cache_path: str = "cache/embedding_cache.json"
    ):
        """
        Initialize Gemini embedding model with optional local cache.

        :param model: Gemini embedding model name (default: "models/embedding-001")
        :param api_key: Optional API key (defaults to environment variable GEMINI_API_KEY)
        :param cache_path: Local JSON file to store cached embeddings
        """
        # ① 加载 API key
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY. Please set it in your environment.")

        # ② 配置 Gemini API
        genai.configure(api_key=self.api_key)
        self.model = model

        # ③ 初始化缓存文件
        self.cache_path = cache_path
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        if not os.path.exists(cache_path):
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump({}, f, indent=2, ensure_ascii=False)

        print(f"✅ Gemini Embedder initialized with model: {self.model}")

    def embed(self, text: str) -> list[float]:
        """
        Convert text into an embedding vector.
        Caches results locally to minimize redundant API calls.

        :param text: Input text to embed
        :return: list[float] - Embedding vector
        """
        # ① 输入验证
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Input text must be a non-empty string.")

        # ② 读取缓存文件
        with open(self.cache_path, "r", encoding="utf-8") as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {}

        # ③ 命中缓存 → 直接复用
        if text in cache:
            embedding = cache[text]
            print(f"[EMBED CACHE ✓] {text[:60]}...")
            return embedding

        # ④ 未命中缓存 → 调用 Gemini API
        print(f"[API EMBED → Gemini] {text[:60]}...")
        result = genai.embed_content(
            model=self.model,
            content=text,
            task_type="retrieval_document"
        )

        embedding = result["embedding"]

        # ⑤ 写入缓存文件
        cache[text] = embedding
        with open(self.cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)

        print(f"🔹 Generated embedding (dim={len(embedding)}) for text: {text[:60]}...")
        return embedding
