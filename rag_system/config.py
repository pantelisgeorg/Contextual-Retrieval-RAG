import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    openai_api_key: str
    openai_base_url: str
    openai_model: str
    ollama_host: str
    ollama_embed_model: str
    ollama_query_expansion_model: str
    reranker_model: str
    chroma_db_path: str
    sources_dir: str
    chunk_size: int
    chunk_overlap: int
    query_expansion: bool
    reranking: bool
    bm25_hybrid: bool
    semantic_weight: float
    bm25_weight: float
    top_k: int
    use_docling: bool
    ocr_enabled: bool
    ocr_langs: str
    answer_synthesis: bool
    answer_model: str
    answer_max_tokens: int

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
            ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text-v2-moe"),
            ollama_query_expansion_model=os.getenv(
                "OLLAMA_QUERY_EXPANSION_MODEL", "hf.co/tobil/qmd-query-expansion-1.7b-GGUF:Q4_K_M"
            ),
            reranker_model=os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
            chroma_db_path=os.getenv("CHROMA_DB_PATH", "./data/chromadb"),
            sources_dir=os.getenv("SOURCES_DIR", "./sources"),
            chunk_size=int(os.getenv("CHUNK_SIZE", "2800")),
            chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "400")),
            query_expansion=os.getenv("QUERY_EXPANSION", "false").lower() == "true",
            reranking=os.getenv("RERANKING", "true").lower() == "true",
            bm25_hybrid=os.getenv("BM25_HYBRID", "true").lower() == "true",
            semantic_weight=float(os.getenv("SEMANTIC_WEIGHT", "0.8")),
            bm25_weight=float(os.getenv("BM25_WEIGHT", "0.2")),
            top_k=int(os.getenv("TOP_K", "10")),
            use_docling=os.getenv("USE_DOCLING", "false").lower() == "true",
            ocr_enabled=os.getenv("OCR_ENABLED", "false").lower() == "true",
            ocr_langs=os.getenv("OCR_LANGS", "ell+eng"),
            answer_synthesis=os.getenv("ANSWER_SYNTHESIS", "true").lower() == "true",
            answer_model=os.getenv("ANSWER_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini")),
            answer_max_tokens=int(os.getenv("ANSWER_MAX_TOKENS", "800")),
        )
