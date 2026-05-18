from typing import Dict, List

from .config import Config


class Reranker:
    def __init__(self, config: Config):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as e:
            raise ImportError(
                "sentence-transformers is required for reranking. "
                "Install it with: pip install sentence-transformers"
            ) from e
        print(f"Loading reranker from Hugging Face: {config.reranker_model} ...")
        try:
            self.model = CrossEncoder(config.reranker_model, device="cpu")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load reranker '{config.reranker_model}'. "
                "Ensure you have internet access for the first download, "
                "or set the model path to a local directory."
            ) from e

    def rerank(self, query: str, chunks: List[Dict], top_n: int = 10) -> List[Dict]:
        if not chunks:
            return []
        documents = [
            f"{c['metadata']['original_content']}\n\nContext: {c['metadata']['contextualized_content']}"
            for c in chunks
        ]
        scores = self.model.predict([[query, doc] for doc in documents])
        scored = [(chunks[i], float(scores[i])) for i in range(len(chunks))]
        scored.sort(key=lambda x: x[1], reverse=True)
        return [
            {
                "metadata": item[0]["metadata"],
                "chunk_id": item[0]["chunk_id"],
                "rerank_score": item[1],
            }
            for item in scored[:top_n]
        ]
