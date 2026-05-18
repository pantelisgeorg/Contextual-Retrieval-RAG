from typing import List

import ollama
from tqdm import tqdm

from .config import Config


class Embedder:
    def __init__(self, config: Config):
        self.client = ollama.Client(host=config.ollama_host)
        self.model = config.ollama_embed_model

    def embed(self, texts: List[str], batch_size: int = 32) -> List[List[float]]:
        results = []
        for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
            batch = texts[i : i + batch_size]
            resp = self.client.embed(model=self.model, input=batch)
            results.extend(resp.embeddings)
        return results
