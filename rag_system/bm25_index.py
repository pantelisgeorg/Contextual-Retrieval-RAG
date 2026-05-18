import pickle
from typing import Dict, List

from rank_bm25 import BM25Okapi


class BM25Index:
    def __init__(self):
        self.corpus: List[str] = []
        self.ids: List[str] = []
        self.index: BM25Okapi = None

    def build(self, texts: List[str], ids: List[str]):
        self.corpus = texts
        self.ids = ids
        tokenized = [t.lower().split() for t in texts]
        self.index = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 20) -> List[Dict]:
        if self.index is None:
            return []
        tokenized_query = query.lower().split()
        scores = self.index.get_scores(tokenized_query)
        top_k = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        results = []
        for idx in top_k:
            results.append(
                {
                    "chunk_id": self.ids[idx],
                    "score": float(scores[idx]),
                }
            )
        return results

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({"corpus": self.corpus, "ids": self.ids}, f)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.corpus = data["corpus"]
        obj.ids = data["ids"]
        tokenized = [t.lower().split() for t in obj.corpus]
        obj.index = BM25Okapi(tokenized)
        return obj
