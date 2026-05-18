from typing import Dict, List, Optional

from .bm25_index import BM25Index
from .config import Config
from .database import VectorDatabase
from .embedder import Embedder
from .query_expander import QueryExpander
from .reranker import Reranker


class Retriever:
    def __init__(
        self,
        config: Config,
        db: VectorDatabase,
        embedder: Embedder,
        bm25: Optional[BM25Index] = None,
        reranker: Optional[Reranker] = None,
        query_expander: Optional[QueryExpander] = None,
    ):
        self.config = config
        self.db = db
        self.embedder = embedder
        self.bm25 = bm25
        self.reranker = reranker
        self.query_expander = query_expander

    def retrieve(
        self,
        query: str,
        k: Optional[int] = None,
        *,
        pipeline: Optional[Dict] = None,
    ) -> List[Dict]:
        k = k or self.config.top_k

        # 1. Optional query expansion. The qmd-query-expansion model emits
        #    structured lex/vec/hyde lines; route them to the right index.
        vector_queries = [query]
        bm25_queries = [query]
        expansion_applied = False
        if self.config.query_expansion and self.query_expander:
            expanded = self.query_expander.expand(query)
            vector_queries = expanded.vector_queries()
            bm25_queries = expanded.bm25_queries()
            expansion_applied = True
            print(
                f"  [retriever] expanded → {len(vector_queries)} vec / "
                f"{len(bm25_queries)} bm25 queries"
            )

        # 2. Embed all vector queries
        query_embeddings = self.embedder.embed(vector_queries)

        # 3. Semantic search: union results from all queries
        semantic_results_map: Dict[str, Dict] = {}
        for emb in query_embeddings:
            results = self.db.search(emb, k=max(k * 4, 20))
            for r in results:
                cid = r["chunk_id"]
                # Keep highest similarity score across queries
                if cid not in semantic_results_map or r["similarity"] > semantic_results_map[cid]["similarity"]:
                    semantic_results_map[cid] = r

        semantic_results = list(semantic_results_map.values())
        semantic_ids = {r["chunk_id"]: i for i, r in enumerate(semantic_results)}
        semantic_map = {r["chunk_id"]: r for r in semantic_results}

        # 4. BM25 search: union results from all queries
        bm25_ids = {}
        if self.config.bm25_hybrid and self.bm25:
            bm25_results_map: Dict[str, Dict] = {}
            for q in bm25_queries:
                results = self.bm25.search(q, k=max(k * 4, 20))
                for r in results:
                    cid = r["chunk_id"]
                    if cid not in bm25_results_map or r["score"] > bm25_results_map[cid]["score"]:
                        bm25_results_map[cid] = r
            bm25_ids = {r["chunk_id"]: i for i, r in enumerate(bm25_results_map.values())}

        # 5. Reciprocal Rank Fusion
        candidate_ids = list(semantic_map.keys())
        if bm25_ids:
            candidate_ids = list(set(candidate_ids) | set(bm25_ids.keys()))

        fused = {}
        for cid in candidate_ids:
            score = 0.0
            if cid in semantic_ids:
                score += self.config.semantic_weight * (1.0 / (semantic_ids[cid] + 1))
            if cid in bm25_ids:
                score += self.config.bm25_weight * (1.0 / (bm25_ids[cid] + 1))
            fused[cid] = score

        sorted_ids = sorted(fused.keys(), key=lambda x: fused[x], reverse=True)

        # 6. Fetch BM25-only results from DB
        bm25_only_ids = [cid for cid in sorted_ids[: max(k * 4, 20)] if cid not in semantic_map]
        bm25_chunks = {c["chunk_id"]: c for c in self.db.get_by_ids(bm25_only_ids)}

        final = []
        for cid in sorted_ids[: max(k * 4, 20)]:
            if cid in semantic_map:
                final.append(semantic_map[cid])
            elif cid in bm25_chunks:
                final.append(bm25_chunks[cid])

        # 7. Reranking
        candidate_count = len(final)
        rerank_applied = bool(self.config.reranking and self.reranker)
        if rerank_applied:
            final = self.reranker.rerank(query, final, top_n=k)
        else:
            final = final[:k]

        if pipeline is not None:
            pipeline["query_expansion"] = {
                "applied": expansion_applied,
                "vector_queries": len(vector_queries),
                "bm25_queries": len(bm25_queries),
            }
            pipeline["bm25_hybrid"] = {
                "applied": bool(self.config.bm25_hybrid and self.bm25),
            }
            pipeline["reranker"] = {
                "applied": rerank_applied,
                "candidates": candidate_count,
                "returned": len(final),
            }

        return final
