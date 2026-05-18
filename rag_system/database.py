import os
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from .chunker import Chunk
from .config import Config


class VectorDatabase:
    def __init__(self, config: Config, collection_name: str = "rag_chunks"):
        self.config = config
        os.makedirs(config.chroma_db_path, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=config.chroma_db_path,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunks: List[Chunk],
        embeddings: List[List[float]],
        contexts: Dict[str, str],
    ):
        ids = [c.chunk_id for c in chunks]
        documents = [f"{contexts.get(c.chunk_id, '')}\n\n{c.content}" for c in chunks]
        metadatas = [
            {
                "chunk_id": c.chunk_id,
                "doc_id": c.doc_id,
                "original_index": c.original_index,
                "original_content": c.content,
                "contextualized_content": contexts.get(c.chunk_id, ""),
            }
            for c in chunks
        ]

        batch_size = 128
        for i in range(0, len(chunks), batch_size):
            self.collection.add(
                ids=ids[i : i + batch_size],
                embeddings=embeddings[i : i + batch_size],
                documents=documents[i : i + batch_size],
                metadatas=metadatas[i : i + batch_size],
            )

    def search(self, query_embedding: List[float], k: int = 20) -> List[Dict]:
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
        output = []
        for i in range(len(results["ids"][0])):
            output.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "document": results["documents"][0][i],
                    "distance": results["distances"][0][i],
                    "similarity": 1.0 - results["distances"][0][i],
                }
            )
        return output

    def count(self) -> int:
        return self.collection.count()

    def get_all_metadata(self) -> List[Dict]:
        """Retrieve all metadata for BM25 indexing."""
        result = self.collection.get(include=["metadatas"])
        return result["metadatas"]

    def get_by_ids(self, ids: List[str]) -> List[Dict]:
        if not ids:
            return []
        results = self.collection.get(ids=ids, include=["metadatas", "documents"])
        output = []
        for i in range(len(results["ids"])):
            output.append(
                {
                    "chunk_id": results["ids"][i],
                    "metadata": results["metadatas"][i],
                    "document": results["documents"][i],
                    "distance": None,
                    "similarity": None,
                }
            )
        return output

    def clear(self):
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
