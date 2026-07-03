import os
from typing import Callable, Optional

from .bm25_index import BM25Index
from .chunker import chunk_documents
from .config import Config
from .contextualizer import Contextualizer
from .database import VectorDatabase
from .document_loader import load_documents
from .embedder import Embedder


EventCallback = Optional[Callable[[dict], None]]


def bm25_path(config: Config) -> str:
    return os.path.join(config.chroma_db_path, "bm25_index.pkl")


def build_bm25(db: VectorDatabase, config: Config) -> Optional[BM25Index]:
    if not config.bm25_hybrid:
        return None
    metadata = db.get_all_metadata()
    if not metadata:
        return None
    texts = [
        f"{m['original_content']} {m.get('contextualized_content', '')}" for m in metadata
    ]
    bm25 = BM25Index()
    bm25.build(texts, [m["chunk_id"] for m in metadata])
    return bm25


def load_or_build_bm25(db: VectorDatabase, config: Config) -> Optional[BM25Index]:
    if not config.bm25_hybrid:
        return None
    path = bm25_path(config)
    if os.path.exists(path):
        return BM25Index.load(path)
    bm25 = build_bm25(db, config)
    if bm25:
        bm25.save(path)
    return bm25


def run_ingest(
    config: Config,
    clear: bool = False,
    file: Optional[str] = None,
    on_event: EventCallback = None,
) -> None:
    def emit(**ev):
        if on_event:
            on_event(ev)

    if config.use_docling:
        emit(type="info", msg=f"Docling enabled — model={config.docling_model}")
        if config.docling_model == "smoldocling":
            emit(type="info", msg="SmolDocling VLM (GPU, float16) — best for scanned documents.")
        else:
            emit(type="info", msg="Standard pipeline + TableFormer v2 (structured tables from text layer).")
        if config.ocr_enabled:
            emit(type="info", msg=f"OCR enabled (Tesseract, langs={config.ocr_langs})")

    emit(type="stage", stage="loading", msg=f"Loading documents from {config.sources_dir}")
    docs = load_documents(
        config.sources_dir,
        use_docling=config.use_docling,
        single_file=file,
        ocr_enabled=config.ocr_enabled,
        ocr_langs=config.ocr_langs,
        docling_model=config.docling_model,
        vlm_max_size=config.docling_vlm_max_size,
    )
    if not docs:
        emit(type="done", chunks=0, msg="No documents found.")
        return
    emit(type="info", msg=f"Loaded {len(docs)} document(s)")

    emit(type="stage", stage="chunking", msg="Chunking documents")
    chunks = chunk_documents(docs, config.chunk_size, config.chunk_overlap)
    emit(type="info", msg=f"Created {len(chunks)} chunks")

    emit(
        type="stage",
        stage="contextualizing",
        msg=f"Contextualizing {len(chunks)} chunks ({config.openai_model})",
    )
    contextualizer = Contextualizer(config)
    contexts = contextualizer.contextualize_chunks(chunks, docs, parallel_threads=3)

    emit(type="stage", stage="embedding", msg=f"Embedding ({config.ollama_embed_model})")
    embedder = Embedder(config)
    texts_to_embed = [
        f"{contexts.get(c.chunk_id, '')}\n\n{c.content}" for c in chunks
    ]
    embeddings = embedder.embed(texts_to_embed)

    emit(type="stage", stage="storing", msg="Storing in ChromaDB")
    db = VectorDatabase(config)
    if clear or not file:
        db.clear()
    db.add_chunks(chunks, embeddings, contexts)
    total = db.count()

    if config.bm25_hybrid:
        emit(type="stage", stage="bm25", msg="Building BM25 index")
        bm25 = build_bm25(db, config)
        if bm25:
            bm25.save(bm25_path(config))

    emit(type="done", chunks=total, msg="Ingestion complete!")
