import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .answerer import Answerer
from .config import Config
from .database import VectorDatabase
from .document_loader import DOC_IMAGES_ROOT, _supported_extensions
from .embedder import Embedder
from .ingest import load_or_build_bm25, run_ingest
from .query_expander import QueryExpander
from .reranker import Reranker
from .retriever import Retriever


STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: Optional[int] = None


class ResultItem(BaseModel):
    chunk_id: str
    doc_id: str
    score: float
    passage: str
    context: str


class CitationItem(BaseModel):
    n: int
    quote: str


class QueryResponse(BaseModel):
    query: str
    count: int
    results: list[ResultItem]
    pipeline: dict
    answer: Optional[str] = None
    citations: list[CitationItem] = []
    answer_error: Optional[str] = None


class IngestRequest(BaseModel):
    clear: bool = False
    file: Optional[str] = None


def _resolve_source_file(config: Config, filename: str) -> str:
    """Resolve a basename or relative path inside sources/. Rejects traversal."""
    root = Path(config.sources_dir).resolve()
    target = (root / filename).resolve()
    if root != target and root not in target.parents:
        raise HTTPException(status_code=400, detail="file must be inside sources/")
    if not target.is_file():
        raise HTTPException(status_code=404, detail=f"file not found: {filename}")
    return str(target)


def _build_retriever(config: Config) -> Retriever:
    db = VectorDatabase(config)
    if db.count() == 0:
        raise RuntimeError("Database is empty. Run 'rag-system ingest' first.")
    embedder = Embedder(config)
    bm25 = load_or_build_bm25(db, config)
    reranker = Reranker(config) if config.reranking else None
    query_expander = QueryExpander(config) if config.query_expansion else None
    return Retriever(config, db, embedder, bm25, reranker, query_expander)


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = Config.from_env()
    app.state.config = config
    app.state.ingest_lock = asyncio.Lock()
    try:
        app.state.retriever = _build_retriever(config)
    except RuntimeError:
        app.state.retriever = None
    app.state.answerer = Answerer(config) if config.answer_synthesis else None
    yield


app = FastAPI(title="Contextual Retrieval RAG", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
DOC_IMAGES_ROOT.mkdir(parents=True, exist_ok=True)
app.mount("/images", StaticFiles(directory=DOC_IMAGES_ROOT), name="images")


@app.get("/")
async def root() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
async def health() -> dict:
    config: Config = app.state.config
    db = VectorDatabase(config)
    return {"chunks": db.count()}


@app.get("/api/models")
async def models() -> dict:
    config: Config = app.state.config
    return {
        "doc_loader": "docling" if config.use_docling else "pymupdf",
        "contextualizer": config.openai_model,
        "embedder": config.ollama_embed_model,
        "reranker": config.reranker_model if config.reranking else None,
        "query_expander": config.ollama_query_expansion_model if config.query_expansion else None,
        "bm25_hybrid": config.bm25_hybrid,
        "answerer": config.answer_model if config.answer_synthesis else None,
    }


@app.post("/api/query", response_model=QueryResponse)
async def query(req: QueryRequest) -> QueryResponse:
    if app.state.retriever is None:
        raise HTTPException(status_code=409, detail="Database is empty. Ingest documents first.")
    retriever: Retriever = app.state.retriever
    config: Config = app.state.config
    k = req.top_k if req.top_k is not None else config.top_k
    if k <= 0:
        raise HTTPException(status_code=400, detail="top_k must be positive")

    pipeline: dict = {}
    raw = retriever.retrieve(req.query, k=k, pipeline=pipeline)
    items = []
    for r in raw:
        meta = r["metadata"]
        score = r.get("rerank_score", r.get("similarity", 0.0))
        items.append(
            ResultItem(
                chunk_id=r["chunk_id"],
                doc_id=meta["doc_id"],
                score=float(score),
                passage=meta["original_content"],
                context=meta.get("contextualized_content", ""),
            )
        )

    answer: Optional[str] = None
    citations: list[CitationItem] = []
    answer_error: Optional[str] = None
    answerer: Optional[Answerer] = app.state.answerer
    if answerer is not None and raw:
        try:
            result = await asyncio.to_thread(answerer.synthesize, req.query, raw)
            answer = result.answer
            citations = [CitationItem(n=c.n, quote=c.quote) for c in result.citations]
        except Exception as e:
            answer_error = str(e)

    return QueryResponse(
        query=req.query,
        count=len(items),
        results=items,
        pipeline=pipeline,
        answer=answer,
        citations=citations,
        answer_error=answer_error,
    )


@app.get("/api/sources")
async def sources() -> dict:
    config: Config = app.state.config
    root = Path(config.sources_dir)
    if not root.is_dir():
        return {"files": []}
    exts = _supported_extensions(config.use_docling)
    files = sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )
    return {"files": files}


@app.post("/api/ingest")
async def ingest(req: IngestRequest):
    lock: asyncio.Lock = app.state.ingest_lock
    if lock.locked():
        raise HTTPException(status_code=409, detail="Another ingest is in progress")

    config: Config = app.state.config
    file_arg = _resolve_source_file(config, req.file) if req.file else None

    queue: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def on_event(ev: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, ev)

    async def runner() -> None:
        async with lock:
            try:
                await asyncio.to_thread(run_ingest, config, req.clear, file_arg, on_event)
            except Exception as e:
                queue.put_nowait({"type": "error", "msg": str(e)})
            finally:
                try:
                    db = VectorDatabase(config)
                    if db.count() > 0:
                        app.state.retriever = _build_retriever(config)
                    else:
                        app.state.retriever = None
                except Exception as e:
                    queue.put_nowait({"type": "warning", "msg": f"Retriever reload failed: {e}"})
                queue.put_nowait(None)

    asyncio.create_task(runner())

    async def stream():
        while True:
            ev = await queue.get()
            if ev is None:
                return
            yield json.dumps(ev) + "\n"

    return StreamingResponse(stream(), media_type="application/x-ndjson")
