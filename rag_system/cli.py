import argparse
import sys

from dotenv import load_dotenv

from .config import Config
from .database import VectorDatabase
from .embedder import Embedder
from .ingest import load_or_build_bm25, run_ingest
from .query_expander import QueryExpander
from .reranker import Reranker
from .retriever import Retriever


def _format_passage(text: str, max_chars: int | None = None) -> str:
    """Normalize whitespace so PDFs with hard line breaks don't look broken.

    Chunks are already bounded at ingest time by CHUNK_SIZE, so we display the
    full chunk by default. Pass `max_chars` only if you want a preview.
    """
    text = text.strip()
    if max_chars is not None and len(text) > max_chars:
        cut = text[:max_chars]
        for sep in (". ", "! ", "? ", " "):
            idx = cut.rfind(sep)
            if idx > max_chars * 0.6:
                cut = cut[: idx + 1]
                break
        text = cut.rstrip() + "…"
    return " ".join(text.split())


def _print_results(query: str, results: list[dict]) -> None:
    if not results:
        print("No relevant passages found.")
        return
    print(f"\nQuery: {query}\n")
    print(f"Found {len(results)} relevant passage(s):\n")
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        score = r.get("rerank_score", r.get("similarity", 0))
        passage = _format_passage(meta["original_content"])
        print(f"[{i}] {meta['doc_id']}  (relevance {score:.2f})")
        print(f"    {passage}\n")


def cmd_ingest(args):
    config = Config.from_env()

    def on_event(ev: dict) -> None:
        if ev["type"] in ("info", "stage", "done") and "msg" in ev:
            print(ev["msg"])

    run_ingest(config, clear=args.clear, file=args.file, on_event=on_event)


def cmd_query(args):
    config = Config.from_env()
    db = VectorDatabase(config)
    if db.count() == 0:
        print("Database is empty. Run 'ingest' first.")
        return

    embedder = Embedder(config)
    bm25 = load_or_build_bm25(db, config)
    reranker = Reranker(config) if config.reranking else None
    query_expander = QueryExpander(config) if config.query_expansion else None

    retriever = Retriever(config, db, embedder, bm25, reranker, query_expander)
    k = config.top_k if args.top_k is None else args.top_k
    results = retriever.retrieve(args.query, k=k)
    _print_results(args.query, results)


def cmd_serve(args):
    import uvicorn
    print(f"RAG server: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    uvicorn.run("rag_system.web:app", host=args.host, port=args.port, log_level="warning")


def cmd_chat(args):
    config = Config.from_env()
    db = VectorDatabase(config)
    if db.count() == 0:
        print("Database is empty. Run 'ingest' first.")
        return

    embedder = Embedder(config)
    bm25 = load_or_build_bm25(db, config)
    reranker = Reranker(config) if config.reranking else None
    query_expander = QueryExpander(config) if config.query_expansion else None
    retriever = Retriever(config, db, embedder, bm25, reranker, query_expander)

    print("RAG Chat (type 'exit' to quit)\n")
    while True:
        try:
            query = input("> ")
        except (EOFError, KeyboardInterrupt):
            break
        if query.strip().lower() in ("exit", "quit"):
            break
        if not query.strip():
            continue

        results = retriever.retrieve(query, k=config.top_k)
        _print_results(query, results)


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(prog="rag_system", description="Contextual Retrieval RAG CLI")
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Ingest documents from sources/")
    ingest_parser.add_argument("--file", type=str, default=None, help="Ingest a single file instead of the whole sources/ folder")
    ingest_parser.add_argument("--clear", action="store_true", help="Clear the database before ingesting")
    ingest_parser.set_defaults(func=cmd_ingest)

    query_parser = subparsers.add_parser("query", help="Run a single query")
    query_parser.add_argument("query", type=str, help="Query string")
    query_parser.add_argument("--top-k", type=int, default=None, help="Number of results")
    query_parser.set_defaults(func=cmd_query)

    chat_parser = subparsers.add_parser("chat", help="Interactive chat mode")
    chat_parser.set_defaults(func=cmd_chat)

    serve_parser = subparsers.add_parser("serve", help="Run the web UI (FastAPI)")
    serve_parser.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default 127.0.0.1)")
    serve_parser.add_argument("--port", type=int, default=8000, help="Bind port (default 8000)")
    serve_parser.set_defaults(func=cmd_serve)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)
    args.func(args)


if __name__ == "__main__":
    main()
