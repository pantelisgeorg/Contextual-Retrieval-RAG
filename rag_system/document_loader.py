import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List


DOC_IMAGES_ROOT = Path("./data/images")


@dataclass
class Document:
    doc_id: str
    path: str
    content: str


PLAIN_TEXT_EXTENSIONS = {
    ".txt", ".md", ".markdown", ".rst",
    ".py", ".js", ".ts", ".jsx", ".tsx",
    ".java", ".c", ".cpp", ".h", ".hpp", ".go", ".rs",
    ".rb", ".php", ".sh", ".bash", ".zsh",
    ".json", ".yaml", ".yml", ".toml", ".xml", ".csv",
    ".css", ".scss", ".sass",
    ".sql", ".r", ".swift", ".kt",
}

PYMUPDF_EXTENSIONS = {".pdf"}

DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".htm"}


def _supported_extensions(use_docling: bool) -> set[str]:
    exts = set(PLAIN_TEXT_EXTENSIONS)
    exts |= DOCLING_EXTENSIONS if use_docling else PYMUPDF_EXTENSIONS
    return exts


def _clean_pdf_text(text: str) -> str:
    """Remove page numbers, line numbers, and repeated headers from PDF text."""
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if re.fullmatch(r"\d{1,4}", stripped):
            continue
        if stripped in ("Knowledge Management and Organizational Learning",):
            continue
        cleaned.append(line)
    return "\n".join(cleaned)


def _extract_pdf_pymupdf(path: Path) -> str:
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError(
            "PyMuPDF is required for PDF support. Install it with: pip install pymupdf"
        )
    doc = fitz.open(str(path))
    texts = []
    for page in doc:
        raw = page.get_text()
        texts.append(_clean_pdf_text(raw))
    doc.close()
    return "\n\n".join(texts)


_docling_converter = None


def _get_docling_converter():
    global _docling_converter
    if _docling_converter is None:
        try:
            from docling.datamodel.base_models import InputFormat
            from docling.datamodel.pipeline_options import PdfPipelineOptions
            from docling.document_converter import DocumentConverter, PdfFormatOption
        except ImportError:
            raise ImportError(
                "Docling is required when USE_DOCLING=true. Install it with: pip install docling"
            )
        pdf_opts = PdfPipelineOptions()
        pdf_opts.images_scale = 2.0
        pdf_opts.generate_picture_images = True
        _docling_converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)
            }
        )
    return _docling_converter


def _doc_slug(stem: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", stem).strip("_")
    return (safe[:80] or "doc")


def _extract_with_docling(path: Path) -> str:
    from docling_core.types.doc import ImageRefMode

    converter = _get_docling_converter()
    result = converter.convert(str(path))

    slug = _doc_slug(path.stem)
    images_dir = (DOC_IMAGES_ROOT / slug).resolve()
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        md_file = Path(tmp) / "doc.md"
        result.document.save_as_markdown(
            md_file,
            artifacts_dir=images_dir,
            image_mode=ImageRefMode.REFERENCED,
        )
        md = md_file.read_text(encoding="utf-8")

    # Rewrite absolute artifact paths to URLs the FastAPI server can serve.
    md = md.replace(str(images_dir), f"/images/{slug}")
    return md


def _load_file(path: Path, use_docling: bool, root: Path | None = None) -> Document | None:
    suffix = path.suffix.lower()
    try:
        if use_docling and suffix in DOCLING_EXTENSIONS:
            content = _extract_with_docling(path)
        elif suffix in PYMUPDF_EXTENSIONS:
            content = _extract_pdf_pymupdf(path)
        else:
            content = path.read_text(encoding="utf-8", errors="ignore")
        if not content.strip():
            return None
        doc_id = str(path.name if root is None else path.relative_to(root))
        return Document(doc_id=doc_id, path=str(path), content=content)
    except Exception as e:
        print(f"Warning: could not read {path}: {e}")
        return None


def load_documents(
    sources_dir: str,
    use_docling: bool = False,
    single_file: str | None = None,
) -> List[Document]:
    docs = []
    root = Path(sources_dir)
    if not root.exists():
        raise FileNotFoundError(f"Sources directory not found: {sources_dir}")

    extensions = _supported_extensions(use_docling)

    if single_file:
        path = Path(single_file)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {single_file}")
        if path.suffix.lower() not in extensions:
            raise ValueError(f"Unsupported file type: {path.suffix}")
        doc = _load_file(path, use_docling)
        if doc:
            docs.append(doc)
        return docs

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            doc = _load_file(path, use_docling, root)
            if doc:
                docs.append(doc)

    return docs
