import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List


DOC_IMAGES_ROOT = Path("./data/images")
DOC_MARKDOWN_ROOT = Path("./data/markdown")
DOC_EXPORT_ROOT = Path("./data/export")


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
_docling_converter_key: tuple | None = None


def _build_default_converter(ocr_enabled: bool = False, ocr_langs: str = "ell+eng"):
    """Standard Docling layout pipeline with the v2 TableFormer.

    The v2 table-structure model keeps cell text intact (the v1 TableFormer
    split cells mid-phrase). Born-digital PDFs get accurate text straight from
    the embedded text layer.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        TableStructureV2Options,
        TesseractCliOcrOptions,
    )

    pdf_opts = PdfPipelineOptions()
    pdf_opts.images_scale = 2.0
    pdf_opts.generate_picture_images = True
    pdf_opts.do_table_structure = True
    pdf_opts.table_structure_options = TableStructureV2Options(do_cell_matching=True)
    pdf_opts.do_ocr = ocr_enabled
    if ocr_enabled:
        pdf_opts.ocr_options = TesseractCliOcrOptions(lang=ocr_langs.split("+"))
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_opts)}
    )


def _build_smoldocling_converter(vlm_max_size: int):
    """SmolDocling VLM pipeline (docling-project/SmolDocling-256M-preview).

    Runs the 256M vision-language model in-process via the transformers backend
    (no vLLM needed). Best for scanned documents with no embedded text layer.
    Forced to float16 because Pascal GPUs (sm_61) lack native bfloat16.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.pipeline_options import (
        VlmPipelineOptions,
        VlmConvertOptions,
        AcceleratorOptions,
    )
    from docling.datamodel.accelerator_options import AcceleratorDevice
    from docling.datamodel.vlm_engine_options import VlmEngineType
    from docling.pipeline.vlm_pipeline import VlmPipeline
    import torch

    vlm_opts = VlmConvertOptions.from_preset("smoldocling")
    if torch.cuda.is_available():
        # Pascal (sm_61) has no native bfloat16; float16 halves memory and runs natively.
        vlm_opts.model_spec.engine_overrides[VlmEngineType.TRANSFORMERS].torch_dtype = "float16"
    vlm_opts.scale = 1.0
    vlm_opts.max_size = vlm_max_size
    device = AcceleratorDevice.CUDA if torch.cuda.is_available() else AcceleratorDevice.CPU
    pipe = VlmPipelineOptions(
        generate_page_images=True,
        generate_picture_images=True,
        images_scale=1.0,
        vlm_options=vlm_opts,
        accelerator_options=AcceleratorOptions(
            num_threads=4, device=device, cuda_use_flash_attention2=False
        ),
    )
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_cls=VlmPipeline, pipeline_options=pipe)
        }
    )


def _get_docling_converter(
    ocr_enabled: bool = False,
    ocr_langs: str = "ell+eng",
    docling_model: str = "default",
    vlm_max_size: int = 1280,
):
    global _docling_converter, _docling_converter_key
    key = (ocr_enabled, ocr_langs, docling_model, vlm_max_size)
    if _docling_converter is None or _docling_converter_key != key:
        try:
            if docling_model == "smoldocling":
                _docling_converter = _build_smoldocling_converter(vlm_max_size)
            else:
                _docling_converter = _build_default_converter(ocr_enabled, ocr_langs)
        except ImportError:
            raise ImportError(
                "Docling is required when USE_DOCLING=true. Install it with: pip install docling"
            )
        _docling_converter_key = key
    return _docling_converter


def _fix_table_column_order(conv_result) -> None:
    """Re-assign table column indices from horizontal position.

    The TableFormer sometimes gives cells correct bounding boxes but wrong
    column indices (e.g. a cyclic shift), so a school name lands under the
    wrong header. We re-derive columns from x-coordinates using the header row
    (or first row) as left-to-right anchors, so every cell sits under the
    correct column.
    """
    for table in getattr(conv_result.document, "tables", []) or []:
        data = getattr(table, "data", None)
        if data is None or data.num_cols < 2 or not data.table_cells:
            continue

        def _cx(cell):
            b = getattr(cell, "bbox", None)
            if b is None or getattr(b, "l", None) is None or getattr(b, "r", None) is None:
                return None
            return (b.l + b.r) / 2

        cells = data.table_cells
        cxs = [x for x in (_cx(c) for c in cells) if x is not None]
        if len(cxs) < 2:
            continue
        spread = (max(cxs) - min(cxs)) or 1.0
        tol = max(5.0, spread / (data.num_cols * 4))

        anchors = [c for c in cells if getattr(c, "column_header", False) and _cx(c) is not None]
        if not anchors:
            anchors = [c for c in cells if c.start_row_offset_idx == 0 and _cx(c) is not None]
        if not anchors:
            continue
        anchors = sorted(anchors, key=_cx)

        ref = []
        for c in anchors:
            x = _cx(c)
            if not ref or abs(x - ref[-1][0]) > tol:
                ref.append((x, len(ref)))
            if len(ref) >= data.num_cols:
                break
        if len(ref) < 2:
            continue

        for c in cells:
            x = _cx(c)
            if x is None:
                continue
            span = c.end_col_offset_idx - c.start_col_offset_idx
            nc = min(ref, key=lambda r: abs(r[0] - x))[1]
            c.start_col_offset_idx = nc
            c.end_col_offset_idx = nc + (span if span > 0 else 1)
        for row in data.grid:
            row.sort(key=lambda c: c.start_col_offset_idx)


def _doc_slug(stem: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", stem).strip("_")
    return (safe[:80] or "doc")


def _extract_with_docling(
    path: Path,
    ocr_enabled: bool = False,
    ocr_langs: str = "ell+eng",
    docling_model: str = "default",
    vlm_max_size: int = 1280,
) -> str:
    from docling_core.types.doc import ImageRefMode

    converter = _get_docling_converter(ocr_enabled, ocr_langs, docling_model, vlm_max_size)
    result = converter.convert(str(path))
    _fix_table_column_order(result)

    slug = _doc_slug(path.stem)
    images_dir = (DOC_IMAGES_ROOT / slug).resolve()
    if images_dir.exists():
        shutil.rmtree(images_dir)
    images_dir.mkdir(parents=True, exist_ok=True)
    DOC_MARKDOWN_ROOT.mkdir(parents=True, exist_ok=True)
    md_out = (DOC_MARKDOWN_ROOT / f"{slug}.md").resolve()

    result.document.save_as_markdown(
        md_out,
        artifacts_dir=images_dir,
        image_mode=ImageRefMode.REFERENCED,
    )
    md = md_out.read_text(encoding="utf-8")

    # Rewrite absolute artifact paths to URLs the FastAPI server can serve,
    # and persist the URL-rewritten markdown.
    md = md.replace(str(images_dir), f"/images/{slug}")
    md_out.write_text(md, encoding="utf-8")

    # Also write a self-contained markdown with images embedded as base64
    # data URIs — single-file artifact suitable for wikis that ingest only .md.
    DOC_EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    result.document.save_as_markdown(
        DOC_EXPORT_ROOT / f"{slug}.md",
        image_mode=ImageRefMode.EMBEDDED,
    )

    return md


def _load_file(
    path: Path,
    use_docling: bool,
    root: Path | None = None,
    ocr_enabled: bool = False,
    ocr_langs: str = "ell+eng",
    docling_model: str = "default",
    vlm_max_size: int = 1280,
) -> Document | None:
    suffix = path.suffix.lower()
    try:
        if use_docling and suffix in DOCLING_EXTENSIONS:
            content = _extract_with_docling(
                path, ocr_enabled, ocr_langs, docling_model, vlm_max_size
            )
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
    ocr_enabled: bool = False,
    ocr_langs: str = "ell+eng",
    docling_model: str = "default",
    vlm_max_size: int = 1280,
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
        doc = _load_file(
            path, use_docling, ocr_enabled=ocr_enabled, ocr_langs=ocr_langs,
            docling_model=docling_model, vlm_max_size=vlm_max_size,
        )
        if doc:
            docs.append(doc)
        return docs

    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            doc = _load_file(
                path, use_docling, root, ocr_enabled=ocr_enabled, ocr_langs=ocr_langs,
                docling_model=docling_model, vlm_max_size=vlm_max_size,
            )
            if doc:
                docs.append(doc)

    return docs
