import hashlib
import re
from dataclasses import dataclass
from typing import List

from .document_loader import Document


@dataclass
class Chunk:
    doc_id: str
    chunk_id: str
    original_index: int
    content: str


SEPARATORS = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
MD_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")


def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Simple recursive character text splitter."""
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            for sep in SEPARATORS:
                idx = text.rfind(sep, start, end)
                if idx > start:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end])
        if end >= len(text):
            break
        next_start = end - chunk_overlap
        if next_start <= start:
            next_start = end
        start = next_start
    return chunks


def _split_on_headings(text: str) -> List[tuple[List[str], str]]:
    """Split markdown into (heading_path, body) pairs.

    heading_path is the stack of ancestor headings, e.g. ["# Intro", "## Background"].
    Returns [([], text)] if no headings are found.
    """
    matches = list(HEADING_RE.finditer(text))
    if not matches:
        return [([], text)]

    sections: List[tuple[List[str], str]] = []
    stack: List[tuple[int, str]] = []

    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(([], preamble))

    for i, m in enumerate(matches):
        level = len(m.group(1))
        heading_line = m.group(0).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, heading_line))
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        path = [line for _, line in stack]
        sections.append((path, body))

    return sections


def _render_section(path: List[str], body: str) -> str:
    """One section as markdown text including its heading path."""
    if not body and not path:
        return ""
    if not path:
        return body
    if not body:
        return "\n".join(path)
    return "\n".join(path) + "\n\n" + body


def _is_table_block(text: str) -> bool:
    """True if every non-blank line in the block is a markdown table row."""
    lines = [l for l in text.splitlines() if l.strip()]
    return len(lines) >= 2 and all(MD_TABLE_ROW_RE.match(l) for l in lines)


def _text_to_blocks(text: str) -> List[str]:
    """Split text into blocks on blank lines. Adjacent table-row blocks with no
    blank line between them are merged so a whole markdown table is one block."""
    lines = text.split("\n")
    blocks: List[str] = []
    cur: List[str] = []
    for line in lines:
        if line.strip() == "":
            if cur:
                blocks.append("\n".join(cur))
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append("\n".join(cur))
    merged: List[str] = []
    for b in blocks:
        if merged and _is_table_block(merged[-1]) and _is_table_block(b):
            merged[-1] = merged[-1] + "\n" + b
        else:
            merged.append(b)
    return merged


def _pack_sections(sections: List[str], chunk_size: int, chunk_overlap: int) -> List[str]:
    """Greedily pack blocks into chunks ≤ chunk_size, keeping markdown tables
    intact (a table is never split mid-row). Oversized non-table blocks are
    split via recursive_split; an oversized table stands alone as one chunk.
    A final pass merges tiny tail chunks into their previous neighbor.
    """
    blocks: List[str] = []
    for sec in sections:
        for part in _text_to_blocks(sec.strip()):
            if _is_table_block(part) or len(part) <= chunk_size:
                blocks.append(part)
            else:
                blocks.extend(recursive_split(part, chunk_size, chunk_overlap))

    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0
    sep = "\n\n"

    def flush():
        nonlocal buf, buf_len
        if buf:
            chunks.append(sep.join(buf))
            buf = []
            buf_len = 0

    for b in blocks:
        addition = len(b) + (len(sep) if buf else 0)
        if buf and buf_len + addition > chunk_size:
            flush()
            addition = len(b)
        buf.append(b)
        buf_len += addition
        if len(b) > chunk_size:
            flush()
    flush()

    # Merge tiny chunks back into their previous neighbor (allow up to 20% overflow).
    if len(chunks) < 2:
        return chunks
    min_size = max(80, chunk_size // 4)
    max_merged = int(chunk_size * 1.2)
    merged: List[str] = [chunks[0]]
    for c in chunks[1:]:
        if len(c) < min_size and len(merged[-1]) + len(c) + len(sep) <= max_merged:
            merged[-1] = merged[-1] + sep + c
        else:
            merged.append(c)
    return merged


def chunk_documents(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    all_chunks = []
    for doc in docs:
        sections = _split_on_headings(doc.content)
        has_headings = any(path for path, _ in sections)
        if not has_headings:
            texts = _pack_sections([doc.content], chunk_size, chunk_overlap)
        else:
            rendered = [_render_section(path, body) for path, body in sections]
            texts = _pack_sections(rendered, chunk_size, chunk_overlap)

        for i, text in enumerate(texts):
            chunk_id = hashlib.md5(f"{doc.doc_id}:{i}:{text[:50]}".encode()).hexdigest()
            all_chunks.append(
                Chunk(
                    doc_id=doc.doc_id,
                    chunk_id=chunk_id,
                    original_index=i,
                    content=text,
                )
            )
    return all_chunks
