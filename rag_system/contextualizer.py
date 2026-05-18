import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

from openai import OpenAI
from tqdm import tqdm

from .chunker import Chunk
from .config import Config
from .document_loader import Document


DOCUMENT_CONTEXT_PROMPT = """<document>
{doc_content}
</document>
"""

CHUNK_CONTEXT_PROMPT = """Here is the chunk we want to situate within the whole document
<chunk>
{chunk_content}
</chunk>

Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk.
Answer only with the succinct context and nothing else.
"""


class Contextualizer:
    def __init__(self, config: Config):
        self.client = OpenAI(
            api_key=config.openai_api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.openai_base_url,
        )
        self.model = config.openai_model
        self.token_counts = {"input": 0, "output": 0}
        self._lock = threading.Lock()

    def situate(self, doc_content: str, chunk_content: str) -> str:
        import time
        messages = [
            {
                "role": "user",
                "content": (
                    DOCUMENT_CONTEXT_PROMPT.format(doc_content=doc_content)
                    + CHUNK_CONTEXT_PROMPT.format(chunk_content=chunk_content)
                ),
            }
        ]
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    max_tokens=256,
                    temperature=0.0,
                )
                break
            except Exception as e:
                if "rate limit" in str(e).lower() and attempt < 2:
                    time.sleep(2 ** attempt)
                    continue
                raise
        with self._lock:
            self.token_counts["input"] += response.usage.prompt_tokens
            self.token_counts["output"] += response.usage.completion_tokens
        content = response.choices[0].message.content
        return content.strip() if content else ""

    def contextualize_chunks(
        self, chunks: List[Chunk], docs: List[Document], parallel_threads: int = 3
    ) -> Dict[str, str]:
        doc_map = {d.doc_id: d for d in docs}
        results: Dict[str, str] = {}

        def _process(chunk: Chunk) -> tuple:
            doc = doc_map.get(chunk.doc_id)
            if not doc:
                return chunk.chunk_id, ""
            context = self.situate(doc.content, chunk.content)
            return chunk.chunk_id, context

        print(f"Contextualizing {len(chunks)} chunks with {parallel_threads} threads...")
        with ThreadPoolExecutor(max_workers=parallel_threads) as executor:
            futures = {executor.submit(_process, c): c for c in chunks}
            for future in tqdm(as_completed(futures), total=len(chunks), desc="Contextualizing"):
                cid, context = future.result()
                results[cid] = context

        print(
            f"Done. Tokens: input={self.token_counts['input']}, output={self.token_counts['output']}"
        )
        return results
