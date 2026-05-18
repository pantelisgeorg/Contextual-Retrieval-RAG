from dataclasses import dataclass, field

import requests

from .config import Config


# qmd-query-expansion is a fine-tuned Qwen3-1.7B. The Ollama Modelfile has
# TEMPLATE `{{ .Prompt }}`, so we must wrap the input in Qwen3 chat tags
# ourselves. `/no_think` disables the <think> block. The model is trained to
# emit lines prefixed with `lex:`, `vec:`, or `hyde:`.
_PROMPT = (
    "<|im_start|>user\n"
    "/no_think Expand this search query: {query}<|im_end|>\n"
    "<|im_start|>assistant\n"
)

_VALID_PREFIXES = ("lex:", "vec:", "hyde:")


@dataclass
class ExpandedQuery:
    original: str
    lex: list[str] = field(default_factory=list)   # keyword-style → BM25
    vec: list[str] = field(default_factory=list)   # natural language → embedding
    hyde: list[str] = field(default_factory=list)  # hypothetical passages → embedding

    def vector_queries(self) -> list[str]:
        """Queries to embed for semantic search (original + vec + hyde)."""
        return _dedupe([self.original, *self.vec, *self.hyde])

    def bm25_queries(self) -> list[str]:
        """Queries for BM25 (original + lex)."""
        return _dedupe([self.original, *self.lex])


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for s in items:
        s = s.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    return out


class QueryExpander:
    def __init__(self, config: Config):
        self.host = config.ollama_host
        self.model = config.ollama_query_expansion_model

    def expand(self, query: str) -> ExpandedQuery:
        """Call the qmd-query-expansion model and parse its lex:/vec:/hyde: output.

        Falls back to an ExpandedQuery containing only the original query on any
        failure, which makes the rest of the pipeline behave as if expansion is
        disabled.
        """
        prompt = _PROMPT.format(query=query.strip())
        try:
            response = requests.post(
                f"{self.host}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "raw": True,  # we already applied the chat template
                    "stream": False,
                    "options": {
                        "num_predict": 256,
                        "temperature": 0.3,
                        "top_p": 0.9,
                        "num_ctx": 2048,
                        # Stop as soon as the model tries to start a new turn.
                        "stop": ["<|im_end|>", "<|im_start|>"],
                    },
                },
                timeout=300,
            )
            response.raise_for_status()
            text = response.json().get("response", "")
        except Exception as e:
            print(f"  [query-expander] LLM call failed: {e}; using original query only")
            return ExpandedQuery(original=query)

        return self._parse(query, text)

    @staticmethod
    def _parse(query: str, text: str) -> ExpandedQuery:
        result = ExpandedQuery(original=query)
        if not text:
            return result

        # Strip any leftover <think>...</think> block (in case /no_think failed).
        if "<think>" in text and "</think>" in text:
            text = text.split("</think>", 1)[1]

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                continue
            lower = line.lower()
            for prefix in _VALID_PREFIXES:
                if lower.startswith(prefix):
                    value = line[len(prefix):].strip().strip("`*-•").strip()
                    if not value:
                        break
                    bucket = getattr(result, prefix[:-1])
                    bucket.append(value)
                    break

        result.lex = _dedupe(result.lex)
        result.vec = _dedupe(result.vec)
        result.hyde = _dedupe(result.hyde)
        return result
