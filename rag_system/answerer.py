import json
import os
from dataclasses import dataclass
from typing import List

from openai import OpenAI

from .config import Config


SYSTEM_PROMPT = """You answer questions strictly from the provided passages and return JSON.

Rules:
- Answer ONLY using information present in the passages. Do not use outside knowledge.
- If the passages do not contain enough information, say so explicitly in the same language as the question (e.g. "The provided passages do not contain this information.") and return an empty citations array.
- Reply in the same language as the user's question. If the question is in Greek, answer in Greek.
- Cite the passages you used with bracketed numbers like [1], [3] that match the passage numbers shown. Place citations inline next to the claims they support.
- For EVERY distinct [n] you cite, include exactly one entry in "citations" with a short verbatim 1-2 sentence excerpt from passage [n] that directly supports the claim. Do not paraphrase the excerpt — copy it verbatim from the passage. Keep it short.
- Do not invent quotes. Do not include excerpts for passages you did not cite.

Output strictly valid JSON with this shape and no other keys:
{
  "answer": "<your answer in markdown, with inline [n] citations>",
  "citations": [
    {"n": 1, "quote": "<verbatim 1-2 sentence excerpt from passage 1>"}
  ]
}
"""


@dataclass
class Citation:
    n: int
    quote: str


@dataclass
class AnswerResult:
    answer: str
    citations: List[Citation]


def _format_passages(results: List[dict]) -> str:
    blocks = []
    for i, r in enumerate(results, 1):
        meta = r["metadata"]
        doc_id = meta.get("doc_id", "?")
        text = (meta.get("original_content") or "").strip()
        blocks.append(f"[{i}] doc: {doc_id}\n{text}")
    return "\n\n".join(blocks)


class Answerer:
    def __init__(self, config: Config):
        self.client = OpenAI(
            api_key=config.openai_api_key or os.getenv("OPENAI_API_KEY"),
            base_url=config.openai_base_url,
        )
        self.model = config.answer_model
        self.max_tokens = config.answer_max_tokens

    def synthesize(self, query: str, results: List[dict]) -> AnswerResult:
        if not results:
            return AnswerResult(answer="", citations=[])
        passages = _format_passages(results)
        user_msg = (
            f"Passages:\n\n{passages}\n\n"
            f"Question: {query}\n\n"
            "Return JSON only."
        )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=self.max_tokens,
            temperature=0.0,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or ""
        try:
            data = json.loads(content)
            answer = (data.get("answer") or "").strip()
            citations_raw = data.get("citations") or []
            citations = [
                Citation(n=int(c["n"]), quote=str(c.get("quote", "")).strip())
                for c in citations_raw
                if isinstance(c, dict) and "n" in c
            ]
            return AnswerResult(answer=answer, citations=citations)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            return AnswerResult(answer=content.strip(), citations=[])
