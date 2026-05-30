from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import httpx


SYSTEM_PROMPT = """
You are an expert research-paper analyst. Answer using ONLY the retrieved passages, with clear, publication-quality exposition.

Formatting (required):
- Structure answers with markdown: start with a 1–2 sentence direct answer, then ## sections (e.g. Summary, Method, Results, Limitations).
- Use **bold** for key terms, bullet or numbered lists for steps, and blockquotes for short direct quotes.
- For mathematics: use LaTeX — inline $...$ and display equations on their own line as $$...$$. Never wrap LaTeX in code fences.
- Use tables when comparing quantities or variants. Use `inline code` only for identifiers, hyperparameters, or filenames.

Citations:
- Cite inline as [Source Title, Page N] immediately after the claim, using the passage metadata.

Quality:
- Synthesize across passages; do not repeat passage text verbatim unless quoting.
- If context is partial, state what is supported and what is missing.
- Be concise but thorough; prefer precise technical language over vague summaries.
""".strip()

USER_PROMPT = """
## Retrieved passages

{formatted_chunks}

---

## Question

{query}

Write a well-structured, richly formatted answer (markdown + LaTeX where appropriate). Lead with the takeaway, then develop the reasoning with citations.
""".strip()


@dataclass
class ChunkLike:
    doc_title: str
    type: str
    content: str
    metadata: dict[str, Any]


class GroqGenerator:
    def __init__(self, api_key: str | None = None, model: str = "llama-3.3-70b-versatile"):
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        self.client = httpx.Client(timeout=120)

    def generate(self, query: str, retrieved_chunks: list[Any]) -> str:
        return self._generate(query=query, retrieved_chunks=retrieved_chunks, stream=False)

    def generate_with_citations(self, query: str, chunks: list[Any]) -> str:
        return self.generate(query, chunks)

    def stream(self, query: str, retrieved_chunks: list[Any]):
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        formatted_chunks = self._format_chunks(retrieved_chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(formatted_chunks=formatted_chunks, query=query)},
        ]
        with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={"model": self.model, "messages": messages, "temperature": 0.2, "stream": True},
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload == "[DONE]":
                    break
                data = json.loads(payload)
                for choice in data.get("choices", []):
                    delta = choice.get("delta", {})
                    text = delta.get("content")
                    if text:
                        yield text

    def _generate(self, query: str, retrieved_chunks: list[Any], stream: bool = False) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        formatted_chunks = self._format_chunks(retrieved_chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT.format(formatted_chunks=formatted_chunks, query=query)},
        ]
        response = self.client.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json={"model": self.model, "messages": messages, "temperature": 0.2, "stream": stream},
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def _format_chunks(chunks: list[Any]) -> str:
        parts: list[str] = []
        for chunk in chunks:
            metadata = getattr(chunk, "metadata", {}) if not isinstance(chunk, dict) else chunk.get("metadata", {})
            doc_title = getattr(chunk, "doc_title", None) if not isinstance(chunk, dict) else chunk.get("doc_title")
            chunk_type = getattr(chunk, "type", None) if not isinstance(chunk, dict) else chunk.get("type", "text")
            content = getattr(chunk, "content", None) if not isinstance(chunk, dict) else chunk.get("content", "")
            page = metadata.get("page_number", "-") if isinstance(metadata, dict) else "-"
            title = doc_title or (chunk.get("doc_id", "") if isinstance(chunk, dict) else "")
            parts.append(f"[Source: {title}, Type: {chunk_type}, Location: {page}]\n{content}")
        return "\n\n".join(parts)
