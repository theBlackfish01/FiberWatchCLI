"""Retrieval Augmented Generation utilities backed by Pinecone."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence

import tiktoken
from openai import OpenAI
from pinecone import Pinecone

import config.config as cfg

_INDEX_NAME = getattr(cfg, "PINECONE_INDEX_NAME", "fiberwatch")
_DEFAULT_NAMESPACE = getattr(cfg, "PINECONE_NAMESPACE", None)
_EMBED_MODEL = "text-embedding-3-large"
_DEFAULT_MAX_CONTEXT_TOKENS = 1600
_TOKENIZER = tiktoken.get_encoding("cl100k_base")

_client = OpenAI(api_key=cfg.OPENAI_API_KEY)
_pinecone = Pinecone(api_key=cfg.PINECONE_API_KEY)
_index = _pinecone.Index(_INDEX_NAME)


@dataclass(slots=True)
class RetrievedDocument:
    """Container for a document returned from vector search."""

    id: str
    text: str
    score: float
    metadata: dict[str, Any]
    tokens: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata,
            "tokens": self.tokens,
        }


def _count_tokens(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def _truncate_to_tokens(text: str, token_limit: int) -> str:
    if token_limit <= 0:
        return ""
    token_ids = _TOKENIZER.encode(text, disallowed_special=())
    if len(token_ids) <= token_limit:
        return text
    return _TOKENIZER.decode(token_ids[:token_limit])


def _embed_query(query: str) -> List[float]:
    resp = _client.embeddings.create(model=_EMBED_MODEL, input=[query])
    return list(resp.data[0].embedding)


def retrieve(query: str, k: int = 5, *, namespace: str | None = None) -> List[dict[str, Any]]:
    """Query the Pinecone index and return the top *k* documents."""

    vector = _embed_query(query)
    search_kwargs: dict[str, Any] = {
        "vector": vector,
        "top_k": k,
        "include_metadata": True,
    }
    resolved_namespace = _DEFAULT_NAMESPACE if namespace is None else namespace
    if resolved_namespace:
        search_kwargs["namespace"] = resolved_namespace
    if resolved_namespace is None:
        print("[rag] Querying Pinecone default namespace; set PINECONE_NAMESPACE to scope documents.")
    elif namespace is not None and namespace != _DEFAULT_NAMESPACE:
        print(f"[rag] Overriding configured namespace {_DEFAULT_NAMESPACE!r} with {namespace!r}.")
    response = _index.query(**search_kwargs)

    results: List[RetrievedDocument] = []
    for match in getattr(response, "matches", []) or []:
        metadata = match.metadata or {}
        raw_text = metadata.get("text") or metadata.get("content") or metadata.get("chunk") or ""
        token_count = _count_tokens(raw_text)
        results.append(
            RetrievedDocument(
                id=str(match.id),
                text=raw_text,
                score=float(match.score or 0.0),
                metadata=metadata,
                tokens=token_count,
            )
        )

    return [doc.as_dict() for doc in results]


def build_reference_block(
    docs: Sequence[dict[str, Any]] | Sequence[RetrievedDocument],
    *,
    max_context_tokens: int = _DEFAULT_MAX_CONTEXT_TOKENS,
    include_scores: bool = True,
) -> str:
    """Convert retrieved documents into a prompt-ready context block."""

    block_parts: List[str] = []
    remaining = max_context_tokens

    for idx, doc in enumerate(docs, start=1):
        current = doc.as_dict() if isinstance(doc, RetrievedDocument) else doc
        text = current.get("text", "")
        if not text:
            continue
        tokens = int(current.get("tokens") or _count_tokens(text))
        if remaining <= 0:
            break
        if tokens > remaining:
            text = _truncate_to_tokens(text, remaining)
            tokens = remaining
        remaining -= tokens

        meta = current.get("metadata") or {}
        source = meta.get("source") or meta.get("title") or current.get("id", f"chunk-{idx}")
        score_fragment = f" (score: {current.get('score'):.3f})" if include_scores and current.get("score") is not None else ""
        block_parts.append(f"[{idx}] {source}{score_fragment}\n{text}")

    return "\n\n".join(block_parts)


def iter_texts(docs: Sequence[dict[str, Any]] | Sequence[RetrievedDocument]) -> Iterable[str]:
    """Yield the raw text payload from retrieved documents."""

    for doc in docs:
        if isinstance(doc, RetrievedDocument):
            yield doc.text
        else:
            text = doc.get("text")
            if text:
                yield str(text)
