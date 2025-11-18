from __future__ import annotations

import argparse
import json
import time
import uuid
from pathlib import Path
from typing import List, Sequence

from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

import config.config as cfg
from corpus.scripts.make_chunks import generate_chunks

_INDEX_NAME = "fiberwatch"
_EMBED_MODEL = "text-embedding-3-large"
_EMBED_DIM = 3072
_DEFAULT_BATCH = 32
_DEFAULT_NAMESPACE = None

ROOT_DIR = Path(__file__).parent.parent
DEFAULT_DOCS_PATH = ROOT_DIR / "docs.json"


def load_chunks(source: Path | None) -> List[dict]:
    if source and source.exists():
        data = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError(f"Expected a list of chunks in {source}")
        return data
    raise FileNotFoundError(f"docs.json not found at {source}")


def _index_names(pc: Pinecone) -> set[str]:
    listing = pc.list_indexes()
    if hasattr(listing, "names"):
        return set(listing.names())
    names: set[str] = set()
    for item in listing:
        if hasattr(item, "name"):
            names.add(item.name)
        elif isinstance(item, dict):
            val = item.get("name")
            if val:
                names.add(val)
    return names


def _extract_dimension(description: object) -> int | None:
    """Best effort extraction of the index dimension."""

    candidates: list[object | None] = [description]
    if hasattr(description, "to_dict"):
        try:
            candidates.append(description.to_dict())
        except Exception:  # pragma: no cover - defensive
            pass
    if hasattr(description, "model_dump"):
        try:
            candidates.append(description.model_dump())
        except Exception:  # pragma: no cover - defensive
            pass

    while candidates:
        current = candidates.pop()
        if current is None:
            continue
        if isinstance(current, dict):
            dim = current.get("dimension")
            if dim is not None:
                return int(dim)
            config = current.get("config") or current.get("index_config")
            if config:
                candidates.append(config)
            continue
        if hasattr(current, "dimension"):
            try:
                return int(getattr(current, "dimension"))
            except (TypeError, ValueError):
                pass
        for attr in ("config", "index_config"):
            if hasattr(current, attr):
                candidates.append(getattr(current, attr))
    return None


def _wait_for_index_ready(pc: Pinecone) -> None:
    while True:
        desc = pc.describe_index(_INDEX_NAME)
        status = getattr(desc, "status", {})
        ready = status.get("ready") if isinstance(status, dict) else getattr(status, "ready", False)
        if ready:
            break
        time.sleep(5)
    print(f"[pinecone] Index {_INDEX_NAME} ready")


def ensure_index(pc: Pinecone) -> None:
    names = _index_names(pc)
    if _INDEX_NAME in names:
        desc = pc.describe_index(_INDEX_NAME)
        dimension = _extract_dimension(desc)
        if dimension == _EMBED_DIM:
            return
        print(
            f"[pinecone] Index {_INDEX_NAME} has dimension {dimension} but {_EMBED_DIM} required. "
            "Recreating index ..."
        )
        pc.delete_index(_INDEX_NAME)
        while _INDEX_NAME in _index_names(pc):
            time.sleep(2)

    print(f"[pinecone] Creating index {_INDEX_NAME} ...")
    pc.create_index(
        name=_INDEX_NAME,
        dimension=_EMBED_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-west-2"),
    )
    _wait_for_index_ready(pc)


def chunk_iterator(args: argparse.Namespace) -> List[dict]:
    if args.raw_dir:
        return generate_chunks(Path(args.raw_dir), limit_words=args.limit_words)
    return load_chunks(Path(args.docs_path))


def upsert_chunks(chunks: Sequence[dict], namespace: str | None, batch_size: int) -> None:
    client = OpenAI(api_key=cfg.OPENAI_API_KEY)
    pinecone = Pinecone(api_key=cfg.PINECONE_API_KEY)
    ensure_index(pinecone)
    index = pinecone.Index(_INDEX_NAME)

    total = len(chunks)
    for start in range(0, total, batch_size):
        batch = list(chunks[start : start + batch_size])
        texts = [chunk.get("text", "") for chunk in batch]
        embeddings = client.embeddings.create(model=_EMBED_MODEL, input=texts).data
        vectors = []
        for chunk, emb, offset in zip(batch, embeddings, range(start, start + len(batch))):
            chunk_id = chunk.get("id") or str(uuid.uuid4())
            metadata = dict(chunk.get("meta") or {})
            metadata.update(
                {
                    "text": chunk.get("text", ""),
                    "chunk_index": offset,
                }
            )
            vectors.append(
                {
                    "id": str(chunk_id),
                    "values": list(emb.embedding),
                    "metadata": metadata,
                }
            )
        index.upsert(vectors=vectors, namespace=namespace)
        print(f"[pinecone] Upserted {start + len(batch)}/{total} chunks")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sync OTDR corpus chunks into Pinecone")
    parser.add_argument("--docs-path", type=Path, default=DEFAULT_DOCS_PATH, help="Path to docs.json")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        help="Optional raw corpus directory to chunk on the fly (skips docs.json)",
    )
    parser.add_argument("--namespace", default=_DEFAULT_NAMESPACE, help="Pinecone namespace")
    parser.add_argument("--batch-size", type=int, default=_DEFAULT_BATCH, help="Embedding batch size")
    parser.add_argument(
        "--limit-words",
        type=int,
        default=200,
        help="Word budget per chunk when --raw-dir is provided",
    )
    args = parser.parse_args()

    chunks = chunk_iterator(args)
    print(f"Loaded {len(chunks)} chunks")
    upsert_chunks(chunks, namespace=args.namespace, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
