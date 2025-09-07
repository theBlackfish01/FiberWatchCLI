# PHI-OTDR/src/rag.py
from __future__ import annotations
"""
FAISS RAG utilities for Φ-OTDR:
- build: embed corpus (.txt/.md/.pdf) into a FAISS index
- retrieve: query top-k chunks for augmentation

Default embedding: OpenAI `text-embedding-3-large`.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple, Iterable

import numpy as np
import faiss

# ---------- Optional API key via project config; falls back to env ----------
try:
    import config.config as cfg  # project-level config (PHI-OTDR/src/config/config.py)
    _OPENAI_KEY = getattr(cfg, "OPENAI_API_KEY", None)
except Exception:
    _OPENAI_KEY = None


# -------------------------- IO & text utilities -------------------------- #

_SUPPORTED_TXT = {".txt", ".md", ".rst"}
_SUPPORTED_PDF = {".pdf"}

def _read_text_file(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def _read_pdf_file(p: Path) -> str:
    # Lightweight PDF extractor using pypdf
    from pypdf import PdfReader
    out = []
    reader = PdfReader(p.as_posix())
    for page in reader.pages:
        txt = page.extract_text() or ""
        out.append(txt)
    return "\n".join(out)

def _iter_corpus(corpus_dir: Path) -> Iterable[Tuple[Path, str]]:
    for p in sorted(Path(corpus_dir).rglob("*")):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        try:
            if ext in _SUPPORTED_TXT:
                text = _read_text_file(p)
            elif ext in _SUPPORTED_PDF:
                text = _read_pdf_file(p)
            else:
                continue
            text = " ".join(text.split())  # normalize whitespace
            if text.strip():
                yield p, text
        except Exception:
            # Skip unreadable files without crashing
            continue

def _chunk(text: str, chunk_chars: int = 1300, overlap: int = 200) -> List[Tuple[int, int, str]]:
    """
    Simple char-based chunker (token-agnostic) suitable for embeddings.
    Returns list of (start, end, chunk_text).
    """
    if len(text) <= chunk_chars:
        return [(0, len(text), text)]
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + chunk_chars)
        chunk_text = text[start:end]
        # try not to cut in the middle of a word
        if end < len(text):
            rspace = chunk_text.rfind(" ")
            if rspace >= int(chunk_chars * 0.6):
                end = start + rspace
                chunk_text = text[start:end]
        chunks.append((start, end, chunk_text))
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


# ------------------------------- Embeddings ------------------------------- #

def _get_openai_client():
    key = _OPENAI_KEY or os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set (env or config).")
    from openai import OpenAI
    return OpenAI(api_key=key)

def embed_texts(texts: List[str], model: str = "text-embedding-3-large", batch_size: int = 128) -> np.ndarray:
    """
    Returns float32 array of shape (N, D). Uses OpenAI embeddings.
    """
    client = _get_openai_client()
    out_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        resp = client.embeddings.create(model=model, input=batch)
        vecs = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        out_vecs.append(np.vstack(vecs))
    embs = np.vstack(out_vecs)
    # Normalize for cosine similarity with IndexFlatIP
    faiss.normalize_L2(embs)
    return embs


# ------------------------------ Build/Search ------------------------------ #

def build_index(
    corpus_dir: Path,
    index_path: Path,
    store_path: Path,
    embed_model: str = "text-embedding-3-large",
) -> None:
    """
    Build FAISS index and sidecar JSONL store with chunk metadata.
    """
    corpus_dir = Path(corpus_dir)
    index_path = Path(index_path)
    store_path = Path(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[Dict] = []
    texts: List[str] = []

    idx = 0
    for src, text in _iter_corpus(corpus_dir):
        chunks = _chunk(text, chunk_chars=1300, overlap=200)
        for j, (a, b, chunk_text) in enumerate(chunks):
            records.append({
                "id": idx,
                "source": src.as_posix(),
                "chunk_index": j,
                "char_start": a,
                "char_end": b,
                "text": chunk_text,
            })
            texts.append(chunk_text)
            idx += 1

    if not texts:
        raise RuntimeError(f"No text extracted under {corpus_dir}")

    embs = embed_texts(texts, model=embed_model)  # (N, D)
    d = embs.shape[1]
    index = faiss.IndexFlatIP(d)  # cosine (with L2-normalized vectors)
    index.add(embs)

    faiss.write_index(index, index_path.as_posix())
    with store_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    meta = {
        "embed_model": embed_model,
        "dimension": d,
        "num_chunks": len(records),
        "corpus_dir": corpus_dir.as_posix(),
    }
    (store_path.with_suffix(".meta.json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[RAG] Built index with {len(records)} chunks -> {index_path.name} / {store_path.name}")


def _load_store(store_path: Path) -> List[Dict]:
    with Path(store_path).open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def retrieve(
    query: str,
    k: int,
    index_path: Path,
    store_path: Path,
    embed_model: str = "text-embedding-3-large",
) -> List[Dict]:
    """
    Return top-k chunks: [{"text": str, "source": str, "score": float}, ...]
    """
    index = faiss.read_index(Path(index_path).as_posix())
    store = _load_store(store_path)
    q_vec = embed_texts([query], model=embed_model)  # (1, D)
    scores, idxs = index.search(q_vec, k)
    out = []
    for score, ix in zip(scores[0].tolist(), idxs[0].tolist()):
        if ix < 0 or ix >= len(store):
            continue
        rec = store[ix]
        out.append({"text": rec["text"], "source": rec["source"], "score": float(score)})
    return out


# --------------------------------- CLI --------------------------------- #

import click

@click.group(context_settings=dict(help_option_names=["-h", "--help"]))
def cli():
    """FAISS RAG indexer for Φ-OTDR."""
    pass

@cli.command("build")
@click.option("--corpus", "corpus_dir", type=click.Path(path_type=Path), required=True,
              help="Folder with .txt/.md/.pdf knowledge.")
@click.option("--index", "index_path", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "corpus" / "index.faiss",
              show_default=True)
@click.option("--store", "store_path", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "corpus" / "chunks.jsonl",
              show_default=True)
@click.option("--embed-model", type=str, default="text-embedding-3-large", show_default=True)
def build_cmd(corpus_dir: Path, index_path: Path, store_path: Path, embed_model: str):
    build_index(corpus_dir, index_path, store_path, embed_model)

@cli.command("search")
@click.option("--query", type=str, required=True)
@click.option("--k", type=int, default=5, show_default=True)
@click.option("--index", "index_path", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "corpus" / "index.faiss",
              show_default=True)
@click.option("--store", "store_path", type=click.Path(path_type=Path),
              default=lambda: Path(__file__).resolve().parent / "corpus" / "chunks.jsonl",
              show_default=True)
@click.option("--embed-model", type=str, default="text-embedding-3-large", show_default=True)
def search_cmd(query: str, k: int, index_path: Path, store_path: Path, embed_model: str):
    hits = retrieve(query, k, index_path, store_path, embed_model)
    for i, h in enumerate(hits, 1):
        print(f"[{i}] score={h['score']:.3f} source={h['source']}\n{h['text']}\n")

if __name__ == "__main__":
    cli()
