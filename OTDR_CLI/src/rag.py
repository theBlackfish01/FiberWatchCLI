# src/rag.py
from __future__ import annotations

import json
import pathlib
from typing import List

import faiss
import numpy as np
from openai import OpenAI
import config.config as cfg

_CDIR = pathlib.Path(__file__).parent / "corpus"
_INDEX = faiss.read_index(str(_CDIR / "faiss.index"))
_DOCS: List[dict] = json.loads((_CDIR / "docs.json").read_text(encoding="utf-8"))
_EMB_MODEL = "text-embedding-3-small"

_client = OpenAI(api_key=cfg.OPENAI_API_KEY)                 # picks up OPENAI_API_KEY from env


def retrieve(query: str, k: int = 5) -> List[dict]:
    """
    Vector‑search the local FAISS index for *k* docs most relevant to *query*.

    Returns a list of documents as dicts stored in docs.json.
    """
    # -------- 1) embed query --------------------------------------------
    resp = _client.embeddings.create(model=_EMB_MODEL, input=[query])
    emb = np.asarray([resp.data[0].embedding], dtype="float32")

    # -------- 2) similarity search --------------------------------------
    faiss.normalize_L2(emb)
    _, I = _INDEX.search(emb, k)

    # -------- 3) fetch docs ---------------------------------------------
    return [_DOCS[i] for i in I[0]]
