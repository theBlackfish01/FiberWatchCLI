from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path
from typing import Iterable, List

ROOT_DIR = Path(__file__).parent.parent
RAW = ROOT_DIR / "raw"
CORPUS = ROOT_DIR / "docs.json"


def _iter_raw_files(raw_dir: Path) -> Iterable[Path]:
    if not raw_dir.exists():
        raise FileNotFoundError(f"Corpus raw directory not found: {raw_dir}")
    yield from sorted(raw_dir.glob("*.txt"))


def generate_chunks(raw_dir: Path = RAW, *, limit_words: int = 200) -> List[dict]:
    """Split raw corpus text files into small JSON chunks."""

    chunks: List[dict] = []
    for fp in _iter_raw_files(raw_dir):
        text = fp.read_text(encoding="utf-8", errors="ignore")
        paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
        buf: list[str] = []
        for para in paragraphs:
            buf.append(para)
            if sum(len(x.split()) for x in buf) >= limit_words:
                chunk_txt = " ".join(buf)
                chunks.append(
                    {
                        "id": str(uuid.uuid4()),
                        "text": chunk_txt,
                        "meta": {"source": fp.stem},
                    }
                )
                buf = []
        if buf:
            chunks.append(
                {
                    "id": str(uuid.uuid4()),
                    "text": " ".join(buf),
                    "meta": {"source": fp.stem},
                }
            )
    return chunks


def write_chunks(chunks: List[dict], out_path: Path = CORPUS) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path.resolve()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create OTDR corpus chunks for RAG.")
    parser.add_argument("--raw-dir", type=Path, default=RAW, help="Path to raw .txt corpus files")
    parser.add_argument(
        "--output", type=Path, default=CORPUS, help="Destination docs.json path"
    )
    parser.add_argument(
        "--limit-words",
        type=int,
        default=200,
        help="Approximate token-equivalent limit per chunk (in words)",
    )
    args = parser.parse_args()

    chunks = generate_chunks(args.raw_dir, limit_words=args.limit_words)
    print(f"Generated {len(chunks)} chunks from {args.raw_dir.resolve()}")
    out_path = write_chunks(chunks, args.output)
    size_kb = args.output.stat().st_size / 1024
    print(f"Corpus saved to {out_path} ({size_kb:.2f} KB)")


if __name__ == "__main__":
    main()
