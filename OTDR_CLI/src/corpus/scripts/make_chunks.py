import glob, json, textwrap, uuid, pathlib

ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
CORPUS = ROOT_DIR / "corpus" / "docs.json"
RAW = ROOT_DIR / "corpus" / "raw"
OUT = pathlib.Path(CORPUS)
chunks = []

for fp in RAW.glob("*.txt"):
    text = open(fp, encoding="utf-8", errors="ignore").read()
    paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    buf, limit = [], 200  # ≈ 200 tokens (≈ 150 words)
    for para in paragraphs:
        buf.append(para)
        if sum(len(x.split()) for x in buf) >= limit:
            chunk_txt = " ".join(buf)
            chunks.append({
                "id": str(uuid.uuid4()),
                "text": chunk_txt,
                "meta": {
                    "source": pathlib.Path(fp).stem,
                },
            })
            buf = []
    if buf:
        chunks.append({
            "id": str(uuid.uuid4()),
            "text": " ".join(buf),
            "meta": {"source": pathlib.Path(fp).stem},
        })

print(f"{len(chunks)=}")
OUT.write_text(json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Corpus saved to {OUT} ({OUT.stat().st_size / 1024:.2f} KB)")
