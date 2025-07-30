import json, pathlib, numpy as np, tqdm, openai, faiss
from OTDR_CLI.src.config import config as cfg

EMB_MODEL = "text-embedding-3-small"   # 3k ctx, cheapest v3 model
ROOT_DIR = pathlib.Path(__file__).parent.parent.parent
CORPUS = ROOT_DIR / "corpus" / "docs.json"
INDEX_OUT = ROOT_DIR / "corpus" / "faiss.index"


docs = json.loads(CORPUS.read_text(encoding="utf-8"))
vecs = []

# Create a client instance 
client = openai.OpenAI(api_key=cfg.OPENAI_API_KEY)

for d in tqdm.tqdm(docs):
    # Updated API call format
    resp = client.embeddings.create(model=EMB_MODEL, input=d["text"])
    # Updated response structure
    vecs.append(resp.data[0].embedding)

vecs = np.asarray(vecs, dtype="float32")
index = faiss.IndexFlatIP(vecs.shape[1])    # cosine (after re-norm)
faiss.normalize_L2(vecs)                    # in-place
index.add(vecs)
faiss.write_index(index, str(INDEX_OUT))
print(f"Embeddings: {vecs.shape}, index saved → {INDEX_OUT}")