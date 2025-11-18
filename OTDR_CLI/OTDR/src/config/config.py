import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "fiberwatch")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "otdr-prod")

if OPENAI_API_KEY is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. Please set it in your .env file."
    )

if PINECONE_API_KEY is None:
    raise ValueError(
        "PINECONE_API_KEY environment variable is not set. Please set it in your .env file."
    )

if WANDB_API_KEY is None:
    # WANDB is optional – only warn so RAG and inference can run without it.
    print("[config] WANDB_API_KEY not set; WANDB logging will be disabled.")