import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if OPENAI_API_KEY is None:
    raise ValueError(
        "OPENAI_API_KEY environment variable is not set. Please set it in your .env file."
    )

if PINECONE_API_KEY is None:
    raise ValueError(
        "PINECONE_API_KEY environment variable is not set. Please set it in your .env file."
    )

if WANDB_API_KEY is None:
    raise ValueError(
        "WANDB_API_KEY environment variable is not set. Please set it in your .env file."
    )