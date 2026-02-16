import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
WANDB_API_KEY = os.getenv("WANDB_API_KEY")

if OPENAI_API_KEY is None:
    #raise ValueError(
    print("OPENAI_API_KEY environment variable is not set. Please set it in your .env file.")
    

if PINECONE_API_KEY is None:
    #raise ValueError(
    print(
        "PINECONE_API_KEY environment variable is not set. Please set it in your .env file."
    )

if WANDB_API_KEY is None:
    print("[WARN] WANDB_API_KEY environment variable is not set. Logging to Weights & Biases will be disabled.")