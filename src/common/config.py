import torch

DATA_FOLDER = "data"
MODELS_FOLDER = "models"
INPUT_DATA_FOLDER = f"{DATA_FOLDER}/input"
OUTPUT_DATA_FOLDER = f"{DATA_FOLDER}/output"
PROCESSED_DATA_FOLDER = f"{DATA_FOLDER}/processed"
VECTOR_STORE_FOLDER = f"{DATA_FOLDER}/vector_store"

FAISS_INDEX_PATH = f"{VECTOR_STORE_FOLDER}/index.faiss"
METADATA_PATH = f"{VECTOR_STORE_FOLDER}/metadata.pkl"

GENERATOR_MODEL_NAME = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
EMBEDDING_MODEL_NAME = 'intfloat/multilingual-e5-small'
EMBEDDING_DIMENSION = 384

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

LOGGING_ENABLED = True

CHUNK_DEFAULT_SIZE = 500
CHUNK_DEFAULT_OVERLAP = 180
TOP_K_RESULTS = 2