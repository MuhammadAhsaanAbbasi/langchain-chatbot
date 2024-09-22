from starlette.config import Config
from starlette.datastructures import Secret

try:
    config = Config(".env")
except FileNotFoundError:
    config = Config()

HUGGINGFACE_TOKEN=config("HUGGINGFACE_TOKEN", cast=str)
LANGCHAIN_API_KEY=config("LANGCHAIN_API_KEY", cast=Secret)