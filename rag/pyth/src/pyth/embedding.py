from llama_index.embeddings.openai_like import OpenAILikeEmbedding

API_BASE: str = "http://spark-8b01.local:8082/v1"
API_KEY: str = "fake"
MODEL: str = "qwen3-embedding";
DIMENSIONS: int = 1024;

embedder = OpenAILikeEmbedding(
    model_name=MODEL,
    api_base=API_BASE,
    api_key=API_KEY,
    dimensions=DIMENSIONS,
)

def embed_text(text: str) -> list[float]:
    return embedder.get_text_embedding(text)

def embed_texts(texts: list[str]) -> list[list[float]]:
    return embedder.get_text_embedding_batch(texts)