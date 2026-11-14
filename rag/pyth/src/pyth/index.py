from llama_index.core import VectorStoreIndex
from llama_index.core.schema import BaseNode
from pyth.embedding import embedder

def create_index(chunks: list[BaseNode]) -> VectorStoreIndex:
    return VectorStoreIndex(nodes=chunks, embed_model=embedder)