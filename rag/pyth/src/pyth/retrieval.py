from llama_index.core import VectorStoreIndex
from llama_index.core.schema import NodeWithScore

TOP_K: int = 3

def retrieve_relevant_chunks(index: VectorStoreIndex, question: str) -> list[NodeWithScore]:
    retriever = index.as_retriever(similarity_top_k=TOP_K)
    return retriever.retrieve(question)
