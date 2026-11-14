from pathlib import Path
from llama_index.core import Document, SimpleDirectoryReader 
from llama_index.core.schema import BaseNode
from llama_index.core.node_parser import SentenceSplitter

CHUNK_SIZE: int = 256
CHUNK_OVERLAP: int = 50

def load_documents() -> list[Document]:
    # data_dir = (Path(__file__).resolve() / "../../../../data").resolve()
    data_dir = Path(__file__).resolve().parents[3] / "data"
    return SimpleDirectoryReader(input_dir=data_dir).load_data()

def chunk_documents(documents: list[Document]) -> list[BaseNode]:
    splitter = SentenceSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.get_nodes_from_documents(documents)