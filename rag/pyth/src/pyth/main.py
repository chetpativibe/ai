from pyth.ingest import load_documents, chunk_documents
from pyth.embedding import embed_text, embed_texts
from pyth.index import create_index
from pyth.retrieval import retrieve_relevant_chunks
from pyth.generation import generate_answer

def main() -> None:
    documents = load_documents()
    chunks = chunk_documents(documents=documents)
    # print(f"Number of chunks : {len(chunks)}")

    # for i, chunk in enumerate(chunks[:3]):
    #      print(f"\nChunk {i + 1}")
    #      print(f"Words: {len(chunk.text.split())} words (approx.)")
    #      print(f"Text:\n{chunk.text[:300]}")

    # for chunk in chunks[:1]:
    #     vector = embed_text(text=chunk.text)
    #     print(f"Embedding dimensions: {len(vector)}")

    texts: list[str] = []
    # texts = []
    for chunk in chunks:    
        texts.append(chunk.text)
    # texts = [chunk.text for chunk in chunks]
    vectors = embed_texts(texts)

    # print(f"Number of embeddings: {len(vectors)}")
    # print(f"Embedding dimensions: {len(vectors[0])}")

    for chunk, vector in zip(chunks, vectors):
        chunk.embedding = vector

    # print(len(chunks[0].embedding))

    index = create_index(chunks)
    # print(f"Indexed nodes: {len(index.docstore.docs)}")

    question = "What did Paul Graham do before starting Y Combinator?"
    results = retrieve_relevant_chunks(index=index, question=question)

    print(f"Question: {question}")
    for result in results:
        print(f"Similarity score: {result.score}\nRetrieved chunk text:\n{result.node.text}\n\n")

    # texts = []
    # for result in results:
    #     texts.append(result.node.text)
    # context = "\n\n".join(texts)
    context = "\n\n".join(result.node.text for result in results)

    print(f"\n\nContext:\n{context}")

    answer = generate_answer(question=question, context=context)

    print(f"\n\nLLM generated answer:\n{answer}")

if __name__ == "__main__":
    main()