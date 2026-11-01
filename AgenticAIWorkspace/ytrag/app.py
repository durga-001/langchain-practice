from src.data_loader import load_all_documents
from src.vectorstore import FaissVectorStore
from src.search import RAGSearch
from src.embedding import EmbeddingPipeline
# Example usage
if __name__ == "__main__":
    docs=load_all_documents("data")
    # chunks=EmbeddingPipeline().chunk_documents(docs)
    # chunkvectors=EmbeddingPipeline().embed_chunks(chunks)
    store=FaissVectorStore("faiss_store")
    # store.build_from_documents(docs)
    store.load()
    print(store.query("What is attention mechanism?", top_k=3))
    print(chunkvectors)
    print(docs)