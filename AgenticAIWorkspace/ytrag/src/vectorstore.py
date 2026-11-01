
import os
import pickle
import faiss
import numpy as np

from typing import List, Any
from src.embedding import EmbeddingPipeline


class FaissVectorStore:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ):

        self.persist_dir = persist_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self.index = None
        self.metadata = []

        self.embedding_model = embedding_model
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.emb_pipe = EmbeddingPipeline(
            model_name=embedding_model,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

        print(f"[INFO] Loaded embedding model: {embedding_model}")

    def build_from_documents(self, documents: List[Any]):

        print(f"[INFO] Building vector store from {len(documents)} raw documents...")

        chunks = self.emb_pipe.chunk_documents(documents)

        embeddings = self.emb_pipe.embed_chunks(chunks)

        metadatas = [
            {"text": chunk.page_content, "source": chunk.metadata}
            for chunk in chunks
        ]

        self.add_embeddings(
            np.array(embeddings).astype("float32"),
            metadatas
        )

        self.save()

        print(f"[INFO] Vector store built and saved to {self.persist_dir}")

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadatas: List[Any] = None
    ):

        dim = embeddings.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatL2(dim)

        elif self.index.d != dim:
            raise ValueError("Embedding dimension does not match FAISS index")

        if metadatas is not None and len(metadatas) != embeddings.shape[0]:
            raise ValueError("Number of embeddings and metadata entries must match")

        self.index.add(embeddings)

        if metadatas:
            self.metadata.extend(metadatas)

        print(f"[INFO] Added {embeddings.shape[0]} vectors to FAISS index")

    def save(self):

        if self.index is None:
            raise ValueError("Cannot save an empty FAISS index")

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        faiss.write_index(self.index, faiss_path)

        with open(meta_path, "wb") as f:
            pickle.dump(self.metadata, f)

        print(f"[INFO] Saved FAISS index and metadata to {self.persist_dir}")

    def load(self):

        faiss_path = os.path.join(self.persist_dir, "faiss.index")
        meta_path = os.path.join(self.persist_dir, "metadata.pkl")

        self.index = faiss.read_index(faiss_path)

        with open(meta_path, "rb") as f:
            self.metadata = pickle.load(f)

        if self.index.ntotal != len(self.metadata):
            raise ValueError("FAISS index and metadata count mismatch")

        print(f"[INFO] Loaded FAISS index and metadata from {self.persist_dir}")

    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        D, I = self.index.search(query_embedding, top_k)
        results = []
        for idx, dist in zip(I[0], D[0]):
            meta = self.metadata[idx] if idx < len(self.metadata) else None
            results.append({"index":idx, "distance":dist, "metadata": meta})
        return results

    def query(self, query_text: str, top_k:int=5):
        print(f"[INFO] Querying vector store for: '{query_text}'")
        query_emb = self.model.encode([query_text]).astype('float32')
        return self.search(query_emb, top_k=top_k)