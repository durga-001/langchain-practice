
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

from src.vectorstore import FaissVectorStore
from src.data_loader import load_all_documents

load_dotenv()


class RAGSearch:

    def __init__(
        self,
        persist_dir: str = "faiss_store",
        embedding_model: str = "all-MiniLM-L6-v2",
        llm_model: str = "llama-3.3-70b-versatile"
    ):

        self.vectorstore = FaissVectorStore(
            persist_dir=persist_dir,
            embedding_model=embedding_model
        )

        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")

        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):

            print("[INFO] Building new vector store...")

            docs = load_all_documents()

            self.vectorstore.build_from_documents(docs)

        else:

            print("[INFO] Loading existing vector store...")

            self.vectorstore.load()

        groq_api_key = os.getenv("GROQ_API_KEY")

        if not groq_api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")

        self.llm = ChatGroq(
            api_key=groq_api_key,
            model=llm_model,
            temperature=0
        )

        print(f"[INFO] Groq LLM initialized: {llm_model}")

    def search_and_summarize(
        self,
        query: str,
        top_k: int = 5
    ) -> str:

        # Generate query embedding
        query_embedding = self.vectorstore.emb_pipe.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")

        # Retrieve relevant documents
        results = self.vectorstore.search(
            query_embedding,
            top_k=top_k
        )

        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)

        if not context:
            return "No relevant documents found."

        prompt = f"""
        Answer the user's query using the following context.

        Query: {query}

        Context:
        {context}

        Instructions:
        - Give a clear and accurate answer.
        - Use only the provided context.
        - If the answer is not available, say so.
        """

        response = self.llm.invoke(prompt)

        return response.content
