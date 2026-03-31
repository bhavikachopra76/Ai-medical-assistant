from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


class MedicalVectorStore:
    def __init__(self, save_path="backend/faiss_index"):
        self.save_path = save_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    def build_and_save(self, structured_chunks: list[dict]):
        """
        Converts our structured chunks into Vectors and saves the database to disk.
        """
        print(f"Converting {len(structured_chunks)} chunks into vectors...")

        # 1. Convert our dictionaries into LangChain Document objects
        documents = []
        for chunk in structured_chunks:
            # We embed the "enriched" text so the search engine has the full context
            doc = Document(
                page_content=chunk["enriched"],
                metadata=chunk["metadata"]
            )
            documents.append(doc)

        # 2. Build the FAISS vector database
        vector_db = FAISS.from_documents(documents, self.embeddings)

        # 3. Save it to your hard drive
        vector_db.save_local(self.save_path)
        print(f"Vector database successfully saved to {self.save_path}")

        return vector_db

    def load_existing(self):
        """
        Loads a previously saved vector database from disk.
        """
        if os.path.exists(self.save_path):
            return FAISS.load_local(
                self.save_path,
                self.embeddings,
                allow_dangerous_deserialization=True  # Required for local FAISS loading
            )
        else:
            raise FileNotFoundError("No Vector Store found. Please build it first.")
