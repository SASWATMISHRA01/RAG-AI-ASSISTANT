"""
vectorstore.py

Creates, saves, loads, and manages the FAISS vector database.
"""

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document


class VectorStoreManager:
    """
    Manages the FAISS vector database.
    """

    def __init__(self, embedding_model, db_path: str = "vector_db"):
        """
        Parameters
        ----------
        embedding_model
            HuggingFace embedding model.

        db_path : str
            Folder where the FAISS index is stored.
        """

        self.embedding_model = embedding_model
        self.db_path = db_path

    def create_vectorstore(self, documents: list[Document]):
        """
        Create a FAISS vector database from documents.
        """

        vectorstore = FAISS.from_documents(
            documents=documents,
            embedding=self.embedding_model
        )

        return vectorstore

    def save_vectorstore(self, vectorstore):
        """
        Save the FAISS database to disk.
        """

        Path(self.db_path).mkdir(exist_ok=True)

        vectorstore.save_local(self.db_path)

    def load_vectorstore(self):
        """
        Load the FAISS database from disk.
        """

        vectorstore = FAISS.load_local(
            self.db_path,
            self.embedding_model,
            allow_dangerous_deserialization=True
        )

        return vectorstore