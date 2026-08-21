"""
embeddings.py

Creates the embedding model used by the RAG pipeline.
"""

from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    """
    Wrapper around the Hugging Face embedding model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        """
        Initialize the embedding model.
        """

        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name
        )

    def get_model(self):
        """
        Return the embedding model.
        """

        return self.embeddings