"""
retriever.py

Creates a retriever from the FAISS vector store.
"""

from langchain_core.documents import Document


class RetrieverManager:
    """
    Creates and uses a retriever.
    """

    def __init__(
        self,
        vectorstore,
        k: int = 3
    ):
        """
        Parameters
        ----------
        vectorstore
            FAISS vector database.

        k : int
            Number of chunks to retrieve.
        """

        self.retriever = vectorstore.as_retriever(
            search_kwargs={
                "k": k
            }
        )

    def retrieve(
        self,
        question: str
    ) -> list[Document]:
        """
        Retrieve relevant document chunks.

        Parameters
        ----------
        question : str

        Returns
        -------
        list[Document]
        """

        documents = self.retriever.invoke(question)

        return documents