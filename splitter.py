"""
splitter.py

Splits documents into smaller chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


class DocumentSplitter:

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100
    ):
        """
        Parameters
        ----------
        chunk_size : int
            Maximum characters per chunk.

        chunk_overlap : int
            Overlapping characters.
        """

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )

    def split(
        self,
        documents: list[Document]
    ) -> list[Document]:
        """
        Split documents into chunks.

        Parameters
        ----------
        documents : list[Document]

        Returns
        -------
        list[Document]
        """

        chunks = self.text_splitter.split_documents(documents)

        return chunks