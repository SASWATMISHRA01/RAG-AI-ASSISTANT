"""
loader.py

Loads PDF documents using LangChain.
"""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    """
    Loads PDF documents from disk.
    """

    def __init__(self, pdf_path: str):
        """
        Parameters
        ----------
        pdf_path : str
            Path to the PDF file.
        """

        self.pdf_path = pdf_path

    def load(self) -> list[Document]:
        """
        Load the PDF.

        Returns
        -------
        list[Document]
        """

        # Check whether the file exists
        if not Path(self.pdf_path).exists():
            raise FileNotFoundError(
                f"PDF not found: {self.pdf_path}"
            )

        loader = PyPDFLoader(self.pdf_path)

        documents = loader.load()

        return documents