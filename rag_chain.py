"""
rag_chain.py

Creates the complete RAG pipeline.
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.prompt import RAG_PROMPT


def format_docs(documents):
    """
    Combine retrieved documents into one string.
    """

    return "\n\n".join(
        doc.page_content
        for doc in documents
    )


class RAGChain:

    def __init__(
        self,
        retriever,
        llm
    ):

        self.chain = (
            {
                "context": retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | RAG_PROMPT
            | llm
            | StrOutputParser()
        )

    def ask(self, question: str):

        return self.chain.invoke(question)