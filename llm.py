"""
llm.py

Creates the Groq LLM used by the RAG assistant.
"""

from langchain_groq import ChatGroq

from config import GROQ_API_KEY


class LLMManager:
    """
    Creates and manages the Groq LLM.
    """

    def __init__(
        self,
        model_name: str = "openai/gpt-oss-120b",
        temperature: float = 0.2,
    ):
        """
        Parameters
        ----------
        model_name : str
            Name of the Groq model.

        temperature : float
            Controls randomness.
        """

        self.llm = ChatGroq(
            api_key=GROQ_API_KEY,
            model=model_name,
            temperature=temperature,
        )

    def get_llm(self):
        """
        Return the initialized LLM.
        """

        return self.llm