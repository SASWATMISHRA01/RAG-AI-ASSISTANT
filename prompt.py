"""
prompt.py

Defines the prompt template used by the RAG pipeline.
"""

from langchain_core.prompts import ChatPromptTemplate


RAG_PROMPT = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Use ONLY the information provided in the context below to answer the user's question.

If the answer is not present in the context, reply exactly:

"I couldn't find the answer in the provided document."

Do not make up facts.
Do not use outside knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)