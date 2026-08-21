"""
app.py

Interactive Streamlit application for the RAG AI Assistant.
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader

from src.splitter import DocumentSplitter
from src.embeddings import EmbeddingModel
from src.vectorstore import VectorStoreManager
from src.retriever import RetrieverManager
from src.llm import LLMManager
from src.rag_chain import RAGChain


# ============================================================
# ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .info-card {
        padding: 18px;
        border-radius: 15px;
        background-color: #1f2937;
        border: 1px solid #374151;
        text-align: center;
        margin-bottom: 15px;
    }

    .info-number {
        font-size: 28px;
        font-weight: 700;
    }

    .info-label {
        color: #9ca3af;
        font-size: 14px;
    }

    .source-box {
        padding: 12px;
        border-radius: 10px;
        background-color: #111827;
        border-left: 4px solid #10b981;
        margin-top: 10px;
        margin-bottom: 10px;
    }

    .welcome-box {
        padding: 20px;
        border-radius: 15px;
        background-color: #1f2937;
        border: 1px solid #374151;
        margin-bottom: 20px;
    }

    .footer {
        text-align: center;
        color: #9ca3af;
        padding: 30px;
        margin-top: 40px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "rag" not in st.session_state:
    st.session_state.rag = None

if "pdf_info" not in st.session_state:
    st.session_state.pdf_info = None

if "document_loaded" not in st.session_state:
    st.session_state.document_loaded = False

if "suggested_question" not in st.session_state:
    st.session_state.suggested_question = None


# ============================================================
# WELCOME / GREETING
# ============================================================

st.title("🤖 RAG AI Assistant")

st.caption(
    "Chat with your documents using Retrieval-Augmented Generation."
)

if not st.session_state.document_loaded:

    st.info(
        """
        👋 **Hello! Welcome to your RAG AI Assistant!**

        I'm ready to help you understand your documents.

        **Getting started is easy:**

        1. 📂 Upload a PDF using the sidebar.
        2. 🧠 I'll process and understand the document.
        3. 💬 Ask me questions about the PDF.
        4. 🤖 I'll generate answers using information from your document.
        """
    )

else:

    st.success(
        f"""
        👋 **Welcome back!**

        I'm ready to answer questions about
        **{st.session_state.pdf_info['name']}**.

        What would you like to know?
        """
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📂 Document")

    uploaded_file = st.file_uploader(
        "Upload your PDF",
        type=["pdf"],
        help="Upload a PDF document to start asking questions.",
    )

    st.divider()

    st.subheader("⚙️ System")

    st.write("**LLM:** GPT-OSS 120B")
    st.write("**Embeddings:** all-MiniLM-L6-v2")
    st.write("**Vector Database:** FAISS")
    st.write("**Framework:** LangChain")

    st.divider()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):

        st.session_state.messages = []

        st.rerun()

    if st.button(
        "🔄 Reset Document",
        use_container_width=True,
    ):

        st.session_state.rag = None
        st.session_state.pdf_info = None
        st.session_state.document_loaded = False
        st.session_state.messages = []

        st.rerun()


# ============================================================
# PROCESS UPLOADED PDF
# ============================================================

if uploaded_file is not None:

    # Process only when a new document is uploaded
    if (
        not st.session_state.document_loaded
        or st.session_state.pdf_info["name"]
        != uploaded_file.name
    ):

        with st.status(
            "📚 Processing your PDF...",
            expanded=True,
        ) as status:

            pdf_path = None

            try:

                # ------------------------------------------------
                # Save uploaded PDF temporarily
                # ------------------------------------------------

                st.write("📥 Reading PDF...")

                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=".pdf",
                ) as temp_file:

                    temp_file.write(
                        uploaded_file.getbuffer()
                    )

                    pdf_path = temp_file.name

                # ------------------------------------------------
                # Load PDF
                # ------------------------------------------------

                st.write("📄 Extracting document text...")

                loader = PyPDFLoader(pdf_path)

                documents = loader.load()

                # ------------------------------------------------
                # Split document
                # ------------------------------------------------

                st.write(
                    "✂️ Splitting document into chunks..."
                )

                splitter = DocumentSplitter()

                chunks = splitter.split(documents)

                # ------------------------------------------------
                # Create embeddings
                # ------------------------------------------------

                st.write(
                    "🧠 Creating document embeddings..."
                )

                embedding_model = (
                    EmbeddingModel()
                    .get_model()
                )

                # ------------------------------------------------
                # Create vector database
                # ------------------------------------------------

                st.write(
                    "🔎 Building FAISS vector database..."
                )

                vector_manager = VectorStoreManager(
                    embedding_model
                )

                vectorstore = (
                    vector_manager
                    .create_vectorstore(chunks)
                )

                # ------------------------------------------------
                # Retriever
                # ------------------------------------------------

                st.write(
                    "🔍 Creating document retriever..."
                )

                retriever_manager = (
                    RetrieverManager(vectorstore)
                )

                retriever = (
                    retriever_manager.retriever
                )

                # ------------------------------------------------
                # LLM
                # ------------------------------------------------

                st.write(
                    "🤖 Connecting to Groq..."
                )

                llm_manager = LLMManager()

                llm = (
                    llm_manager
                    .get_llm()
                )

                # ------------------------------------------------
                # RAG Chain
                # ------------------------------------------------

                st.write(
                    "🔗 Building RAG pipeline..."
                )

                rag = RAGChain(
                    retriever,
                    llm,
                )

                # ------------------------------------------------
                # Save in session state
                # ------------------------------------------------

                st.session_state.rag = rag

                st.session_state.pdf_info = {
                    "name": uploaded_file.name,
                    "pages": len(documents),
                    "chunks": len(chunks),
                    "size": round(
                        uploaded_file.size / (1024 * 1024),
                        2,
                    ),
                }

                st.session_state.document_loaded = True

                st.session_state.messages = []

                # ------------------------------------------------
                # Delete temporary PDF
                # ------------------------------------------------

                if pdf_path and os.path.exists(pdf_path):
                    os.unlink(pdf_path)

                status.update(
                    label="✅ PDF processed successfully!",
                    state="complete",
                    expanded=False,
                )

            except Exception as e:

                if pdf_path and os.path.exists(pdf_path):
                    os.unlink(pdf_path)

                status.update(
                    label="❌ PDF processing failed",
                    state="error",
                )

                st.error(
                    f"Something went wrong: {str(e)}"
                )


# ============================================================
# DOCUMENT INFORMATION
# ============================================================

if st.session_state.document_loaded:

    info = st.session_state.pdf_info

    st.success(
        f"📄 **{info['name']}** is ready. "
        "Ask me anything about it!"
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "📄 Pages",
            info["pages"],
        )

    with col2:

        st.metric(
            "🧩 Chunks",
            info["chunks"],
        )

    with col3:

        st.metric(
            "💾 Size",
            f"{info['size']} MB",
        )

    with col4:

        st.metric(
            "🤖 Status",
            "Ready",
        )


# ============================================================
# SUGGESTED QUESTIONS
# ============================================================

if st.session_state.document_loaded:

    st.subheader("💡 Try asking")

    suggestions = [
        "Give me a summary of this document.",
        "What are the main topics discussed?",
        "Explain the most important concept.",
        "What are the key points I should remember?",
    ]

    cols = st.columns(4)

    for i, suggestion in enumerate(suggestions):

        with cols[i]:

            if st.button(
                suggestion,
                key=f"suggestion_{i}",
                use_container_width=True,
            ):

                st.session_state.suggested_question = (
                    suggestion
                )


# ============================================================
# CHAT SECTION
# ============================================================

if st.session_state.document_loaded:

    st.subheader("💬 Chat with your document")

    st.caption(
        "Ask anything about the uploaded PDF."
    )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander(
                "📚 View Sources"
            ):

                for source in message["sources"]:

                    st.markdown(
                        f"""
                        <div class="source-box">
                        📄 Page {source}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# CHAT INPUT
# ============================================================

prompt = st.chat_input(
    "💬 Ask me anything about your document..."
)


# ============================================================
# HANDLE SUGGESTED QUESTION
# ============================================================

if st.session_state.suggested_question:

    prompt = (
        st.session_state.suggested_question
    )

    st.session_state.suggested_question = None


# ============================================================
# PROCESS QUESTION
# ============================================================

if prompt:

    if not st.session_state.document_loaded:

        st.warning(
            "📂 Please upload a PDF first."
        )

        st.stop()

    # --------------------------------------------------------
    # USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    with st.chat_message("user"):

        st.markdown(prompt)

    # --------------------------------------------------------
    # AI RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤔 Searching the document and generating an answer..."
        ):

            try:

                answer = st.session_state.rag.ask(
                    prompt
                )

                st.markdown(answer)

                # ------------------------------------------------
                # Save assistant response
                # ------------------------------------------------

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                    }
                )

                # ------------------------------------------------
                # Download answer
                # ------------------------------------------------

                st.download_button(
                    "📥 Download Answer",
                    data=answer,
                    file_name="rag_answer.txt",
                    mime="text/plain",
                )

            except Exception as e:

                st.error(
                    f"❌ Error while generating answer: {str(e)}"
                )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">

    🤖 <b>RAG AI Assistant</b><br>

    Built with Streamlit • LangChain • FAISS •
    HuggingFace Embeddings • Groq

    <br><br>

    Developed by <b>Saswat Mishra</b>

    </div>
    """,
    unsafe_allow_html=True,
)