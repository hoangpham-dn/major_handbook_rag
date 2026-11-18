import os
import streamlit as st
from pathlib import Path

from loader import load_pdf_files
from chunker import chunk_documents
from vectorstore import create_vectorstore
from rag import ask

from langchain_ollama import OllamaLLM


# 1. CONFIGURATIONS
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / "data" / "major_handbooks"
VECTOR_STORE = BASE_DIR.parent / "data" / "vector_store"

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

EMBED_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3.5:4b-mlx"

TEMPERATURE = 0.1
OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://127.0.0.1:11434"
)

TOP_K = 10
SCORE_THRESHOLD = 0.3


# 2. INITIALIZE RAG BACKEND
@st.cache_resource
def initialize_rag():

    documents = load_pdf_files(directory=str(DATA_DIR))

    chunks = chunk_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )

    vectorstore = create_vectorstore(
        embedding_model=EMBED_MODEL,
        chunks=chunks,
        persist_directory=str(VECTOR_STORE),
        ollama_url=OLLAMA_URL
    )

    llm = OllamaLLM(
        model=LLM_MODEL,
        base_url=OLLAMA_URL,
        temperature=TEMPERATURE,
        reasoning=False
    )

    return vectorstore, llm

st.set_page_config(
    page_title="History Major Advisor",
    page_icon="📕",
    layout="centered"
)

st.title("History Major Advisor")
st.caption("Ask me questions about the History major at Fulbright University." +
           "\n Disclaimer: I do not replace your actual major advisor.")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.header("RAG Settings")
    st.write(f"**LLM:** {LLM_MODEL}")
    st.write(f"**Embedding:** {EMBED_MODEL}")
    st.write(f"**Top K:** {TOP_K}")
    st.write(f"**Score threshold:** {SCORE_THRESHOLD}")

    if st.button("Clear conversation"):
        st.session_state.chat_history = []
        st.session_state.messages = []
        st.rerun()
try:
    with st.spinner("Initializing RAG system..."):
        db, llm = initialize_rag()

except Exception as e:
    st.error(f"Failed to initialize RAG system: {e}")
    st.stop()


# 7. DISPLAY PREVIOUS MESSAGES
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display sources for assistant messages
        if message["role"] == "assistant":
            sources = message.get("sources", [])
            if sources:
                with st.expander("Retrieved sources"):
                    for i, source in enumerate(sources, 1):
                        st.markdown(
                            f"**{i}. {source['source']}**"
                        )
                        st.caption(
                            f"Similarity: {source['score']}"
                        )
                        st.write(source["content"])


question = st.chat_input("Ask a question about the History major...")
if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })
    with st.chat_message("user"):
        st.markdown(question)

    # Generate assistant response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = ask(
                question=question,
                db=db,
                llm=llm,
                top_k=TOP_K,
                score_threshold=SCORE_THRESHOLD,
                chat_history=st.session_state.chat_history
            )

        answer = response["answer"]
        sources = response["sources"]
        st.markdown(answer)

        if sources:
            with st.expander("Retrieved sources"):
                for i, source in enumerate(sources, 1):
                    st.markdown(
                        f"**{i}. {source['source']}**"
                    )
                    st.caption(
                        f"Similarity: {source['score']}"
                    )
                    st.write(source["content"])
        elif response["used_fallback"]:
            st.caption(
                "No sufficiently relevant documents found."
            )
    # Update conversation history
    st.session_state.chat_history = response["chat_history"]
    # Save assistant message for future reruns
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })
