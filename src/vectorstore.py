from pathlib import Path
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

def create_vectorstore(
        embedding_model: str,
        chunks: list[Document],
        persist_directory: str,
        ollama_url: str = "http://127.0.0.1:11434"
) -> Chroma:
    embeddings = OllamaEmbeddings(
        model=embedding_model,
        base_url=ollama_url
    )
    if Path(persist_directory).exists():
        print("Loading existing vector store")
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
    print("No vector store detected, creating a new one...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    return vectorstore