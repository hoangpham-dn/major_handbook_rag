from loader import load_pdf_files
from chunker import chunk_documents
from vectorstore import create_vectorstore
from rag import ask
import argparse
from langchain_ollama import OllamaLLM
from pathlib import Path

# RAG SETTINGS
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR.parent / "data" / "major_handbooks"
VECTOR_STORE = BASE_DIR.parent / "data" / "vector_store"
# COLLECTION = "rag_documents" 
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200 
EMBED_MODEL = "qwen3-embedding:0.6b"
LLM_MODEL = "qwen3.5:2b-mlx" 
TEMPERATURE = 0.1
OLLAMA_URL = "http://127.0.0.1:11434" 
TOP_K = 5
SCORE_THRESHOLD = 0.3 
SAMPLE_QUESTION = "How many students are there at Fulbright?" 



def main():
    parser = argparse.ArgumentParser(description="Simple RAG Q&A (Text Files)")
    parser.add_argument('--question', '-q', default=SAMPLE_QUESTION, help='Question to ask')
    args = parser.parse_args()

    documents = load_pdf_files(directory=str(DATA_DIR))
    chunks = chunk_documents(
        documents=documents,
        chunk_size=CHUNK_SIZE,
        overlap=CHUNK_OVERLAP
    )
    print(f"Loading embeddings. Embeddings use the {EMBED_MODEL} model")
    vectorstore = create_vectorstore(
        embedding_model=EMBED_MODEL,
        chunks=chunks,
        persist_directory=str(VECTOR_STORE)
    )
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=TEMPERATURE, reasoning=False)

    print("Q: " + args.question)
    result = ask(
        question=args.question,
        db=vectorstore,
        llm=llm,
        top_k=TOP_K,
        score_threshold=SCORE_THRESHOLD
    )

    print("\nA: " + result['answer'] + "\n")
    if result['used_fallback']:
        print("(answered from model knowledge — no relevant documents found)")
    else:
        print("Sources:")
        for s in result['sources']:
            print("   " + str(s['source']) + "  score=" + str(s['score']))
            print(f"content:")
            print(s['content'])
    return

if __name__ == "__main__":
    main()