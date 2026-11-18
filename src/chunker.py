import sys
import pymupdf
from pathlib import Path
from typing import Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(
        documents: list[Document],
        chunk_size: int = 1000,
        overlap: int = 200) -> list[Document]:
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = splitter.split_documents(documents)

    return chunks


