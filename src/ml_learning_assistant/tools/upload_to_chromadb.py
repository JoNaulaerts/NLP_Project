import os
from pathlib import Path
from typing import Dict, Any, List

import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

CHROMA_HOST = os.getenv("CHROMA_HOST", "localhost")
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

# ✅ keep your chosen default
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "ml_materials")


def make_ids(filepath: str, n: int) -> List[str]:
    stem = Path(filepath).stem
    return [f"{stem}_{i}" for i in range(n)]


def upload_pdf_to_chromadb(filepath: str) -> Dict[str, Any]:
    try:
        loader = PyPDFLoader(filepath)
        pages = loader.load()
        if not pages:
            return {"success": False, "message": "No content in PDF.", "chunks": 0, "pages": 0}

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(pages)

        ids = make_ids(filepath, len(chunks))
        documents = [c.page_content for c in chunks]
        metadatas = [{"source": Path(filepath).name, "page": c.metadata.get("page", 0)} for c in chunks]

        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_or_create_collection(name=COLLECTION_NAME)
        collection.add(ids=ids, documents=documents, metadatas=metadatas)

        return {
            "success": True,
            "message": f"Indexed {Path(filepath).name} into {COLLECTION_NAME}",
            "chunks": len(chunks),
            "pages": len(pages),
        }
    except Exception as e:
        return {"success": False, "message": f"Error: {str(e)}", "chunks": 0, "pages": 0}
