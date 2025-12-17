"""Direct ChromaDB RAG tool (bypasses MCP gateway issues)"""
import os
from typing import List, Optional
from crewai.tools import BaseTool
from pydantic import BaseModel, Field
import chromadb


class ChromaQueryInput(BaseModel):
    query: str = Field(..., description="Search query text")
    n_results: int = Field(default=5, description="Number of results to return")


class ChromaRAGTool(BaseTool):
    name: str = "chroma_rag_search"
    description: str = "Search the ML materials knowledge base for relevant information. Use this for questions about uploaded course materials, lecture notes, or previously indexed documents."
    args_schema: type[BaseModel] = ChromaQueryInput
    
    def _run(self, query: str, n_results: int = 5) -> str:
        """Execute RAG search against ChromaDB"""
        try:
            host = os.getenv("CHROMA_HOST", "chromadb")
            port = int(os.getenv("CHROMA_PORT", "8000"))
            collection_name = os.getenv("CHROMA_COLLECTION", "ml_materials")
            
            client = chromadb.HttpClient(host=host, port=port)
            collection = client.get_collection(name=collection_name)
            
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            if not results["documents"] or not results["documents"][0]:
                return "No relevant information found in the knowledge base."
            
            # Format results
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            dists = results["distances"][0]
            
            formatted = []
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, dists), 1):
                source = meta.get("source", "unknown")
                page = meta.get("page", "?")
                formatted.append(
                    f"[Result {i}] (source: {source}, page: {page}, relevance: {1-dist:.2f})\n{doc}\n"
                )
            
            return "\n".join(formatted)
            
        except Exception as e:
            return f"Error searching knowledge base: {str(e)}"
