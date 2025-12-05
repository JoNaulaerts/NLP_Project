"""
RAG Tool - Retrieval Augmented Generation
Searches over indexed documents using vector similarity.
"""

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class RAGToolInput(BaseModel):
    """Input schema for RAG Tool."""
    query: str = Field(..., description="The search query to find relevant documents")


class RAGTool(BaseTool):
    """
    RAG Tool for searching over indexed ML documents and textbooks.
    Uses LlamaIndex for document processing and ChromaDB for vector storage.
    """
    name: str = "rag_search"
    description: str = """Search through the indexed ML textbooks and documents.
    Use this tool when you need to find specific information about Machine Learning concepts
    from the knowledge base. Returns relevant passages with source citations."""
    args_schema: Type[BaseModel] = RAGToolInput
    
    def __init__(self, index=None, **kwargs):
        super().__init__(**kwargs)
        self._index = index
    
    def _run(self, query: str) -> str:
        """Execute RAG search over indexed documents."""
        if self._index is None:
            return "RAG index not initialized. Please load documents first."
        
        try:
            # Query the index
            query_engine = self._index.as_query_engine()
            response = query_engine.query(query)
            
            # Format response with sources
            result = f"**Answer:**\n{response.response}\n\n"
            
            if response.source_nodes:
                result += "**Sources:**\n"
                for i, node in enumerate(response.source_nodes, 1):
                    source = node.node.metadata.get("file_name", "Unknown")
                    result += f"{i}. {source}\n"
            
            return result
            
        except Exception as e:
            return f"Error searching documents: {str(e)}"
