"""
Web Search Tool - DuckDuckGo Integration
Searches the web for ML-related information.
"""

from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field


class WebSearchInput(BaseModel):
    """Input schema for Web Search Tool."""
    query: str = Field(..., description="The search query for web search")
    max_results: int = Field(default=5, description="Maximum number of results to return")


class WebSearchTool(BaseTool):
    """
    Web Search Tool using DuckDuckGo API.
    Searches the web for Machine Learning related information.
    """
    name: str = "web_search"
    description: str = """Search the web for Machine Learning information.
    Use this tool when you need current information or topics not covered in the document base.
    Returns web search results with titles, snippets, and URLs."""
    args_schema: Type[BaseModel] = WebSearchInput
    
    def _run(self, query: str, max_results: int = 5) -> str:
        """Execute web search using DuckDuckGo."""
        try:
            from duckduckgo_search import DDGS
            
            results = []
            with DDGS() as ddgs:
                search_results = list(ddgs.text(query, max_results=max_results))
                
                for i, result in enumerate(search_results, 1):
                    results.append(
                        f"{i}. **{result.get('title', 'No title')}**\n"
                        f"   {result.get('body', 'No description')}\n"
                        f"   URL: {result.get('href', 'No URL')}\n"
                    )
            
            if results:
                return "**Web Search Results:**\n\n" + "\n".join(results)
            else:
                return "No results found for the query."
                
        except ImportError:
            return "DuckDuckGo search library not installed. Run: pip install duckduckgo-search"
        except Exception as e:
            return f"Error performing web search: {str(e)}"
