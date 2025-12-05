"""
Research Agent - Information Gatherer
Handles RAG search over documents and web search.
"""

from crewai import Agent


class ResearchAgent:
    """
    Research Agent (Information Gatherer)
    - RAG search over uploaded documents + pre-loaded ML textbooks
    - Web search via DuckDuckGo
    - Returns cited sources
    """
    
    def __init__(self, llm, tools=None):
        self.llm = llm
        self.tools = tools or []
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create and configure the Research Agent."""
        return Agent(
            role="ML Research Specialist",
            goal="Find accurate and relevant information about Machine Learning topics from documents and web sources",
            backstory="""You are a research specialist with expertise in Machine Learning.
            You have access to:
            - A knowledge base of ML textbooks and documents (RAG search)
            - Web search capabilities for latest information
            
            Your job is to find accurate, relevant information and always cite your sources.
            You provide factual, well-researched answers with proper references.""",
            llm=self.llm,
            tools=self.tools,
            verbose=True
        )
    
    def get_agent(self) -> Agent:
        """Return the CrewAI agent instance."""
        return self.agent
