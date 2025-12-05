"""
Planning Agent - Coordinator
Routes user queries to appropriate agents and manages workflow.
"""

from crewai import Agent


class PlanningAgent:
    """
    Planning Agent (Coordinator)
    - Routes user queries to appropriate agents
    - Manages workflow and response aggregation
    - Maintains conversation context
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create and configure the Planning Agent."""
        return Agent(
            role="Planning Coordinator",
            goal="Analyze user queries and route them to the appropriate specialized agents",
            backstory="""You are an expert coordinator for a Machine Learning learning platform.
            Your job is to understand what the user needs and delegate tasks to the right agents:
            - Conceptual questions → Research + Educational Agents
            - Quiz requests → Assessment Agent
            - Code examples → Research + Educational Agents
            You ensure smooth workflow and aggregate responses effectively.""",
            llm=self.llm,
            verbose=True,
            allow_delegation=True
        )
    
    def get_agent(self) -> Agent:
        """Return the CrewAI agent instance."""
        return self.agent
