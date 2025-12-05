"""
Educational Agent - Content Creator
Generates explanations, summaries, and educational content.
"""

from crewai import Agent


class EducationalAgent:
    """
    Educational Agent (Content Creator)
    - Generates explanations and summaries
    - Adapts content for university level
    - Combines insights from Research Agent
    - Tutoring and clarification
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create and configure the Educational Agent."""
        return Agent(
            role="ML Education Expert",
            goal="Create clear, comprehensive educational content about Machine Learning concepts",
            backstory="""You are an experienced Machine Learning educator specializing in 
            university-level content. Your expertise includes:
            - Breaking down complex ML concepts into understandable explanations
            - Creating examples and analogies that resonate with students
            - Adapting content difficulty to the learner's level
            - Providing step-by-step explanations with mathematical foundations
            
            You work with the Research Agent's findings to create educational content
            that is accurate, engaging, and pedagogically sound.""",
            llm=self.llm,
            verbose=True
        )
    
    def get_agent(self) -> Agent:
        """Return the CrewAI agent instance."""
        return self.agent
