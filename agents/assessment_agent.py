"""
Assessment Agent - Quiz Master
Creates quizzes and validates answers.
"""

from crewai import Agent


class AssessmentAgent:
    """
    Assessment Agent (Quiz Master)
    - Creates MCQ and True/False questions
    - Validates answers with explanations
    - Adaptive difficulty based on user responses
    """
    
    def __init__(self, llm):
        self.llm = llm
        self.agent = self._create_agent()
    
    def _create_agent(self) -> Agent:
        """Create and configure the Assessment Agent."""
        return Agent(
            role="ML Assessment Specialist",
            goal="Create effective quiz questions and provide helpful feedback on answers",
            backstory="""You are an assessment specialist for Machine Learning education.
            Your expertise includes:
            - Creating Multiple Choice Questions (MCQ) that test understanding
            - Designing True/False questions that address common misconceptions
            - Providing detailed explanations for correct and incorrect answers
            - Adapting question difficulty based on student performance
            
            You create questions that are:
            - Clear and unambiguous
            - Testing real understanding, not just memorization
            - At appropriate difficulty for university students
            - Educational even when answered incorrectly""",
            llm=self.llm,
            verbose=True
        )
    
    def get_agent(self) -> Agent:
        """Return the CrewAI agent instance."""
        return self.agent
