"""
CrewAI Configuration
Configures the multi-agent crew with all agents and tasks.
"""

from crewai import Crew, Task, Process
from typing import Optional

from agents import PlanningAgent, ResearchAgent, EducationalAgent, AssessmentAgent
from tools import RAGTool, WebSearchTool, DocumentProcessor


class MLLearningCrew:
    """
    Multi-Agent ML Learning Assistant Crew
    Orchestrates 4 agents for educational ML assistance.
    """
    
    def __init__(self, llm=None):
        """
        Initialize the ML Learning Crew.
        
        Args:
            llm: Language model to use for all agents (Llama 3.2)
        """
        self.llm = llm
        self.doc_processor = DocumentProcessor()
        self.doc_processor.setup_directories()
        
        # Initialize tools
        self.rag_tool = None
        self.web_search_tool = WebSearchTool()
        
        # Initialize agents
        self._init_agents()
        
        # Crew instance
        self.crew = None
        
    def _init_agents(self):
        """Initialize all agents."""
        self.planning_agent = PlanningAgent(self.llm)
        self.research_agent = ResearchAgent(
            self.llm, 
            tools=[self.web_search_tool]  # RAG tool added after index creation
        )
        self.educational_agent = EducationalAgent(self.llm)
        self.assessment_agent = AssessmentAgent(self.llm)
        
    def load_documents(self):
        """Load and index documents for RAG."""
        documents = self.doc_processor.load_documents()
        if documents:
            index = self.doc_processor.create_index(documents)
            self.rag_tool = RAGTool(index=index)
            # Add RAG tool to research agent
            self.research_agent = ResearchAgent(
                self.llm,
                tools=[self.rag_tool, self.web_search_tool]
            )
    
    def create_research_task(self, query: str) -> Task:
        """Create a research task for the given query."""
        return Task(
            description=f"""Research the following Machine Learning topic:
            
            Query: {query}
            
            Use the RAG search tool to find relevant information from the knowledge base.
            If needed, supplement with web search for current information.
            
            Provide accurate information with citations.""",
            expected_output="Detailed research findings with source citations",
            agent=self.research_agent.get_agent()
        )
    
    def create_educational_task(self, query: str) -> Task:
        """Create an educational content task."""
        return Task(
            description=f"""Create educational content for the following topic:
            
            Topic: {query}
            
            Based on the research findings, create a clear and comprehensive explanation.
            Target audience: University students learning Machine Learning.
            
            Include:
            - Clear explanation of concepts
            - Examples where appropriate
            - Mathematical foundations if relevant
            - Practical applications""",
            expected_output="Clear, educational explanation suitable for university students",
            agent=self.educational_agent.get_agent()
        )
    
    def create_quiz_task(self, topic: str, num_questions: int = 5) -> Task:
        """Create a quiz generation task."""
        return Task(
            description=f"""Create a quiz about the following Machine Learning topic:
            
            Topic: {topic}
            Number of questions: {num_questions}
            
            Create a mix of:
            - Multiple Choice Questions (MCQ) with 4 options
            - True/False questions
            
            For each question:
            - Make it test real understanding
            - Include the correct answer
            - Provide an explanation for why the answer is correct""",
            expected_output=f"{num_questions} quiz questions with answers and explanations",
            agent=self.assessment_agent.get_agent()
        )
    
    def run_learning_query(self, query: str) -> str:
        """
        Process a learning query through the crew.
        
        Args:
            query: User's question or request.
            
        Returns:
            Aggregated response from the agents.
        """
        # Determine query type and create appropriate tasks
        tasks = []
        
        # Research task first
        research_task = self.create_research_task(query)
        tasks.append(research_task)
        
        # Educational task to explain the research
        educational_task = self.create_educational_task(query)
        tasks.append(educational_task)
        
        # Create crew with sequential process
        self.crew = Crew(
            agents=[
                self.research_agent.get_agent(),
                self.educational_agent.get_agent()
            ],
            tasks=tasks,
            process=Process.sequential,
            verbose=True
        )
        
        # Execute the crew
        result = self.crew.kickoff()
        return str(result)
    
    def run_quiz(self, topic: str, num_questions: int = 5) -> str:
        """
        Generate a quiz on the given topic.
        
        Args:
            topic: ML topic for the quiz.
            num_questions: Number of questions to generate.
            
        Returns:
            Generated quiz with questions and answers.
        """
        quiz_task = self.create_quiz_task(topic, num_questions)
        
        self.crew = Crew(
            agents=[self.assessment_agent.get_agent()],
            tasks=[quiz_task],
            process=Process.sequential,
            verbose=True
        )
        
        result = self.crew.kickoff()
        return str(result)
