#!/usr/bin/env python
"""
Main entry point for ML Learning Assistant CLI
"""

from .crew import MLLearningAssistantCrew


def run():
    """Run the ML Learning Assistant crew with example question."""
    crew = MLLearningAssistantCrew()
    
    print("=" * 60)
    print("ML LEARNING ASSISTANT - Example Run")
    print("=" * 60)
    
    # Example: Ask a question
    result = crew.ask_question(
        query="Explain gradient descent and how it's used in neural networks",
        topic="gradient descent"
    )
    print("\n" + "=" * 60)
    print("RESULT:")
    print("=" * 60)
    print(result)


def run_quiz():
    """Generate a quiz example."""
    crew = MLLearningAssistantCrew()
    
    print("=" * 60)
    print("ML LEARNING ASSISTANT - Quiz Generation")
    print("=" * 60)
    
    result = crew.generate_quiz(topic="neural networks", num_questions=5)
    print("\n" + "=" * 60)
    print("QUIZ:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    run()
