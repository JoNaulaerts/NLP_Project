import os
import time
import warnings
from pathlib import Path
from typing import Optional


# Keep these early (helps Streamlit + reduces noisy telemetry behavior)
os.environ["CREWAI_TELEMETRY"] = "false"
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

from crewai import Agent, Crew, Task, Process
from crewai.project import CrewBase, agent, task

from .llm_config import get_llm, get_embeddings_config
from .mcp_servers import docker_mcp_stdio_params

from crewai_tools import MCPServerAdapter
from mcp import StdioServerParameters


@CrewBase
class MLLearningAssistantCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    def __init__(self):
        self.llm = get_llm()
        self.last_request_time = 0
        self._setup_memory_system()
        self.embedder_config = get_embeddings_config()

        # enforce collection default everywhere
        os.environ["CHROMA_COLLECTION"] = os.getenv("CHROMA_COLLECTION", "ml_materials")

        print(f"✅ Crew initialized with LLM: {self.llm.model}")
        print("✅ Memory system: Short-term, Long-term, Entity tracking enabled")
        print("✅ MCP Integration: ChromaDB (RAG) + Tavily (Web Search) via Docker MCP")

    def _setup_memory_system(self):
        storage_dir_env = os.getenv("CREWAI_STORAGE_DIR", "").strip()
        storage_dir = Path(storage_dir_env) if storage_dir_env else Path("./data/crewai_memory")
        storage_dir = storage_dir.resolve()
        storage_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CREWAI_STORAGE_DIR"] = str(storage_dir)
        print(f"📁 Memory storage: {storage_dir}")

    def _rate_limit_check(self):
        current_time = time.time()
        dt = current_time - self.last_request_time

        model_str = str(self.llm.model).lower()
        if "ollama" in model_str:
            min_delay = 0.6
        elif "cerebras" in model_str:
            min_delay = 1.5
        else:
            min_delay = 3.0

        if dt < min_delay:
            time.sleep(min_delay - dt)

        self.last_request_time = time.time()

    def _clean_response(self, response: str) -> Optional[str]:
        if not response:
            return None
        lines = response.split("\n")
        skip_patterns = [
            "Thought:",
            "Action:",
            "Action Input:",
            "Observation:",
            "Final Answer:",
            "I now know the final answer",
        ]
        cleaned_lines = []
        for line in lines:
            if any(p in line for p in skip_patterns):
                continue
            if line.strip():
                cleaned_lines.append(line)
        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned if cleaned else None

    # -----------------------------
    # MCP tools (via adapter) - STRICT allowlist
    # -----------------------------
    def _get_mcp_tools(self):
        """
        IMPORTANT FIX:
        - Do NOT expose MCP management tools (mcp-exec, mcp-add, code-mode, etc.)
          to the LLM. They cause schema mistakes and degrade chat quality.
        - Only expose the tools the Researcher should call directly.
        """
        if getattr(self, "_mcp_tools", None):
            return self._mcp_tools

        params_dict = docker_mcp_stdio_params()
        stdio_params = StdioServerParameters(
            command=params_dict["command"],
            args=params_dict["args"],
            env=dict(os.environ),
        )

        allowed_names = {
            # Chroma (RAG)
            "chroma_query_documents",
            "chroma_get_collection_count",
            "chroma_get_collection_info",
            # Web
            "tavily-search",
        }

        # Some CrewAI versions let you request only specific tool names in MCPServerAdapter;
        # others don’t. This try/except keeps it compatible.
        try:
            self._mcp_adapter = MCPServerAdapter(stdio_params, *sorted(allowed_names))
            tools = self._mcp_adapter.__enter__()  # keep open
        except TypeError:
            self._mcp_adapter = MCPServerAdapter(stdio_params)
            tools = self._mcp_adapter.__enter__()  # keep open

        # Enforce allowlist no matter what the adapter returned
        filtered = []
        for t in tools:
            name = (getattr(t, "name", "") or "").strip()
            if name in allowed_names:
                filtered.append(t)

        self._mcp_tools = filtered
        return self._mcp_tools

    def close(self):
        try:
            if getattr(self, "_mcp_adapter", None):
                self._mcp_adapter.__exit__(None, None, None)
        except Exception:
            pass

    # ==================== AGENTS ====================
    @agent
    def researcher_agent(self) -> Agent:
        # Tools ONLY here (RAG + web)
        return Agent(
            config=self.agents_config["researcher_agent"],
            llm=self.llm,
            tools=self._get_mcp_tools(),
            verbose=False,
            max_iter=3,  # keep short for local GPU
            allow_delegation=False,
        )

    @agent
    def teacher_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["teacher_agent"],
            llm=self.llm,
            tools=[],
            verbose=False,
            max_iter=2,  # keep short
            allow_delegation=False,
        )

    @agent
    def quiz_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["quiz_agent"],
            llm=self.llm,
            tools=[],  # no tools -> stable JSON
            verbose=False,
            max_iter=2,
            allow_delegation=False,
        )

    # ==================== TASKS ====================
    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"], agent=self.researcher_agent())

    @task
    def teaching_task(self) -> Task:
        return Task(config=self.tasks_config["teaching_task"], agent=self.teacher_agent())

    @task
    def quiz_research_task(self) -> Task:
        return Task(config=self.tasks_config["quiz_research_task"], agent=self.researcher_agent())

    @task
    def quiz_task(self) -> Task:
        return Task(config=self.tasks_config["quiz_task"], agent=self.quiz_agent())

    # ==================== CREWS ====================
    def research_crew(self) -> Crew:
        return Crew(
            agents=[self.researcher_agent()],
            tasks=[self.research_task()],
            process=Process.sequential,
            verbose=True,   # OK to keep logs
            memory=True,    # memory is stored here
            embedder=self.embedder_config,
            cache=True,
            max_rpm=10,
        )

    def teaching_crew(self) -> Crew:
        return Crew(
            agents=[self.teacher_agent()],
            tasks=[self.teaching_task()],
            process=Process.sequential,
            verbose=True,
            memory=False,   # teacher shouldn’t write memory
            cache=False,
            max_rpm=10,
        )

    def quiz_research_crew(self) -> Crew:
        return Crew(
            agents=[self.researcher_agent()],
            tasks=[self.quiz_research_task()],
            process=Process.sequential,
            verbose=True,
            memory=False,   # avoid polluting memory with quiz notes
            cache=False,
            max_rpm=10,
        )

    def quiz_crew(self) -> Crew:
        return Crew(
            agents=[self.quiz_agent()],
            tasks=[self.quiz_task()],
            process=Process.sequential,
            verbose=True,
            memory=False,
            cache=False,
            max_rpm=10,
        )

    # ==================== PUBLIC API ====================
    def ask_question(self, query: str, topic: Optional[str] = None) -> str:
        """
        Researcher -> Teacher pipeline for real questions.
        Greetings are handled directly to avoid tool usage.
        """
        try:
            self._rate_limit_check()

            q = (query or "").strip()
            q_low = q.lower()

            # greeting fast-path
            if q_low in {"hi", "hey", "hello", "yo", "assalamualaikum", "salam"}:
                return "Hello! How can I assist you today?"

            # 1) Research
            research_result = self.research_crew().kickoff(
                inputs={
                    "user_query": q,
                    "topic": topic or q,
                    "conversation_context": "",
                }
            )
            research_notes = str(research_result.raw) if hasattr(research_result, "raw") else str(research_result)

            # 2) Teach
            teach_result = self.teaching_crew().kickoff(
                inputs={"user_query": q, "research_notes": research_notes}
            )
            raw = str(teach_result.raw) if hasattr(teach_result, "raw") else str(teach_result)
            cleaned = self._clean_response(raw)
            return cleaned or raw

        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg or "429" in msg:
                return "⏳ Rate limit reached. Please wait 10 seconds."
            if "timeout" in msg:
                return "⏱️ Request timed out. Try a simpler question or increase OLLAMA_LLM_TIMEOUT."
            if "connection" in msg or "refused" in msg:
                return (
                    "🔌 Connection error.\n\n"
                    "Make sure services are running:\n"
                    "1) Ollama (LLM remote): check OLLAMA_REMOTE_URL\n"
                    "2) Ollama (embeddings local): check OLLAMA_EMBEDDINGS_BASE_URL\n"
                    "3) ChromaDB: docker start chromadb"
                )
            return f"❌ Error: {str(e)[:200]}"

    def generate_quiz(self, topic: str, num_questions: int = 5) -> str:
        """
        Returns quiz as JSON string.
        Pipeline: Researcher (quiz_notes) -> Quiz agent (JSON output).
        """
        try:
            self._rate_limit_check()
            n = max(3, min(int(num_questions), 10))

            # 1) Quiz research notes (RAG-first with tools)
            notes_result = self.quiz_research_crew().kickoff(inputs={"topic": topic})
            quiz_notes = str(notes_result.raw) if hasattr(notes_result, "raw") else str(notes_result)

            # 2) Quiz JSON generation (no tools)
            quiz_result = self.quiz_crew().kickoff(
                inputs={"topic": topic, "num_questions": n, "quiz_notes": quiz_notes}
            )
            raw = str(quiz_result.raw) if hasattr(quiz_result, "raw") else str(quiz_result)

            # Do NOT "clean" JSON aggressively; only strip whitespace
            return (raw or "").strip()

        except Exception as e:
            msg = str(e).lower()
            if "rate" in msg:
                return "⏳ Rate limit reached. Please wait 10 seconds."
            if "timeout" in msg:
                return "⏱️ Quiz request timed out. Increase OLLAMA_LLM_TIMEOUT."
            return f"❌ Quiz error: {str(e)[:200]}"
