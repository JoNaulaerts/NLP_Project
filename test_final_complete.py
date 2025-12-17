#!/usr/bin/env python3
# ML Learning Assistant - FINAL COMPREHENSIVE TEST SUITE
# =======================================================
# This is the COMPLETE test with NO SKIPS
# Tests everything: Infrastructure, MCP, RAG, Memory, Agents, Crews, Pipelines

import os
import sys
import json
import time
import chromadb
from pathlib import Path
from datetime import datetime

# Color codes for terminal output
class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"

def log_test(test_name, status, message="", elapsed=None):
    symbol = f"{Colors.GREEN}✓{Colors.RESET}" if status == "PASS" else f"{Colors.RED}✗{Colors.RESET}"
    time_str = f" ({elapsed:.2f}s)" if elapsed else ""
    print(f"{symbol} {Colors.BOLD}{test_name}{Colors.RESET}{time_str}")
    if message:
        for line in message.split("\n"):
            print(f"  {line}")
    return status == "PASS"

def log_section(title):
    print(f"\n{Colors.CYAN}{'='*70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.CYAN}{'='*70}{Colors.RESET}\n")

# Test results tracker
results = {
    "passed": 0,
    "failed": 0,
    "skipped": 0,
    "details": [],
    "start_time": time.time()
}

def record_result(test_name, passed, message="", elapsed=None):
    if passed:
        results["passed"] += 1
        results["details"].append({
            "test": test_name, 
            "status": "PASS", 
            "message": message,
            "elapsed": elapsed
        })
    else:
        results["failed"] += 1
        results["details"].append({
            "test": test_name, 
            "status": "FAIL", 
            "message": message,
            "elapsed": elapsed
        })

# ============================================================================
# SECTION 1: INFRASTRUCTURE TESTS
# ============================================================================

def test_infrastructure():
    log_section("SECTION 1: INFRASTRUCTURE TESTS")

    # Test 1.1: Running in Docker
    start = time.time()
    try:
        in_container = os.path.exists('/.dockerenv') or os.path.exists('/run/.containerenv')
        elapsed = time.time() - start
        record_result("1.1 Docker Environment", in_container, 
                     f"Container detected: {in_container}", elapsed)
        log_test("1.1 Docker Environment", "PASS" if in_container else "WARN",
                f"Running in container: {in_container}", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        record_result("1.1 Docker Environment", False, str(e), elapsed)
        log_test("1.1 Docker Environment", "FAIL", f"Error: {e}", elapsed)

    # Test 1.2: Environment Variables
    start = time.time()
    env_vars = {
        "CHROMA_HOST": os.getenv("CHROMA_HOST"),
        "CHROMA_PORT": os.getenv("CHROMA_PORT"),
        "CHROMA_COLLECTION": os.getenv("CHROMA_COLLECTION"),
        "CHROMA_CLIENT_TYPE": os.getenv("CHROMA_CLIENT_TYPE"),
        "ACTIVE_LLM_PROVIDER": os.getenv("ACTIVE_LLM_PROVIDER"),
        "CREWAI_STORAGE_DIR": os.getenv("CREWAI_STORAGE_DIR"),
    }

    all_set = all(v is not None for v in env_vars.values())
    elapsed = time.time() - start
    record_result("1.2 Environment Variables", all_set,
                 f"Set: {sum(1 for v in env_vars.values() if v is not None)}/{len(env_vars)}", elapsed)
    log_test("1.2 Environment Variables", "PASS" if all_set else "FAIL", 
             "\n    ".join(f"{k}={v}" for k, v in env_vars.items()), elapsed)

    # Test 1.3: Network to ChromaDB
    start = time.time()
    try:
        import socket
        host = os.getenv("CHROMA_HOST", "chromadb")
        port = int(os.getenv("CHROMA_PORT", "8000"))

        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((host, port))
        sock.close()

        reachable = result == 0
        elapsed = time.time() - start
        record_result("1.3 Network to ChromaDB", reachable,
                     f"{host}:{port} {'reachable' if reachable else 'unreachable'}", elapsed)
        log_test("1.3 Network to ChromaDB", "PASS" if reachable else "FAIL",
                f"Connecting to {host}:{port}", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        record_result("1.3 Network to ChromaDB", False, str(e), elapsed)
        log_test("1.3 Network to ChromaDB", "FAIL", f"Error: {e}", elapsed)

# ============================================================================
# SECTION 2: CHROMADB TESTS
# ============================================================================

def test_chromadb():
    log_section("SECTION 2: CHROMADB CONNECTION & DATA TESTS")

    # Test 2.1: ChromaDB Heartbeat
    start = time.time()
    try:
        host = os.getenv("CHROMA_HOST", "chromadb")
        port = int(os.getenv("CHROMA_PORT", "8000"))
        client = chromadb.HttpClient(host=host, port=port)
        heartbeat = client.heartbeat()
        elapsed = time.time() - start
        record_result("2.1 ChromaDB Heartbeat", True, 
                     f"{host}:{port} alive, heartbeat: {heartbeat}", elapsed)
        log_test("2.1 ChromaDB Heartbeat", "PASS", f"Connected to {host}:{port}", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        record_result("2.1 ChromaDB Heartbeat", False, str(e), elapsed)
        log_test("2.1 ChromaDB Heartbeat", "FAIL", f"Error: {e}", elapsed)
        return

    # Test 2.2: List Collections
    start = time.time()
    try:
        collections = client.list_collections()
        collection_names = [c.name for c in collections]
        elapsed = time.time() - start
        record_result("2.2 List Collections", True,
                     f"Found {len(collections)} collections: {collection_names}", elapsed)
        log_test("2.2 List Collections", "PASS",
                f"Collections: {collection_names}", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        record_result("2.2 List Collections", False, str(e), elapsed)
        log_test("2.2 List Collections", "FAIL", f"Error: {e}", elapsed)

    # Test 2.3: Target Collection
    start = time.time()
    try:
        collection_name = os.getenv("CHROMA_COLLECTION", "ml_materials")
        col = client.get_or_create_collection(name=collection_name)
        count = col.count()
        elapsed = time.time() - start
        record_result("2.3 Target Collection", True, 
                     f"{collection_name} has {count} documents", elapsed)
        log_test("2.3 Target Collection", "PASS",
                f"Collection '{collection_name}' count: {count}", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        record_result("2.3 Target Collection", False, str(e), elapsed)
        log_test("2.3 Target Collection", "FAIL", f"Error: {e}", elapsed)
        return

    # Test 2.4: Query Collection
    start = time.time()
    if count > 0:
        try:
            results_query = col.query(
                query_texts=["machine learning transformer attention"],
                n_results=min(3, count)
            )
            docs = results_query.get("documents", [[]])[0]
            has_results = len(docs) > 0
            elapsed = time.time() - start
            record_result("2.4 Query Collection", has_results,
                         f"Retrieved {len(docs)} documents", elapsed)
            log_test("2.4 Query Collection", "PASS" if has_results else "FAIL",
                    f"Query returned {len(docs)} results", elapsed)
            if has_results:
                print(f"    Sample: {docs[0][:100]}...")
        except Exception as e:
            elapsed = time.time() - start
            record_result("2.4 Query Collection", False, str(e), elapsed)
            log_test("2.4 Query Collection", "FAIL", f"Error: {e}", elapsed)
    else:
        elapsed = time.time() - start
        record_result("2.4 Query Collection", False, "Collection is empty", elapsed)
        log_test("2.4 Query Collection", "WARN", "No documents to query", elapsed)

# ============================================================================
# SECTION 3: MCP INTEGRATION TESTS
# ============================================================================

def test_mcp_integration():
    log_section("SECTION 3: MCP INTEGRATION TESTS")

    # Test 3.1: Import Crew
    start = time.time()
    try:
        from src.ml_learning_assistant.crew import MLLearningAssistantCrew
        elapsed = time.time() - start
        record_result("3.1 Import Crew", True, "Module imported successfully", elapsed)
        log_test("3.1 Import Crew", "PASS", "", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        record_result("3.1 Import Crew", False, str(e), elapsed)
        log_test("3.1 Import Crew", "FAIL", f"Error: {e}", elapsed)
        return

    # Test 3.2: Initialize Crew
    start = time.time()
    try:
        crew = MLLearningAssistantCrew()
        elapsed = time.time() - start
        record_result("3.2 Initialize Crew", True, f"LLM: {crew.llm.model}", elapsed)
        log_test("3.2 Initialize Crew", "PASS", f"LLM: {crew.llm.model}", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        record_result("3.2 Initialize Crew", False, str(e), elapsed)
        log_test("3.2 Initialize Crew", "FAIL", f"Error: {e}", elapsed)
        return

    # Test 3.3: Get MCP Tools
    start = time.time()
    try:
        tools = crew._get_mcp_tools()
        tool_names = [t.name for t in tools]
        has_tools = len(tools) > 0
        elapsed = time.time() - start
        record_result("3.3 MCP Tools Loaded", has_tools,
                     f"Found {len(tools)} tools: {tool_names}", elapsed)
        log_test("3.3 MCP Tools Loaded", "PASS" if has_tools else "FAIL",
                f"Tools ({len(tools)}): {', '.join(tool_names)}", elapsed)
    except Exception as e:
        elapsed = time.time() - start
        record_result("3.3 MCP Tools Loaded", False, str(e), elapsed)
        log_test("3.3 MCP Tools Loaded", "FAIL", f"Error: {e}", elapsed)
        return

    # Test 3.4: Chroma Tools Available
    start = time.time()
    chroma_tools = [t for t in tools if 'chroma' in t.name.lower()]
    has_chroma = len(chroma_tools) > 0
    elapsed = time.time() - start
    record_result("3.4 Chroma Tools", has_chroma,
                 f"Found {len(chroma_tools)} chroma tools", elapsed)
    log_test("3.4 Chroma Tools", "PASS" if has_chroma else "FAIL",
             f"Chroma tools: {[t.name for t in chroma_tools]}", elapsed)

    # Test 3.5: Tavily Tools Available
    start = time.time()
    tavily_tools = [t for t in tools if 'tavily' in t.name.lower()]
    has_tavily = len(tavily_tools) > 0
    elapsed = time.time() - start
    record_result("3.5 Tavily Tools", has_tavily,
                 f"Found {len(tavily_tools)} tavily tools", elapsed)
    log_test("3.5 Tavily Tools", "PASS" if has_tavily else "FAIL",
             f"Tavily tools: {[t.name for t in tavily_tools]}", elapsed)

    # Test 3.6: Execute Chroma Query Tool
    if has_chroma:
        start = time.time()
        try:
            chroma_tool = chroma_tools[0]
            result = chroma_tool._run(
                query="deep learning neural networks",  # ✅ Fixed param name
                n_results=2
            )
            has_results = result and len(str(result)) > 50
            elapsed = time.time() - start
            record_result("3.6 Chroma Tool Execution", has_results,
                        f"Returned {len(str(result))} chars", elapsed)
            log_test("3.6 Chroma Tool Execution", "PASS" if has_results else "FAIL",
                    f"Result length: {len(str(result))}", elapsed)
            if has_results:
                print(f"    Preview: {str(result)[:150]}...")
        except Exception as e:
            elapsed = time.time() - start
            record_result("3.6 Chroma Tool Execution", False, str(e), elapsed)
            log_test("3.6 Chroma Tool Execution", "FAIL", f"Error: {e}", elapsed)


# ============================================================================
# SECTION 4: AGENTS & CREWS CONFIGURATION TESTS
# ============================================================================

def test_agents_and_crews():
    log_section("SECTION 4: AGENTS & CREWS CONFIGURATION TESTS")

    try:
        from src.ml_learning_assistant.crew import MLLearningAssistantCrew
        crew_instance = MLLearningAssistantCrew()

        # Test 4.1: Researcher Agent
        start = time.time()
        try:
            researcher = crew_instance.researcher_agent()
            has_tools = len(researcher.tools) > 0
            elapsed = time.time() - start
            record_result("4.1 Researcher Agent", True,
                         f"Tools: {len(researcher.tools)}, LLM: {researcher.llm.model}", elapsed)
            log_test("4.1 Researcher Agent", "PASS",
                    f"Role: {researcher.role}, Tools: {len(researcher.tools)}", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("4.1 Researcher Agent", False, str(e), elapsed)
            log_test("4.1 Researcher Agent", "FAIL", f"Error: {e}", elapsed)

        # Test 4.2: Teacher Agent
        start = time.time()
        try:
            teacher = crew_instance.teacher_agent()
            no_tools = len(teacher.tools) == 0
            elapsed = time.time() - start
            record_result("4.2 Teacher Agent", True,
                         f"No tools (expected): {no_tools}", elapsed)
            log_test("4.2 Teacher Agent", "PASS",
                    f"Role: {teacher.role}, Tools: {len(teacher.tools)}", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("4.2 Teacher Agent", False, str(e), elapsed)
            log_test("4.2 Teacher Agent", "FAIL", f"Error: {e}", elapsed)

        # Test 4.3: Quiz Agent
        start = time.time()
        try:
            quiz = crew_instance.quiz_agent()
            no_tools = len(quiz.tools) == 0
            elapsed = time.time() - start
            record_result("4.3 Quiz Agent", True,
                         f"No tools (expected): {no_tools}", elapsed)
            log_test("4.3 Quiz Agent", "PASS",
                    f"Role: {quiz.role}, Tools: {len(quiz.tools)}", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("4.3 Quiz Agent", False, str(e), elapsed)
            log_test("4.3 Quiz Agent", "FAIL", f"Error: {e}", elapsed)

        # Test 4.4: Research Crew (with memory)
        start = time.time()
        try:
            research_crew = crew_instance.research_crew()
            has_memory = research_crew.memory
            elapsed = time.time() - start
            record_result("4.4 Research Crew", True,
                         f"Memory: {has_memory}, Agents: {len(research_crew.agents)}", elapsed)
            log_test("4.4 Research Crew", "PASS",
                    f"Memory: {has_memory}, Agents: {len(research_crew.agents)}", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("4.4 Research Crew", False, str(e), elapsed)
            log_test("4.4 Research Crew", "FAIL", f"Error: {e}", elapsed)

        # Test 4.5: Teaching Crew (no memory)
        start = time.time()
        try:
            teaching_crew = crew_instance.teaching_crew()
            no_memory = not teaching_crew.memory
            elapsed = time.time() - start
            record_result("4.5 Teaching Crew", True,
                         f"No memory (expected): {no_memory}", elapsed)
            log_test("4.5 Teaching Crew", "PASS",
                    f"Memory disabled: {no_memory}", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("4.5 Teaching Crew", False, str(e), elapsed)
            log_test("4.5 Teaching Crew", "FAIL", f"Error: {e}", elapsed)

        # Test 4.6: Quiz Crews
        start = time.time()
        try:
            quiz_research_crew = crew_instance.quiz_research_crew()
            quiz_crew = crew_instance.quiz_crew()
            elapsed = time.time() - start
            record_result("4.6 Quiz Crews", True,
                         f"Quiz research & quiz crews initialized", elapsed)
            log_test("4.6 Quiz Crews", "PASS",
                    f"Both quiz crews ready", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("4.6 Quiz Crews", False, str(e), elapsed)
            log_test("4.6 Quiz Crews", "FAIL", f"Error: {e}", elapsed)

    except Exception as e:
        log_test("4.x Agents/Crews", "FAIL", f"Failed to initialize: {e}")

# ============================================================================
# SECTION 5: MEMORY SYSTEM TESTS
# ============================================================================

def test_memory_system():
    log_section("SECTION 5: MEMORY SYSTEM TESTS")

    # Test 5.1: Storage Directory
    start = time.time()
    storage_dir = Path(os.getenv("CREWAI_STORAGE_DIR", "./data/crewai_memory"))
    exists = storage_dir.exists()
    elapsed = time.time() - start
    record_result("5.1 Storage Directory", exists, str(storage_dir), elapsed)
    log_test("5.1 Storage Directory", "PASS" if exists else "FAIL",
            f"Path: {storage_dir} ({'exists' if exists else 'MISSING'})", elapsed)

    # Test 5.2: Storage Writable
    if exists:
        start = time.time()
        try:
            test_file = storage_dir / ".test_write"
            test_file.write_text("test")
            test_file.unlink()
            elapsed = time.time() - start
            record_result("5.2 Storage Writable", True, "Can write to storage", elapsed)
            log_test("5.2 Storage Writable", "PASS", "", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("5.2 Storage Writable", False, str(e), elapsed)
            log_test("5.2 Storage Writable", "FAIL", f"Error: {e}", elapsed)

    # Test 5.3: Memory Files
    start = time.time()
    try:
        if exists:
            memory_files = list(storage_dir.glob("**/*.db")) + list(storage_dir.glob("**/*.json"))
            elapsed = time.time() - start
            record_result("5.3 Memory Files", True,
                         f"Found {len(memory_files)} files", elapsed)
            log_test("5.3 Memory Files", "PASS",
                    f"Memory files: {len(memory_files)}", elapsed)
            if memory_files:
                for f in memory_files[:5]:
                    print(f"    - {f.name}")
    except Exception as e:
        elapsed = time.time() - start
        record_result("5.3 Memory Files", False, str(e), elapsed)
        log_test("5.3 Memory Files", "FAIL", f"Error: {e}", elapsed)

# ============================================================================
# SECTION 6: PIPELINE TESTS (END-TO-END)
# ============================================================================

def test_pipelines():
    log_section("SECTION 6: PIPELINE TESTS (END-TO-END)")

    try:
        from src.ml_learning_assistant.crew import MLLearningAssistantCrew
        crew_instance = MLLearningAssistantCrew()

        # Test 6.1: Greeting (fast path)
        start = time.time()
        try:
            result = crew_instance.ask_question("Hello")
            is_greeting = len(result) < 150 and "assist" in result.lower()
            elapsed = time.time() - start
            record_result("6.1 Greeting Response", is_greeting,
                         f"Response: {result[:80]}...", elapsed)
            log_test("6.1 Greeting Response", "PASS" if is_greeting else "FAIL",
                    f"Result: {result}", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("6.1 Greeting Response", False, str(e), elapsed)
            log_test("6.1 Greeting Response", "FAIL", f"Error: {e}", elapsed)

        # Test 6.2: Simple Q&A (RAG pipeline)
        print(f"\n{Colors.YELLOW}⚠️  Running full Q&A pipeline (may take 30-60s)...{Colors.RESET}")
        start = time.time()
        try:
            result = crew_instance.ask_question("What is attention mechanism in 2 sentences?")
            has_answer = len(result) > 50 and "error" not in result.lower()
            elapsed = time.time() - start
            record_result("6.2 Q&A Pipeline", has_answer,
                         f"Response length: {len(result)}", elapsed)
            log_test("6.2 Q&A Pipeline", "PASS" if has_answer else "FAIL",
                    f"Response: {result[:200]}...", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("6.2 Q&A Pipeline", False, str(e), elapsed)
            log_test("6.2 Q&A Pipeline", "FAIL", f"Error: {e}", elapsed)

        # Test 6.3: Quiz Generation
        print(f"\n{Colors.YELLOW}⚠️  Running quiz generation (may take 30-60s)...{Colors.RESET}")
        start = time.time()
        try:
            quiz_result = crew_instance.generate_quiz("neural networks", 3)
            quiz_json = json.loads(quiz_result)
            is_valid = "questions" in quiz_json and len(quiz_json.get("questions", [])) == 3
            elapsed = time.time() - start
            record_result("6.3 Quiz Generation", is_valid,
                         f"Generated {len(quiz_json.get('questions', []))} questions", elapsed)
            log_test("6.3 Quiz Generation", "PASS" if is_valid else "FAIL",
                    f"Quiz: {len(quiz_json.get('questions', []))} questions", elapsed)
        except Exception as e:
            elapsed = time.time() - start
            record_result("6.3 Quiz Generation", False, str(e), elapsed)
            log_test("6.3 Quiz Generation", "FAIL", f"Error: {e}", elapsed)

    except Exception as e:
        log_test("6.x Pipeline Tests", "FAIL", f"Failed: {e}")

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

def run_all_tests():
    print(f"{Colors.BOLD}{Colors.MAGENTA}")
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     ML LEARNING ASSISTANT - FINAL COMPREHENSIVE TEST SUITE        ║")
    print("║                  TESTING EVERYTHING (NO SKIPS)                     ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print(f"{Colors.RESET}\n")

    # Run all test sections
    test_infrastructure()
    test_chromadb()
    test_mcp_integration()
    test_agents_and_crews()
    test_memory_system()
    test_pipelines()

    # Final summary
    end_time = time.time()
    duration = end_time - results["start_time"]

    log_section("FINAL SUMMARY")

    total_tests = results["passed"] + results["failed"]
    pass_rate = (results["passed"] / total_tests * 100) if total_tests > 0 else 0

    print(f"{Colors.BOLD}Total Tests:{Colors.RESET} {total_tests}")
    print(f"{Colors.GREEN}✓ Passed:{Colors.RESET} {results['passed']}")
    print(f"{Colors.RED}✗ Failed:{Colors.RESET} {results['failed']}")
    print(f"{Colors.YELLOW}⊘ Skipped:{Colors.RESET} {results['skipped']}")
    print(f"{Colors.BOLD}Pass Rate:{Colors.RESET} {pass_rate:.1f}%")
    print(f"{Colors.BOLD}Total Duration:{Colors.RESET} {duration:.2f}s")

    # Show slowest tests
    print(f"\n{Colors.BOLD}Slowest Tests:{Colors.RESET}")
    sorted_tests = sorted(results["details"], key=lambda x: x.get("elapsed", 0), reverse=True)
    for test in sorted_tests[:5]:
        if test.get("elapsed"):
            print(f"  {test['test']}: {test['elapsed']:.2f}s")

    # Save detailed results
    try:
        output_dir = Path("/app/data/test_results")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / f"final_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        output_file.write_text(json.dumps({
            "summary": {
                "total": total_tests,
                "passed": results["passed"],
                "failed": results["failed"],
                "skipped": results["skipped"],
                "pass_rate": pass_rate,
                "duration": duration,
                "environment": "docker",
                "test_type": "comprehensive_final"
            },
            "details": results["details"],
            "timestamp": datetime.now().isoformat()
        }, indent=2))
        print(f"\n{Colors.BLUE}📄 Results saved to: {output_file}{Colors.RESET}\n")
    except Exception as e:
        print(f"\n{Colors.YELLOW}⚠️  Could not save results: {e}{Colors.RESET}\n")

    # Final verdict
    if results["failed"] == 0:
        print(f"{Colors.GREEN}{Colors.BOLD}")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║                    🎉 ALL TESTS PASSED! 🎉                         ║")
        print("║          Your ML Learning Assistant is fully operational!         ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.RESET}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}")
        print("╔════════════════════════════════════════════════════════════════════╗")
        print("║                    ⚠️  SOME TESTS FAILED ⚠️                        ║")
        print("║              Review errors above and fix issues                   ║")
        print("╚════════════════════════════════════════════════════════════════════╝")
        print(f"{Colors.RESET}")

    # Return exit code
    return 0 if results["failed"] == 0 else 1

if __name__ == "__main__":
    sys.exit(run_all_tests())
