"""
Debug script to test ML Learning Assistant components step-by-step
"""
import os
import sys

print("=" * 60)
print("ML LEARNING ASSISTANT - DEBUG TEST")
print("=" * 60)

# Step 1: Check Python version
print(f"\n1. Python Version: {sys.version}")

# Step 2: Check imports
print("\n2. Checking imports...")
try:
    import crewai
    print(f"   ✅ CrewAI: {crewai.__version__}")
except Exception as e:
    print(f"   ❌ CrewAI: {e}")

try:
    import litellm
    print(f"   ✅ LiteLLM: {litellm.__version__}")
except Exception as e:
    print(f"   ❌ LiteLLM: {e}")

try:
    import mcp
    print(f"   ✅ MCP: {mcp.__version__ if hasattr(mcp, '__version__') else 'installed'}")
except Exception as e:
    print(f"   ❌ MCP: {e}")

try:
    import chromadb
    print(f"   ✅ ChromaDB: {chromadb.__version__}")
except Exception as e:
    print(f"   ❌ ChromaDB: {e}")

try:
    from crewai_tools import MCPServerAdapter
    print(f"   ✅ MCPServerAdapter: available")
except Exception as e:
    print(f"   ❌ MCPServerAdapter: {e}")

# Step 3: Check environment variables
print("\n3. Environment Variables:")
print(f"   ACTIVE_LLM_PROVIDER: {os.getenv('ACTIVE_LLM_PROVIDER')}")
print(f"   OLLAMA_BASE_URL: {os.getenv('OLLAMA_BASE_URL')}")
print(f"   OLLAMA_MODEL_NAME: {os.getenv('OLLAMA_MODEL_NAME')}")
print(f"   CHROMA_HOST: {os.getenv('CHROMA_HOST')}")
print(f"   CHROMA_PORT: {os.getenv('CHROMA_PORT')}")

# Step 4: Test Ollama connection
print("\n4. Testing Ollama connection...")
try:
    import httpx
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://localhost:11434')
    response = httpx.get(f"{ollama_url}/api/version", timeout=5)
    if response.status_code == 200:
        print(f"   ✅ Ollama accessible at {ollama_url}")
        print(f"      Version: {response.json()}")
    else:
        print(f"   ❌ Ollama returned status {response.status_code}")
except Exception as e:
    print(f"   ❌ Cannot reach Ollama: {e}")

# Step 5: Test ChromaDB connection
print("\n5. Testing ChromaDB connection...")
try:
    import chromadb
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    client = chromadb.HttpClient(host=host, port=port)
    client.heartbeat()
    print(f"   ✅ ChromaDB accessible at {host}:{port}")
except Exception as e:
    print(f"   ❌ Cannot reach ChromaDB: {e}")

# Step 6: Test LLM initialization
print("\n6. Testing LLM initialization...")
try:
    from src.ml_learning_assistant.llm_config import get_llm
    llm = get_llm()
    print(f"   ✅ LLM initialized: {llm.model}")
except Exception as e:
    print(f"   ❌ LLM initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Step 7: Test MCP Tools (without Streamlit)
print("\n7. Testing MCP Tools initialization...")
try:
    from src.ml_learning_assistant.mcp_servers import docker_mcp_stdio_params
    from mcp import StdioServerParameters
    
    params_dict = docker_mcp_stdio_params()
    stdio_params = StdioServerParameters(
        command=params_dict["command"],
        args=params_dict["args"],
        env=dict(os.environ),
    )
    print(f"   ✅ MCP params created")
    print(f"      Command: {params_dict['command']}")
    print(f"      Args: {params_dict['args']}")
    
    # Try to create adapter (this is where it fails)
    print("\n   Attempting to create MCPServerAdapter...")
    from crewai_tools import MCPServerAdapter
    
    # Check if MCP is available in crewai_tools
    from crewai_tools.adapters.mcp_adapter import MCP_AVAILABLE
    print(f"   MCP_AVAILABLE flag: {MCP_AVAILABLE}")
    
    if not MCP_AVAILABLE:
        print("   ❌ MCP not detected by crewai_tools")
        print("   Trying manual import...")
        try:
            import mcp
            print(f"   ✅ But 'mcp' module IS importable!")
            print(f"      This is a bug in crewai_tools detection")
        except:
            print("   ❌ 'mcp' module really is missing")
    
except Exception as e:
    print(f"   ❌ MCP initialization failed: {e}")
    import traceback
    traceback.print_exc()

# Step 8: Test simple question (without MCP tools)
print("\n8. Testing Crew without MCP tools...")
try:
    from src.ml_learning_assistant.crew import MLLearningAssistantCrew
    
    # Temporarily disable MCP tools
    import src.ml_learning_assistant.crew as crew_module
    original_get_mcp_tools = crew_module.MLLearningAssistantCrew._get_mcp_tools
    
    def mock_get_mcp_tools(self):
        print("   Using mock (empty) MCP tools")
        return []
    
    crew_module.MLLearningAssistantCrew._get_mcp_tools = mock_get_mcp_tools
    
    crew = MLLearningAssistantCrew()
    print("   ✅ Crew initialized (without MCP tools)")
    
    # Test a simple question
    print("\n   Testing simple question...")
    result = crew.ask_question("What is 2+2?")
    print(f"   Result: {result[:100]}...")
    
except Exception as e:
    print(f"   ❌ Crew test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DEBUG TEST COMPLETE")
print("=" * 60)
