"""
LLM Configuration with Rate Limit Protection
"""

import os
from dotenv import load_dotenv

load_dotenv()


def get_llm():
    """
    Get LLM with provider selection and fallback handling.
    """
    active_provider = os.getenv("ACTIVE_LLM_PROVIDER", "").lower().strip()

    try:
        if active_provider == "groq":
            return _get_groq_llm()
        if active_provider == "cerebras":
            return _get_cerebras_llm()
        if active_provider == "ollama":
            return _get_ollama_llm()

        # Auto-detection
        groq_key = os.getenv("GROQ_API_KEY", "").strip()
        if groq_key and groq_key != "gsk_your_groq_api_key_here":
            return _get_groq_llm()

        cerebras_key = os.getenv("CEREBRAS_API_KEY", "").strip()
        if cerebras_key and cerebras_key.startswith("csk-"):
            return _get_cerebras_llm()

        return _get_ollama_llm()
    
    except Exception as e:
        print(f"⚠️ LLM initialization failed: {e}")
        print("🔄 Attempting direct Ollama fallback...")
        return _get_ollama_llm_direct()


def _get_groq_llm():
    from crewai import LLM
    model = os.getenv("GROQ_MODEL_NAME", "llama-3.1-8b-instant")
    print(f"🚀 Using Groq LLM: {model}")
    return LLM(
        model=f"groq/{model}",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.7,
        max_tokens=2048,
        timeout=30,
        max_retries=3,
    )


def _get_cerebras_llm():
    from crewai import LLM
    model = os.getenv("CEREBRAS_MODEL_NAME", "llama3.1-8b")
    print(f"🔄 Using Cerebras LLM: {model}")
    return LLM(
        model=f"cerebras/{model}",
        api_key=os.getenv("CEREBRAS_API_KEY"),
        temperature=0.7,
        max_tokens=4096,
        timeout=30,
        max_retries=3,
    )


def _get_ollama_llm():
    from crewai import LLM
    import os
    
    print("🏠 Using Local Ollama LLM")

    # Start from base URL
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # Optional: switch to remote server (PC)
    mode = os.getenv("OLLAMA_HOST_MODE", "local").lower().strip()
    if mode == "remote":
        ollama_url = os.getenv("OLLAMA_REMOTE_URL", ollama_url)

    # Docker environment detection
    if os.path.exists("/.dockerenv") and "localhost" in ollama_url:
        ollama_url = ollama_url.replace("localhost", "host.docker.internal")

    model = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")
    
    print(f"📍 Ollama URL: {ollama_url}")
    print(f"🤖 Model: {model}")
    
    # Add is_litellm flag to force proper initialization
    return LLM(
        model=f"ollama/{model}",
        base_url=ollama_url,
        temperature=0.7,
        max_tokens=4096,
        timeout=120,
        max_retries=3,
        is_litellm=True,  # ← Add this
    )



def _get_ollama_llm_direct():
    """
    Direct fallback without using CrewAI LLM wrapper.
    Uses litellm directly.
    """
    try:
        from litellm import completion
        print("🔧 Using direct LiteLLM fallback")
        
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        if os.path.exists("/.dockerenv") and "localhost" in ollama_url:
            ollama_url = ollama_url.replace("localhost", "host.docker.internal")
        
        model = os.getenv("OLLAMA_MODEL_NAME", "llama3.2")
        
        # Set litellm environment variables
        os.environ["OLLAMA_API_BASE"] = ollama_url
        
        print(f"📍 Direct LiteLLM - URL: {ollama_url}")
        print(f"🤖 Direct LiteLLM - Model: {model}")
        
        # Return a simple wrapper that CrewAI can use
        from crewai import LLM
        return LLM(
            model=f"ollama/{model}",
            base_url=ollama_url,
            temperature=0.7,
        )
    except Exception as e:
        print(f"❌ Direct fallback also failed: {e}")
        raise


def get_embeddings_config() -> dict:
    """
    Embeddings configuration using Ollama (keep local even if LLM is remote).
    """
    ollama_url = os.getenv("OLLAMA_EMBEDDINGS_BASE_URL",
                           os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))

    # Docker environment detection (only rewrite if we're actually using localhost)
    if os.path.exists("/.dockerenv") and "localhost" in ollama_url:
        ollama_url = ollama_url.replace("localhost", "host.docker.internal")

    return {
        "provider": "ollama",
        "config": {
            "model": os.getenv("OLLAMA_EMBEDDING_MODEL", "mxbai-embed-large"),
            "base_url": ollama_url,
        },
    }
