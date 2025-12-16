# ML Learning Assistant

A multi-agent educational platform powered by CrewAI for university students to learn Machine Learning concepts.

## 🌟 Features

- 💬 **Interactive Chat** with 4 specialized AI agents
- 📚 **RAG-Based Document Search** over ML textbooks
- 🌐 **Web Search** via Tavily API (AI-optimized)
- 📝 **Adaptive Quiz Generation** (MCQ + True/False)
- 📄 **Document Support** (PDF, TXT, MD)
- 💾 **Session Management** - Multiple conversation sessions
- 🎯 **Local LLM** using Ollama (Llama 3.2)

## 🏗️ Architecture

### Multi-Agent System
1. **Planning Agent** - Routes queries and coordinates responses
2. **Research Agent** - RAG search + Tavily web search with citations
3. **Educational Agent** - Creates explanations and summaries
4. **Assessment Agent** - Generates quizzes with detailed explanations

### Tech Stack
- **Framework**: CrewAI with @CrewBase decorator pattern
- **LLM**: Ollama (Llama 3.2) - runs locally
- **Embeddings**: Ollama (nomic-embed-text)
- **Vector DB**: ChromaDB
- **Web Search**: Tavily API (AI-optimized search)
- **Web Interface**: Streamlit
- **Document Processing**: LlamaIndex, PyPDF

## 📁 Project Structure

```
NLP_Project/
├── src/
│   └── ml_learning_assistant/
│       ├── __init__.py
│       ├── main.py              # CLI entry point
│       ├── crew.py              # CrewAI agents & tasks
│       ├── config/
│       │   ├── agents.yaml      # Agent configurations
│       │   └── tasks.yaml       # Task definitions
│       └── tools/
│           ├── __init__.py
│           ├── rag_tool.py      # Document search
│           └── web_search_tool.py  # Tavily web search
├── data/
│   ├── textbooks/              # Pre-loaded ML textbooks
│   └── user_uploads/           # User-uploaded documents
├── app.py                       # Streamlit interface
├── requirements.txt
├── .env
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 🚀 Setup Instructions

### Prerequisites
- Python 3.11+
- [Ollama](https://ollama.com) installed
- Tavily API key (get free at [tavily.com](https://tavily.com))

### 1. Create Virtual Environment

```powershell
# Create venv
python -m venv venv

# Activate
.\venv\Scripts\Activate.ps1  # Windows PowerShell
# or: source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Setup Ollama

```powershell
# Pull required models
ollama pull llama3.2
ollama pull nomic-embed-text
```

### 4. Configure Environment

Your `.env` file already has Tavily configured:

```env
# Ollama Configuration
OLLAMA_BASE_URL=http://localhost:11434

# Tavily API Key (Required for web search)
TAVILY_API_KEY=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
```

### 5. Add ML Textbooks (Optional)

Place PDF/TXT/MD files in `data/textbooks/` directory.

### 6. Run the Application

```powershell
streamlit run app.py
```

The app will open at `http://localhost:8501`

## 🐳 Docker Deployment

### Prerequisites
- Ollama installed and running on **host machine** (not in Docker)
- Models pulled: `ollama pull llama3.2` and `ollama pull nomic-embed-text`

### Build and Run with Docker Compose

```powershell
# 1. Start Ollama on host (separate terminal)
ollama serve

# 2. Build and start Docker services
docker-compose up --build

# Access the app at http://localhost:8501
```

### Services
- **app**: Streamlit application (port 8501)
  - Connects to Ollama on host via `host.docker.internal:11434`
- **chromadb**: Vector database (port 8000)

### Note
Ollama runs on your **host machine**, not in Docker. The app container connects to it via `host.docker.internal` (Windows/Mac) or host IP (Linux).

## 💡 Usage

### Chat Mode
1. Select "💬 Chat" mode
2. Ask questions about ML concepts
3. Get responses with sources from:
   - Pre-loaded textbooks (RAG)
   - User-uploaded documents
   - Web search (Tavily)

### Quiz Mode
1. Select "📝 Quiz" mode
2. Enter a topic (e.g., "neural networks")
3. Choose number of questions
4. Get MCQ/True-False questions with explanations

### Upload Documents
1. Select "📚 Upload Documents" mode
2. Upload PDF/TXT/MD files
3. Documents are indexed for future queries

### Session Management
- **➕ New Session**: Create separate conversation threads
- **Switch Sessions**: Click session name to switch
- **🗑️ Delete**: Remove old conversations
- **📥 Export**: Download conversation history as JSON

## 🔧 Configuration

### Agent Configuration
Edit `src/ml_learning_assistant/config/agents.yaml`:
- Customize agent roles, goals, and backstories
- Enable/disable verbose logging
- Configure delegation settings

### Task Configuration
Edit `src/ml_learning_assistant/config/tasks.yaml`:
- Define task descriptions and expected outputs
- Set task dependencies
- Configure context passing

### LLM Settings
Edit `.env`:
```env
OLLAMA_BASE_URL=http://localhost:11434
TAVILY_API_KEY=your-key-here
```

## 📚 Key Components

### Web Search Tool (Tavily Only)
- Located: `src/ml_learning_assistant/tools/web_search_tool.py`
- **No DuckDuckGo fallback** - Tavily API only
- Returns AI-generated summaries + search results
- Proper error handling with API key validation

### RAG Tool
- Located: `src/ml_learning_assistant/tools/rag_tool.py`
- Searches through local documents
- Uses ChromaDB for vector storage
- Ollama embeddings (nomic-embed-text)

### Crew Configuration
- Located: `src/ml_learning_assistant/crew.py`
- Uses @CrewBase decorator pattern
- Sequential process with memory
- 4 specialized agents

## 🛠️ Development

### Run CLI Version

```powershell
python -m src.ml_learning_assistant.main
```

### Project Dependencies
- `crewai` - Multi-agent framework
- `crewai-tools` - Tool integrations
- `streamlit` - Web interface
- `llama-index` - Document processing
- `chromadb` - Vector database
- `requests` - HTTP client for Tavily
- `tavily-python` - Tavily SDK
- `pypdf` - PDF parsing

## 🐛 Troubleshooting

### Ollama Connection Issues
```powershell
# Check if Ollama is running
ollama list

# Restart Ollama
ollama serve
```

### Tavily API Issues
```powershell
# Verify API key in .env
Get-Content .env | Select-String "TAVILY"

# Test API key
curl -X POST https://api.tavily.com/search -H "Content-Type: application/json" -d '{"api_key":"your-key","query":"test"}'
```

### ChromaDB Issues
```powershell
# Clear ChromaDB cache
Remove-Item -Recurse -Force data/chroma_db
```

### Port Already in Use
```powershell
# Change Streamlit port
streamlit run app.py --server.port=8502
```

## 📧 Support

For issues and questions:
- Open an issue on GitHub
- Check [CrewAI Documentation](https://docs.crewai.com)
- Review [Tavily API Docs](https://docs.tavily.com)

---

**Built with ❤️ for ML learners | Powered by CrewAI & Tavily**
