"""
ML Learning Assistant - Professional Streamlit UI
Enhanced with provider selection, refined design, and IMPROVED TEXT CONTRAST
"""

from __future__ import annotations

import os
os.environ["CREWAI_DISABLE_TELEMETRY"] = "true"
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TRACING_ENABLED"] = "false"

import json
import time
from pathlib import Path
from datetime import datetime
from typing import Any

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Force defaults
os.environ.setdefault("CREWAI_TELEMETRY", "false")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CHROMA_COLLECTION", "ml_materials")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

CREWAI_STORE = (DATA_DIR / "crewai_memory").resolve()
os.environ["CREWAI_STORAGE_DIR"] = str(CREWAI_STORE)

from src.ml_learning_assistant.crew import MLLearningAssistantCrew
from src.ml_learning_assistant.tools.upload_to_chromadb import upload_document_to_chromadb

APP_STATE_DIR = DATA_DIR / "ui_state"
APP_STATE_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploaded_docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

UPLOADED_TRACK_FILE = APP_STATE_DIR / "uploaded_docs.json"

# Page config with custom theme
st.set_page_config(
    page_title="ML Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional styling with IMPROVED TEXT CONTRAST
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --primary: #6366f1;
  --primary-dark: #4f46e5;
  --primary-light: #a5b4fc;
  --secondary: #06b6d4;
  --secondary-dark: #0891b2;
  --accent: #8b5cf6;
  --success: #10b981;
  --warning: #f59e0b;
  --error: #ef4444;
  --bg-dark: #0f172a;
  --bg-darker: #020617;
  --surface: rgba(30, 41, 59, 0.5);
  --surface-light: rgba(51, 65, 85, 0.6);
  --border: rgba(148, 163, 184, 0.15);
  /* IMPROVED TEXT COLORS FOR BETTER CONTRAST */
  --text-primary: #ffffff;
  --text-secondary: #e2e8f0;
  --text-muted: #cbd5e1;
}

* {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

.stApp {
  background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%);
  background-attachment: fixed;
  color: var(--text-primary);
}

.stApp::before {
  content: '';
  position: fixed;
  top: -50%;
  left: -50%;
  width: 200%;
  height: 200%;
  background: 
    radial-gradient(circle at 20% 50%, rgba(99, 102, 241, 0.08) 0%, transparent 50%),
    radial-gradient(circle at 80% 80%, rgba(139, 92, 246, 0.08) 0%, transparent 50%),
    radial-gradient(circle at 40% 10%, rgba(6, 182, 212, 0.06) 0%, transparent 50%);
  animation: drift 30s ease-in-out infinite;
  z-index: 0;
  pointer-events: none;
}

@keyframes drift {
  0%, 100% { transform: translate(0, 0) rotate(0deg); }
  33% { transform: translate(30px, -50px) rotate(2deg); }
  66% { transform: translate(-20px, 20px) rotate(-2deg); }
}

.main > div {
  position: relative;
  z-index: 1;
}

/* IMPROVED HEADING CONTRAST */
h1, h2, h3, h4, h5, h6 {
  color: var(--text-primary) !important;
  font-weight: 700 !important;
  letter-spacing: -0.02em;
}

h1 { font-size: 2.5rem !important; }
h2 { font-size: 1.875rem !important; }
h3 { font-size: 1.5rem !important; }

/* IMPROVED TEXT CONTRAST */
p, li, label, span, div {
  color: var(--text-secondary) !important;
}

.stMarkdown, .stMarkdown p, .stMarkdown li {
  color: var(--text-secondary) !important;
}

/* Make sure captions are still visible but slightly muted */
.stCaption, [data-testid="stCaptionContainer"] {
  color: var(--text-muted) !important;
}

.glass-card {
  background: var(--surface);
  backdrop-filter: blur(16px) saturate(180%);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.5rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2);
  transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.glass-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 12px 48px rgba(99, 102, 241, 0.15);
  border-color: rgba(99, 102, 241, 0.3);
}

.glass-card-strong {
  background: var(--surface-light);
  backdrop-filter: blur(20px) saturate(200%);
  border: 1px solid rgba(99, 102, 241, 0.2);
  border-radius: 16px;
  padding: 1.5rem;
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

.badge {
  display: inline-flex;
  align-items: center;
  padding: 0.375rem 0.875rem;
  border-radius: 9999px;
  font-size: 0.75rem;
  font-weight: 600;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(139, 92, 246, 0.25));
  border: 1px solid rgba(99, 102, 241, 0.5);
  color: #d4d4ff !important;
  margin-right: 0.5rem;
  margin-bottom: 0.5rem;
  box-shadow: 0 2px 8px rgba(99, 102, 241, 0.2);
  transition: all 0.2s;
}

.badge:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
}

.badge-secondary {
  background: linear-gradient(135deg, rgba(6, 182, 212, 0.25), rgba(14, 165, 233, 0.25));
  border-color: rgba(6, 182, 212, 0.5);
  color: #ccf5ff !important;
}

.badge-success {
  background: linear-gradient(135deg, rgba(16, 185, 129, 0.25), rgba(5, 150, 105, 0.25));
  border-color: rgba(16, 185, 129, 0.5);
  color: #d1fae5 !important;
}

section[data-testid="stSidebar"] {
  background: rgba(15, 23, 42, 0.9) !important;
  backdrop-filter: blur(20px) saturate(180%);
  border-right: 1px solid var(--border);
}

section[data-testid="stSidebar"] > div {
  background: transparent;
}

/* Sidebar text contrast */
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] span {
  color: var(--text-secondary) !important;
}

.stChatMessage {
  background: var(--surface) !important;
  backdrop-filter: blur(16px);
  border: 1px solid var(--border) !important;
  border-radius: 16px !important;
  padding: 1.25rem !important;
  margin: 0.75rem 0 !important;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.2);
  transition: all 0.3s ease;
}

.stChatMessage:hover {
  border-color: rgba(99, 102, 241, 0.3) !important;
  transform: translateX(4px);
}

/* Chat message text contrast */
.stChatMessage p, .stChatMessage span {
  color: var(--text-primary) !important;
}

.stButton > button, .stDownloadButton > button {
  border-radius: 12px !important;
  font-weight: 600 !important;
  border: 1px solid var(--border) !important;
  background: var(--surface) !important;
  color: var(--text-primary) !important;
  padding: 0.625rem 1.25rem !important;
  transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.stButton > button:hover, .stDownloadButton > button:hover {
  border-color: var(--primary) !important;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.25), rgba(139, 92, 246, 0.2)) !important;
  transform: translateY(-1px);
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.3) !important;
}

.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(135deg, var(--primary), var(--accent)) !important;
  border: none !important;
  color: white !important;
  box-shadow: 0 4px 16px rgba(99, 102, 241, 0.4);
}

.stButton > button[kind="primary"]:hover {
  background: linear-gradient(135deg, var(--primary-dark), var(--accent)) !important;
  transform: translateY(-2px);
  box-shadow: 0 8px 24px rgba(99, 102, 241, 0.5) !important;
}

/* Improved metric contrast */
div[data-testid="stMetricValue"] {
  color: var(--primary-light) !important;
  font-size: 1.875rem !important;
  font-weight: 700 !important;
}

div[data-testid="stMetricLabel"] {
  color: var(--text-secondary) !important;
  font-weight: 600 !important;
  text-transform: uppercase;
  font-size: 0.75rem;
  letter-spacing: 0.05em;
}

/* Form input contrast */
.stTextInput > div > div > input, 
.stTextArea > div > div > textarea,
.stSelectbox select {
  background: rgba(30, 41, 59, 0.7) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
  color: var(--text-primary) !important;
  padding: 0.75rem 1rem !important;
  transition: all 0.2s ease;
}

.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder {
  color: var(--text-muted) !important;
  opacity: 0.7;
}

.stTextInput > div > div > input:focus, 
.stTextArea > div > div > textarea:focus,
.stSelectbox select:focus {
  border-color: var(--primary) !important;
  box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.15) !important;
  background: rgba(30, 41, 59, 0.9) !important;
}

/* Select dropdown text */
.stSelectbox > div > div > div {
  background: rgba(30, 41, 59, 0.9) !important;
  color: var(--text-primary) !important;
}

.stSelectbox option {
  background: rgba(30, 41, 59, 0.95) !important;
  color: var(--text-primary) !important;
}

.stSlider > div > div {
  background: var(--surface) !important;
  border-radius: 12px !important;
}

/* Slider text */
.stSlider label {
  color: var(--text-secondary) !important;
}

.stExpander {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

/* Expander text contrast */
.stExpander p, .stExpander label, .stExpander span {
  color: var(--text-secondary) !important;
}

.uploadedFile {
  background: var(--surface) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

hr {
  border: none;
  border-top: 1px solid var(--border);
  margin: 1.5rem 0;
  opacity: 0.5;
}

/* Alert text contrast */
.stAlert {
  border-radius: 12px !important;
  border: 1px solid var(--border) !important;
}

.stAlert p, .stAlert span {
  color: var(--text-primary) !important;
}

.header-section {
  margin-bottom: 2rem;
  padding-bottom: 1rem;
  border-bottom: 2px solid rgba(99, 102, 241, 0.3);
}

.status-indicator {
  display: inline-block;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--success);
  box-shadow: 0 0 8px var(--success);
  animation: pulse 2s ease-in-out infinite;
  margin-right: 0.5rem;
}

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}

.provider-selector {
  margin: 1rem 0;
  padding: 1rem;
  background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.08));
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 12px;
}

/* Code block contrast */
code {
  background: rgba(0, 0, 0, 0.5) !important;
  border: 1px solid var(--border) !important;
  border-radius: 6px !important;
  padding: 0.2rem 0.4rem !important;
  color: #e0e7ff !important;
  font-weight: 500;
}

.stCodeBlock {
  background: rgba(0, 0, 0, 0.5) !important;
  border: 1px solid var(--border) !important;
  border-radius: 12px !important;
}

.stCodeBlock code {
  color: #e0e7ff !important;
}

/* Progress bar */
.stProgress > div > div > div {
  background: linear-gradient(90deg, var(--primary), var(--accent)) !important;
}

/* Radio button text */
.stRadio label {
  color: var(--text-secondary) !important;
}

.stRadio p {
  color: var(--text-secondary) !important;
}

/* Info/Warning/Error/Success boxes */
.stInfo, .stWarning, .stError, .stSuccess {
  color: var(--text-primary) !important;
}

.stInfo p, .stWarning p, .stError p, .stSuccess p {
  color: var(--text-primary) !important;
}

/* Spinner text */
.stSpinner > div {
  color: var(--text-secondary) !important;
}
</style>
""", unsafe_allow_html=True)

# Utility functions remain the same
def _load_uploaded_docs() -> list[str]:
    if UPLOADED_TRACK_FILE.exists():
        try:
            return json.loads(UPLOADED_TRACK_FILE.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []

def _save_uploaded_docs(docs: list[str]) -> None:
    try:
        UPLOADED_TRACK_FILE.write_text(json.dumps(sorted(list(set(docs))), indent=2), encoding="utf-8")
    except Exception:
        pass

@st.cache_resource
def get_crew() -> MLLearningAssistantCrew:
    return MLLearningAssistantCrew()

def reset_crew():
    try:
        c = get_crew()
        if hasattr(c, "close"):
            c.close()
    except Exception:
        pass
    st.cache_resource.clear()

def get_file_icon(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    icons = {
        ".pdf": "📕",
        ".txt": "📄",
        ".md": "📝",
        ".docx": "📘",
        ".py": "🐍",
        ".csv": "📊",
        ".pptx": "🎠",
    }
    return icons.get(ext, "📄")

def _extract_json_object(text: str) -> str:
    if not text:
        return ""
    s = text.strip()
    if s.startswith("{") and s.endswith("}"):
        return s
    a = s.find("{")
    b = s.rfind("}")
    if a != -1 and b != -1 and b > a:
        return s[a : b + 1]
    return s

def validate_quiz_schema(obj: dict, expected_n: int) -> tuple[bool, str]:
    if not isinstance(obj, dict):
        return False, "Quiz JSON is not an object."
    if obj.get("num_questions") != expected_n:
        return False, f"num_questions mismatch"
    qs = obj.get("questions")
    if not isinstance(qs, list) or len(qs) != expected_n:
        return False, f"questions length mismatch"
    for i, q in enumerate(qs, start=1):
        if not isinstance(q, dict):
            return False, f"Question {i} is not an object"
        if q.get("id") != i:
            return False, f"Question id mismatch at {i}"
        if not isinstance(q.get("choices"), dict):
            return False, f"choices not dict at {i}"
        for k in ["A", "B", "C", "D"]:
            if k not in q["choices"] or not q["choices"][k].strip():
                return False, f"Invalid choice {k} at {i}"
        if q.get("answer") not in ["A", "B", "C", "D"]:
            return False, f"Invalid answer at {i}"
        if not q.get("question", "").strip():
            return False, f"Missing question text at {i}"
        if not q.get("explanation", "").strip():
            return False, f"Missing explanation at {i}"
    return True, "OK"

def parse_quiz_json(raw: str, expected_n: int) -> tuple[dict | None, str | None]:
    if not raw:
        return None, "Empty quiz output"
    candidate = _extract_json_object(raw)
    try:
        obj = json.loads(candidate)
    except Exception as e:
        return None, f"Invalid JSON: {e}"
    ok, reason = validate_quiz_schema(obj, expected_n)
    if not ok:
        return None, f"Schema invalid: {reason}"
    return obj, None

def get_chroma_index_summary():
    host = os.getenv("CHROMA_HOST", "localhost")
    port = int(os.getenv("CHROMA_PORT", "8000"))
    collection_name = os.getenv("CHROMA_COLLECTION", "ml_materials")
    try:
        import chromadb
        client = chromadb.HttpClient(host=host, port=port)
        col = client.get_or_create_collection(name=collection_name)
        n = col.count()
        sources = []
        try:
            got = col.get(include=["metadatas"], limit=min(50, max(1, n)))
            for md in got.get("metadatas", []) or []:
                if isinstance(md, dict) and md.get("source"):
                    sources.append(md["source"])
            sources = sorted(list(set(sources)))
        except Exception:
            sources = []
        return {"ok": True, "collection": collection_name, "count": n, "sources": sources[:25], "error": None}
    except Exception as e:
        return {"ok": False, "collection": collection_name, "count": None, "sources": [], "error": str(e)}

# Session state
def init_session_state():
    defaults = {
        "page": "chat",
        "sessions": {},
        "current_session_id": None,
        "uploaded_docs": _load_uploaded_docs(),
        "total_questions": 0,
        "total_quizzes": 0,
        "quiz_topic": "gradient descent",
        "quiz_num_questions": 5,
        "quiz_raw_output": None,
        "quiz_obj": None,
        "quiz_answers": {},
        "quiz_submitted": False,
        "quiz_score": None,
        "llm_provider": os.getenv("ACTIVE_LLM_PROVIDER", "ollama"),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v
    if not st.session_state.sessions:
        st.session_state.current_session_id = create_new_session()

def create_new_session() -> str:
    session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    st.session_state.sessions[session_id] = {
        "title": "New Session",
        "created_at": datetime.now().isoformat(),
        "messages": [],
    }
    return session_id

def get_messages() -> list[dict]:
    sid = st.session_state.current_session_id
    return st.session_state.sessions.get(sid, {}).get("messages", [])

def add_message(role: str, content: str) -> None:
    sid = st.session_state.current_session_id
    st.session_state.sessions[sid]["messages"].append(
        {"role": role, "content": content, "ts": datetime.now().isoformat()}
    )
    if role == "user":
        st.session_state.total_questions += 1
        if st.session_state.sessions[sid]["title"] == "New Session":
            st.session_state.sessions[sid]["title"] = (content[:40] + "...") if len(content) > 40 else content

# Sidebar
def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="header-section">', unsafe_allow_html=True)
        st.markdown("## 🧠 ML Learning Assistant")
        st.markdown(
            f'<div style="margin: 1rem 0;">'
            f'<span class="status-indicator"></span>'
            f'<span style="color: var(--text-secondary); font-size: 0.875rem;">System Active</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        # LLM Provider Selection
        st.markdown('<div class="provider-selector">', unsafe_allow_html=True)
        st.markdown("#### 🤖 LLM Provider")

        provider = st.selectbox(
            "Select Provider",
            options=["ollama", "groq", "cerebras"],
            index=["ollama", "groq", "cerebras"].index(st.session_state.llm_provider.lower()),
            key="provider_select",
            label_visibility="collapsed"
        )

        if provider != st.session_state.llm_provider:
            st.session_state.llm_provider = provider
            os.environ["ACTIVE_LLM_PROVIDER"] = provider
            reset_crew()
            st.success(f"Switched to {provider.capitalize()}")
            st.rerun()

        st.markdown(
            f'<span class="badge">{provider.capitalize()}</span>'
            f'<span class="badge-secondary">MCP: Docker</span>',
            unsafe_allow_html=True
        )
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("---")

        # Controls
        st.markdown("### ⚙️ Controls")
        if st.button("🔄 Reset Assistant", use_container_width=True):
            reset_crew()
            st.success("Assistant reset successfully!")
            time.sleep(0.5)
            st.rerun()

        st.caption(f"Collection: `{os.getenv('CHROMA_COLLECTION', 'ml_materials')}`")

        st.markdown("---")

        # Navigation
        st.markdown("### 🧭 Navigation")
        pages = {
            "chat": {"icon": "💬", "label": "Chat"},
            "upload": {"icon": "📚", "label": "Upload"},
            "quiz": {"icon": "📝", "label": "Quiz"},
            "stats": {"icon": "📊", "label": "Stats"}
        }

        cols = st.columns(2)
        for idx, (page_key, page_info) in enumerate(pages.items()):
            with cols[idx % 2]:
                if st.button(
                    f"{page_info['icon']} {page_info['label']}",
                    use_container_width=True,
                    type="primary" if st.session_state.page == page_key else "secondary"
                ):
                    st.session_state.page = page_key
                    st.rerun()

        st.markdown("---")

        # Sessions
        st.markdown("### 💬 Chat Sessions")
        col1, col2 = st.columns([3, 1])
        with col1:
            if st.button("➕ New Session", use_container_width=True):
                st.session_state.current_session_id = create_new_session()
                st.session_state.page = "chat"
                st.rerun()
        with col2:
            if st.button("🗑️", help="Clear all sessions", use_container_width=True):
                st.session_state.sessions = {}
                st.session_state.current_session_id = create_new_session()
                st.rerun()

        for sid in sorted(st.session_state.sessions.keys(), reverse=True)[:6]:
            s = st.session_state.sessions[sid]
            is_current = sid == st.session_state.current_session_id
            if st.button(
                f"{'▸' if is_current else '•'} {s['title'][:28]}",
                key=f"sess_{sid}",
                use_container_width=True,
                type="primary" if is_current else "secondary"
            ):
                st.session_state.current_session_id = sid
                st.session_state.page = "chat"
                st.rerun()

        st.markdown("---")

        # Quick Stats
        st.markdown("### 📈 Quick Stats")
        col1, col2 = st.columns(2)
        col1.metric("Questions", st.session_state.total_questions)
        col2.metric("Quizzes", st.session_state.total_quizzes)
        st.metric("Docs Uploaded", len(st.session_state.uploaded_docs))

# Pages
def render_chat_page():
    st.markdown('<div class="header-section"><h1>💬 Chat Assistant</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: var(--text-secondary); font-size: 1.125rem; margin-bottom: 2rem;">'
        'Ask questions about ML/NLP. The assistant uses your uploaded documents and web search when needed.'
        '</p>',
        unsafe_allow_html=True
    )

    msgs = get_messages()
    if not msgs:
        st.markdown(
            '<div class="glass-card">'
            '<h3>👋 Welcome!</h3>'
            '<p style="color: var(--text-muted);">Start by asking a question about Machine Learning or NLP. '
            'Try: "Explain the attention mechanism in Transformers" or "What is gradient descent?"</p>'
            '</div>',
            unsafe_allow_html=True
        )
    else:
        for m in msgs:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if prompt := st.chat_input("Ask about ML/NLP..."):
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("🔍 Analyzing your question..."):
                try:
                    crew = get_crew()
                    t0 = time.time()
                    ans = crew.ask_question(query=prompt, topic=prompt)
                    dt = time.time() - t0
                    st.markdown(ans)
                    st.caption(f"⏱️ Response time: {dt:.1f}s")
                    add_message("assistant", ans)
                except Exception as e:
                    msg = f"❌ Error: {str(e)[:200]}"
                    st.error(msg)
                    add_message("assistant", msg)

def render_upload_page():
    st.markdown('<div class="header-section"><h1>📚 Document Management</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: var(--text-secondary); font-size: 1.125rem; margin-bottom: 2rem;">'
        'Upload and index documents in multiple formats to enhance the knowledge base.'
        '</p>',
        unsafe_allow_html=True
    )

    # Format support badges
    st.markdown(
        '<div style="margin-bottom: 1.5rem;">'
        '<span class="badge-success">📕 PDF</span>'
        '<span class="badge-success">📄 TXT</span>'
        '<span class="badge-success">📝 MD</span>'
        '<span class="badge-success">📘 DOCX</span>'
        '<span class="badge-success">🐍 PY</span>'
        '<span class="badge-success">📊 CSV</span>'
        '<span class="badge-success">🎠 PPTX</span>'
        '</div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.4, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card-strong">', unsafe_allow_html=True)
        st.markdown("### 📤 Upload Documents")

        files = st.file_uploader(
            "Select files to index",
            type=["pdf", "txt", "md", "docx", "py", "csv", "pptx"],
            accept_multiple_files=True,
            help="Supported formats: PDF, TXT, Markdown, Word, Python, CSV, PowerPoint"
        )

        st.markdown('</div>', unsafe_allow_html=True)

        if files:
            st.info(f"📁 {len(files)} file(s) selected • Ready to index")

            with st.expander("📋 Preview Selected Files"):
                for f in files:
                    icon = get_file_icon(f.name)
                    size_kb = f.size / 1024
                    st.markdown(f"{icon} **{f.name}** ({size_kb:.1f} KB)")

            if st.button("🚀 Index All Files", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                ok = 0
                total_chunks = 0

                for idx, uf in enumerate(files):
                    progress = (idx + 1) / len(files)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing {idx + 1}/{len(files)}: {uf.name}")

                    path = UPLOAD_DIR / uf.name
                    path.write_bytes(uf.getbuffer())

                    with st.spinner(f"Indexing {uf.name}..."):
                        res = upload_document_to_chromadb(str(path))

                    if res.get("success"):
                        ok += 1
                        total_chunks += int(res.get("chunks", 0))
                        if uf.name not in st.session_state.uploaded_docs:
                            st.session_state.uploaded_docs.append(uf.name)

                        icon = get_file_icon(uf.name)
                        pages_label = "Slides" if uf.name.endswith(".pptx") else "Pages" if uf.name.endswith(".pdf") else "Rows" if uf.name.endswith(".csv") else "Sections"
                        st.success(
                            f"✅ {icon} **{uf.name}** • "
                            f"{pages_label}: {res.get('pages')} • "
                            f"Chunks: {res.get('chunks')}"
                        )
                    else:
                        st.error(f"❌ {uf.name} • {res.get('message')}")

                _save_uploaded_docs(st.session_state.uploaded_docs)
                progress_bar.progress(1.0)
                status_text.empty()

                if ok == len(files):
                    st.balloons()
                st.success(
                    f"🎉 **Indexing Complete!** • "
                    f"Success: {ok}/{len(files)} • "
                    f"Total Chunks: {total_chunks}"
                )

        st.markdown("---")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📑 Indexed Documents")

        if st.session_state.uploaded_docs:
            docs_by_type = {}
            for name in st.session_state.uploaded_docs:
                ext = Path(name).suffix.lower()
                if ext not in docs_by_type:
                    docs_by_type[ext] = []
                docs_by_type[ext].append(name)

            for ext in sorted(docs_by_type.keys()):
                icon = get_file_icon(f"file{ext}")
                with st.expander(f"{icon} {ext.upper()} Files ({len(docs_by_type[ext])})"):
                    for name in sorted(docs_by_type[ext]):
                        st.markdown(f"- {name}")
        else:
            st.info("💡 No documents indexed yet. Upload files above to get started!")

        if st.button("🗑️ Clear Document List", use_container_width=True):
            st.session_state.uploaded_docs = []
            _save_uploaded_docs([])
            st.success("Document list cleared")
            st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card-strong">', unsafe_allow_html=True)
        st.markdown("### 🗄️ ChromaDB Status")
        summary = get_chroma_index_summary()

        if summary["ok"]:
            st.metric("Collection", summary["collection"])
            st.metric("Total Chunks", summary["count"])

            if summary["count"] and summary["count"] > 0:
                st.markdown(f'<span class="badge-success">✓ Active</span>', unsafe_allow_html=True)
            else:
                st.warning("Collection is empty. Upload documents to populate.")

            if summary["sources"]:
                with st.expander(f"📚 Indexed Sources ({len(summary['sources'])})"):
                    for s in summary["sources"]:
                        icon = get_file_icon(s)
                        st.markdown(f"{icon} {s}")
        else:
            st.error("⚠️ Could not connect to ChromaDB")
            with st.expander("Error Details"):
                st.code(summary["error"] or "Unknown error")

        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("### 💡 Indexing Tips")
        st.markdown("""
        - **PDF**: Best for lecture slides and papers
        - **PPTX**: PowerPoint presentations (all slides)
        - **TXT/MD**: Fast indexing for notes
        - **DOCX**: Word documents supported
        - **PY**: Index code examples and scripts
        - **CSV**: Tabular data becomes searchable

        📌 Files are chunked into 1000-character segments with 200-character overlap for optimal retrieval.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

def render_quiz_page():
    st.markdown('<div class="header-section"><h1>📝 Interactive Quiz</h1></div>', unsafe_allow_html=True)
    st.markdown(
        '<p style="color: var(--text-secondary); font-size: 1.125rem; margin-bottom: 2rem;">'
        'Generate custom quizzes on any ML/NLP topic with instant feedback.'
        '</p>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.5, 1], gap="large")

    with col1:
        st.markdown('<div class="glass-card-strong">', unsafe_allow_html=True)
        st.markdown("### 🎯 Quiz Configuration")

        st.session_state.quiz_topic = st.text_input(
            "Topic",
            value=st.session_state.quiz_topic,
            placeholder="e.g., Attention Mechanisms"
        )
        st.session_state.quiz_num_questions = st.slider(
            "Number of Questions",
            3, 10,
            int(st.session_state.quiz_num_questions)
        )

        col_a, col_b = st.columns([0.65, 0.35])
        with col_a:
            if st.button("🚀 Generate Quiz", use_container_width=True, type="primary"):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_obj = None
                st.session_state.quiz_raw_output = None

                with st.spinner("🧠 Generating quiz..."):
                    crew = get_crew()
                    raw = crew.generate_quiz(
                        topic=st.session_state.quiz_topic,
                        num_questions=int(st.session_state.quiz_num_questions)
                    )
                    st.session_state.quiz_raw_output = raw

                obj, err = parse_quiz_json(
                    st.session_state.quiz_raw_output or "",
                    expected_n=int(st.session_state.quiz_num_questions)
                )
                if err:
                    st.error(f"❌ {err}")
                else:
                    st.session_state.quiz_obj = obj
                    st.session_state.total_quizzes += 1
                    st.success("✅ Quiz generated successfully!")

        with col_b:
            if st.button("🗑️ Clear", use_container_width=True):
                for key in ["quiz_raw_output", "quiz_obj", "quiz_answers", "quiz_submitted", "quiz_score"]:
                    st.session_state[key] = None if "score" in key else {} if "answers" in key else False if "submitted" in key else None
                st.rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.quiz_obj:
            quiz = st.session_state.quiz_obj
            st.markdown('<div class="glass-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            st.markdown(f"### 📋 Quiz: {quiz.get('topic', 'Unknown Topic')}")

            for q in quiz["questions"]:
                qid = q["id"]
                st.markdown(f"**Question {qid}:** {q['question']}")

                options = ["A", "B", "C", "D"]
                labels = [f"{k}) {q['choices'][k]}" for k in options]
                current = st.session_state.quiz_answers.get(str(qid))
                idx = options.index(current) if current in options else 0

                picked_label = st.radio(
                    "",
                    options=labels,
                    index=idx,
                    key=f"quiz_{qid}",
                    horizontal=False
                )
                picked_key = picked_label.split(")")[0].strip()
                st.session_state.quiz_answers[str(qid)] = picked_key
                st.markdown("---")

            col_s1, col_s2 = st.columns(2)
            with col_s1:
                if st.button("✅ Submit Quiz", use_container_width=True, type="primary"):
                    score = sum(1 for q in quiz["questions"] if st.session_state.quiz_answers.get(str(q["id"])) == q["answer"])
                    st.session_state.quiz_score = score
                    st.session_state.quiz_submitted = True
                    st.rerun()

            with col_s2:
                st.download_button(
                    "💾 Download JSON",
                    data=json.dumps(quiz, indent=2).encode("utf-8"),
                    file_name=f"quiz_{quiz.get('topic', 'quiz').replace(' ', '_')}.json",
                    mime="application/json",
                    use_container_width=True
                )

            if st.session_state.quiz_submitted:
                st.markdown("---")
                n = int(quiz["num_questions"])
                score = int(st.session_state.quiz_score or 0)
                percentage = (score / n) * 100

                st.markdown('<div class="glass-card-strong">', unsafe_allow_html=True)
                st.markdown("### 🎯 Results")

                col_r1, col_r2, col_r3 = st.columns(3)
                col_r1.metric("Score", f"{score}/{n}")
                col_r2.metric("Percentage", f"{percentage:.0f}%")
                col_r3.metric("Grade", "A" if percentage >= 90 else "B" if percentage >= 80 else "C" if percentage >= 70 else "D" if percentage >= 60 else "F")

                if percentage >= 80:
                    st.success("🌟 Excellent work!")
                elif percentage >= 60:
                    st.info("👍 Good job!")
                else:
                    st.warning("📚 Keep studying!")

                with st.expander("📖 Review Answers"):
                    for q in quiz["questions"]:
                        qid = str(q["id"])
                        chosen = st.session_state.quiz_answers.get(qid)
                        correct = q["answer"]
                        is_correct = chosen == correct

                        st.markdown(f"**Q{qid}:** {q['question']}")
                        if is_correct:
                            st.success(f"✅ Your answer: {chosen}")
                        else:
                            st.error(f"❌ Your answer: {chosen} | Correct: {correct}")
                        st.info(f"💡 {q['explanation']}")
                        st.markdown("---")
                st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 💡 Tips for Better Quizzes")
        st.markdown("""
        - 🎯 **Be Specific**: Use precise topics like "LSTM vs GRU"
        - 📚 **Upload First**: Index relevant documents for accurate questions
        - 🔍 **Check Collection**: Ensure ChromaDB has your materials
        - 🧠 **Topic Examples**:
          - "Attention mechanisms in Transformers"
          - "Backpropagation algorithm"
          - "Word2Vec vs GloVe embeddings"
          - "Gradient descent optimization"
        """)
        st.markdown('</div>', unsafe_allow_html=True)

        if st.session_state.total_quizzes > 0:
            st.markdown('<div class="glass-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
            st.markdown("### 📊 Quiz Statistics")
            st.metric("Total Quizzes Generated", st.session_state.total_quizzes)
            if st.session_state.quiz_score is not None:
                st.metric("Last Score", f"{st.session_state.quiz_score}/{st.session_state.quiz_num_questions}")
            st.markdown('</div>', unsafe_allow_html=True)

def render_stats_page():
    st.markdown('<div class="header-section"><h1>📊 Statistics & System Info</h1></div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("💬 Questions Asked", st.session_state.total_questions)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("📝 Quizzes Generated", st.session_state.total_quizzes)
        st.markdown('</div>', unsafe_allow_html=True)

    with col3:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("📄 Docs Uploaded", len(st.session_state.uploaded_docs))
        st.markdown('</div>', unsafe_allow_html=True)

    with col4:
        chroma = get_chroma_index_summary()
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.metric("🗄️ Indexed Chunks", chroma["count"] if chroma["ok"] else "—")
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")

    col_a, col_b = st.columns(2, gap="large")

    with col_a:
        st.markdown('<div class="glass-card-strong">', unsafe_allow_html=True)
        st.markdown("### 🔧 System Configuration")
        config_data = {
            "LLM Provider": st.session_state.llm_provider.capitalize(),
            "Ollama URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            "ChromaDB Host": os.getenv("CHROMA_HOST", "localhost"),
            "ChromaDB Port": os.getenv("CHROMA_PORT", "8000"),
            "Collection": os.getenv("CHROMA_COLLECTION", "ml_materials"),
            "Storage Dir": os.getenv("CREWAI_STORAGE_DIR", "N/A")[:50] + "...",
        }
        for key, value in config_data.items():
            st.markdown(f"**{key}:** `{value}`")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("### 📁 Document Types")
        if st.session_state.uploaded_docs:
            type_counts = {}
            for doc in st.session_state.uploaded_docs:
                ext = Path(doc).suffix.upper() or "OTHER"
                type_counts[ext] = type_counts.get(ext, 0) + 1

            for ext, count in sorted(type_counts.items()):
                icon = get_file_icon(f"file{ext.lower()}")
                st.markdown(f"{icon} **{ext}**: {count} file(s)")
        else:
            st.info("No documents uploaded yet")
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="glass-card-strong">', unsafe_allow_html=True)
        st.markdown("### 💬 Session Overview")
        st.metric("Active Sessions", len(st.session_state.sessions))
        st.metric("Current Session", st.session_state.current_session_id.split("_")[-1] if st.session_state.current_session_id else "None")

        if st.session_state.sessions:
            with st.expander("📋 Session Details"):
                for sid, sess in list(st.session_state.sessions.items())[:5]:
                    st.markdown(f"**{sess['title'][:30]}**")
                    st.caption(f"Messages: {len(sess['messages'])} | Created: {sess['created_at'][:19]}")
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card" style="margin-top: 1.5rem;">', unsafe_allow_html=True)
        st.markdown("### 🗄️ ChromaDB Details")
        chroma = get_chroma_index_summary()
        if chroma["ok"]:
            st.markdown(f"**Status:** <span class='badge-success'>✓ Connected</span>", unsafe_allow_html=True)
            st.markdown(f"**Collection:** {chroma['collection']}")
            st.markdown(f"**Total Chunks:** {chroma['count']}")
            st.markdown(f"**Unique Sources:** {len(chroma['sources'])}")
        else:
            st.error("⚠️ Not connected")
        st.markdown('</div>', unsafe_allow_html=True)

# Main app
def main():
    init_session_state()
    render_sidebar()

    pages = {
        "chat": render_chat_page,
        "upload": render_upload_page,
        "quiz": render_quiz_page,
        "stats": render_stats_page,
    }

    page_func = pages.get(st.session_state.page, render_chat_page)
    page_func()

if __name__ == "__main__":
    main()
