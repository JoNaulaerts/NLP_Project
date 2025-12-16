"""
ML Learning Assistant - Streamlit UI (MCP / STDIO)

Features
- Chat: CrewAI assistant (RAG via Docker MCP Gateway)
- Upload: index PDFs into ChromaDB (collection: CHROMA_COLLECTION, default ml_materials)
- Quiz: generates quizzes in strict JSON and renders interactive quiz UI + scoring + download JSON
- Stats: shows counters + Chroma collection count
- Debug: shows recent test artifacts and tool logs (best-effort)

Run:
  streamlit run app_new.py
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


# -----------------------------
# Load env early + force CrewAI storage path early
# -----------------------------
load_dotenv()

# Strongly recommended: prevent terminal prompts from blocking Streamlit
os.environ.setdefault("CREWAI_TELEMETRY", "false")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")  # see CrewAI docs

# Ensure default collection is the one you want
os.environ.setdefault("CHROMA_COLLECTION", "ml_materials")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Force project-local CrewAI storage (must be set before CrewAI initializes)
CREWAI_STORE = (DATA_DIR / "crewai_memory").resolve()
os.environ["CREWAI_STORAGE_DIR"] = str(CREWAI_STORE)

# Imports after env setup
from src.ml_learning_assistant.crew import MLLearningAssistantCrew
from src.ml_learning_assistant.tools.upload_to_chromadb import upload_pdf_to_chromadb


# -----------------------------
# App paths
# -----------------------------
APP_STATE_DIR = DATA_DIR / "ui_state"
APP_STATE_DIR.mkdir(parents=True, exist_ok=True)

UPLOAD_DIR = DATA_DIR / "uploaded_docs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

UPLOADED_TRACK_FILE = APP_STATE_DIR / "uploaded_docs.json"

# Support both folders:
# - your UI local logs
# - the acceptance test artifact folder
TEST_ARTIFACTS_DIR_A = DATA_DIR / "test_artifacts"
TEST_ARTIFACTS_DIR_B = DATA_DIR / "final_e2e_artifacts"

TEST_LOGS_DIR_A = TEST_ARTIFACTS_DIR_A / "logs"
TEST_OUTPUTS_DIR_A = TEST_ARTIFACTS_DIR_A / "outputs"
TEST_LOGS_DIR_B = TEST_ARTIFACTS_DIR_B / "logs"
TEST_OUTPUTS_DIR_B = TEST_ARTIFACTS_DIR_B / "outputs"

TOOL_TRACE_PATH = Path(os.getenv("APP_TOOL_LOG_PATH", DATA_DIR / "tool_traces" / "tool_trace.jsonl"))
TOOL_TRACE_PATH.parent.mkdir(parents=True, exist_ok=True)


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="ML Learning Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# -----------------------------
# Styling (clean, glassy, modern)
# -----------------------------
st.markdown(
    """
<style>
:root{
  --bg1:#070b16;
  --bg2:#0b1630;
  --panel: rgba(255,255,255,.07);
  --panel2: rgba(255,255,255,.10);
  --border: rgba(255,255,255,.12);
  --text: rgba(255,255,255,.92);
  --muted: rgba(255,255,255,.64);
  --accent: #7c5cff;
  --accent2:#30cfd0;
}

.stApp{
  background:
    radial-gradient(1200px 800px at 18% 8%, rgba(124,92,255,.22) 0%, transparent 55%),
    radial-gradient(900px 700px at 82% 28%, rgba(48,207,208,.18) 0%, transparent 55%),
    linear-gradient(135deg, var(--bg1) 0%, var(--bg2) 100%);
  color: var(--text);
}

h1,h2,h3 { color: var(--text) !important; letter-spacing:-.02em; }
p, li, label, small, .stMarkdown { color: var(--text) !important; }

hr{ border:none; border-top:1px solid var(--border); margin: 1rem 0; }

.card{
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
}

.card-strong{
  background: var(--panel2);
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 16px;
}

.badge{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(124,92,255,.18);
  border: 1px solid rgba(124,92,255,.38);
  color: var(--text);
  margin-right: 8px;
}

.badge2{
  display:inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 12px;
  background: rgba(48,207,208,.14);
  border: 1px solid rgba(48,207,208,.35);
  color: var(--text);
}

.muted{ color: var(--muted) !important; }

section[data-testid="stSidebar"]{
  background: rgba(5,10,20,.55);
  backdrop-filter: blur(12px);
  border-right: 1px solid var(--border);
}

.stChatMessage{
  background: rgba(255,255,255,.06) !important;
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 14px 14px;
}

.stButton > button, .stDownloadButton > button{
  border-radius: 14px !important;
  font-weight: 650 !important;
  border: 1px solid rgba(255,255,255,.14) !important;
}

.stButton > button:hover{ border-color: rgba(124,92,255,.5) !important; }

div[data-testid="stMetricValue"]{ color: var(--text) !important; }
div[data-testid="stMetricLabel"]{ color: var(--muted) !important; }

.codebox{
  background: rgba(0,0,0,.25);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 12px 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Persistence helpers
# -----------------------------
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


# -----------------------------
# Crew (cached resource)
# -----------------------------
@st.cache_resource
def get_crew() -> MLLearningAssistantCrew:
    # IMPORTANT: if your crew.py builds Crew(...), make sure it passes tracing=False there too.
    return MLLearningAssistantCrew()


def reset_crew():
    # Best-effort close, then clear cache.
    try:
        c = get_crew()
        if hasattr(c, "close"):
            c.close()
    except Exception:
        pass
    st.cache_resource.clear()


# -----------------------------
# Quiz JSON helpers
# -----------------------------
def _extract_json_object(text: str) -> str:
    """
    If the model returns extra text, try to extract the first {...} block.
    """
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
        return False, f"num_questions mismatch: got {obj.get('num_questions')} expected {expected_n}"

    qs = obj.get("questions")
    if not isinstance(qs, list) or len(qs) != expected_n:
        return False, f"questions length mismatch: got {len(qs) if isinstance(qs, list) else type(qs)} expected {expected_n}"

    for i, q in enumerate(qs, start=1):
        if not isinstance(q, dict):
            return False, f"Question {i} is not an object."
        if q.get("id") != i:
            return False, f"Question id mismatch at {i}: got {q.get('id')}"
        if not isinstance(q.get("choices"), dict):
            return False, f"choices not dict at {i}"
        for k in ["A", "B", "C", "D"]:
            if k not in q["choices"]:
                return False, f"Missing choice {k} at {i}"
            if not isinstance(q["choices"][k], str) or not q["choices"][k].strip():
                return False, f"Empty choice {k} at {i}"
        if q.get("answer") not in ["A", "B", "C", "D"]:
            return False, f"Invalid answer at {i}: {q.get('answer')}"
        if not isinstance(q.get("question"), str) or not q["question"].strip():
            return False, f"Missing question text at {i}"
        if not isinstance(q.get("explanation"), str) or not q["explanation"].strip():
            return False, f"Missing explanation at {i}"

    return True, "OK"


def parse_quiz_json(raw: str, expected_n: int) -> tuple[dict | None, str | None]:
    if not raw:
        return None, "Empty quiz output."

    candidate = _extract_json_object(raw)
    try:
        obj = json.loads(candidate)
    except Exception as e:
        return None, f"Invalid JSON: {e}"

    ok, reason = validate_quiz_schema(obj, expected_n)
    if not ok:
        return None, f"Schema invalid: {reason}"

    return obj, None


# -----------------------------
# Session state
# -----------------------------
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
        "quiz_raw_output": None,     # raw string
        "quiz_obj": None,            # parsed JSON dict
        "quiz_answers": {},          # {"1":"A", ...}
        "quiz_submitted": False,
        "quiz_score": None,
        "last_error": None,
        "debug_show": False,
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
            st.session_state.sessions[sid]["title"] = (content[:42] + "...") if len(content) > 42 else content


# -----------------------------
# Chroma summary
# -----------------------------
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


# -----------------------------
# Debug artifacts readers
# -----------------------------
def read_tail(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="ignore")
    return text[-max_chars:]


def list_recent_files(folder: Path, suffix: str | None = None, limit: int = 10) -> list[Path]:
    if not folder.exists():
        return []
    files = [p for p in folder.glob("*") if p.is_file()]
    if suffix:
        files = [p for p in files if p.name.endswith(suffix)]
    files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return files[:limit]


# -----------------------------
# Sidebar
# -----------------------------
def provider_badge() -> str:
    p = os.getenv("ACTIVE_LLM_PROVIDER", "ollama").lower().strip()
    return "Ollama" if p not in ("groq", "cerebras") else p.capitalize()


def render_sidebar():
    with st.sidebar:
        st.markdown("## 🧠 ML Learning Assistant")
        st.markdown(
            f"<span class='badge'>LLM: {provider_badge()}</span>"
            f"<span class='badge2'>MCP: Docker STDIO</span>",
            unsafe_allow_html=True,
        )
        st.markdown("<p class='muted'>RAG + Web tools via Docker MCP gateway.</p>", unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("### ⚙️ Controls")
        if st.button("Reset assistant", use_container_width=True, help="Recreate the Crew instance (useful after errors)"):
            reset_crew()
            st.rerun()

        st.caption(f"CHROMA_COLLECTION: {os.getenv('CHROMA_COLLECTION', 'ml_materials')}")

        st.markdown("---")
        st.markdown("### 🧭 Navigation")
        r1, r2 = st.columns(2)
        with r1:
            if st.button("Chat", use_container_width=True, type="primary" if st.session_state.page == "chat" else "secondary"):
                st.session_state.page = "chat"; st.rerun()
            if st.button("Upload", use_container_width=True, type="primary" if st.session_state.page == "upload" else "secondary"):
                st.session_state.page = "upload"; st.rerun()
        with r2:
            if st.button("Quiz", use_container_width=True, type="primary" if st.session_state.page == "quiz" else "secondary"):
                st.session_state.page = "quiz"; st.rerun()
            if st.button("Stats", use_container_width=True, type="primary" if st.session_state.page == "stats" else "secondary"):
                st.session_state.page = "stats"; st.rerun()

        st.markdown("---")
        st.markdown("### 💬 Sessions")
        a, b = st.columns([3, 1])
        with a:
            if st.button("New session", use_container_width=True):
                st.session_state.current_session_id = create_new_session()
                st.session_state.page = "chat"
                st.rerun()
        with b:
            if st.button("🗑️", help="Clear all sessions"):
                st.session_state.sessions = {}
                st.session_state.current_session_id = create_new_session()
                st.rerun()

        for sid in sorted(st.session_state.sessions.keys(), reverse=True)[:8]:
            s = st.session_state.sessions[sid]
            is_current = sid == st.session_state.current_session_id
            if st.button(
                f"{'▸' if is_current else '•'} {s['title'][:30]}",
                key=f"sess_{sid}",
                use_container_width=True,
                type="primary" if is_current else "secondary",
            ):
                st.session_state.current_session_id = sid
                st.session_state.page = "chat"
                st.rerun()

        st.markdown("---")
        st.markdown("### 📈 Quick stats")
        st.metric("Questions", st.session_state.total_questions)
        st.metric("Quizzes", st.session_state.total_quizzes)
        st.metric("Uploaded PDFs", len(st.session_state.uploaded_docs))

        st.markdown("---")
        st.session_state.debug_show = st.toggle("Show Debug Panel", value=st.session_state.debug_show)


# -----------------------------
# Pages
# -----------------------------
def render_chat_page():
    st.markdown("## 💬 Chat")
    st.markdown("<p class='muted'>Ask ML/NLP questions. Documents are used first; web is fallback.</p>", unsafe_allow_html=True)

    msgs = get_messages()
    if not msgs:
        st.markdown(
            """
<div class="card">
  <h3>Welcome</h3>
  <p class="muted">Try: “Summarize my uploaded lecture slides about Transformers” or “Explain attention vs self-attention”.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        for m in msgs:
            with st.chat_message(m["role"]):
                st.markdown(m["content"])

    if prompt := st.chat_input("Ask something about Machine Learning / NLP..."):
        add_message("user", prompt)
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    crew = get_crew()
                    t0 = time.time()
                    ans = crew.ask_question(query=prompt, topic=prompt)
                    dt = time.time() - t0
                    st.caption(f"Response time: {dt:.1f}s")
                    st.markdown(ans)
                    add_message("assistant", ans)
                except Exception as e:
                    msg = f"❌ Error: {str(e)[:280]}"
                    st.error(msg)
                    add_message("assistant", msg)


def render_upload_page():
    st.markdown("## 📚 Upload & Index")
    st.markdown("<p class='muted'>Upload PDFs to index them into ChromaDB for RAG.</p>", unsafe_allow_html=True)

    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        st.markdown("<div class='card-strong'><h3>Upload PDFs</h3></div>", unsafe_allow_html=True)
        files = st.file_uploader("Choose PDF(s)", type=["pdf"], accept_multiple_files=True)

        if files:
            st.caption("Files are saved to ./data/uploaded_docs and then indexed into ChromaDB.")
            if st.button("Index selected PDFs", use_container_width=True, type="primary"):
                ok = 0
                total_chunks = 0
                for uf in files:
                    path = UPLOAD_DIR / uf.name
                    path.write_bytes(uf.getbuffer())
                    with st.spinner(f"Indexing {uf.name}..."):
                        res = upload_pdf_to_chromadb(str(path))

                    if res.get("success"):
                        ok += 1
                        total_chunks += int(res.get("chunks", 0))
                        if uf.name not in st.session_state.uploaded_docs:
                            st.session_state.uploaded_docs.append(uf.name)
                        st.success(f"{uf.name} — pages={res.get('pages')} chunks={res.get('chunks')}")
                    else:
                        st.error(f"{uf.name} — {res.get('message')}")

                _save_uploaded_docs(st.session_state.uploaded_docs)
                st.info(f"Indexed {ok}/{len(files)} PDFs • chunks added: {total_chunks}")

        st.markdown("---")
        st.markdown("<div class='card'><h3>Uploaded files (tracked)</h3></div>", unsafe_allow_html=True)
        if st.session_state.uploaded_docs:
            for name in sorted(st.session_state.uploaded_docs):
                st.write(f"- {name}")
        else:
            st.write("No uploaded files tracked yet.")

        if st.button("Clear uploaded list (UI only)", use_container_width=True):
            st.session_state.uploaded_docs = []
            _save_uploaded_docs([])
            st.rerun()

    with right:
        st.markdown("<div class='card-strong'><h3>ChromaDB status</h3></div>", unsafe_allow_html=True)
        summary = get_chroma_index_summary()
        if summary["ok"]:
            st.metric("Collection", summary["collection"])
            st.metric("Chunks", summary["count"])
            if summary["sources"]:
                st.caption("Sources (best effort):")
                for s in summary["sources"]:
                    st.write(f"- {s}")
        else:
            st.error("Could not query ChromaDB.")
            st.code(summary["error"] or "Unknown error", language="text")


def render_quiz_page():
    st.markdown("## 📝 Quiz")
    st.markdown("<p class='muted'>Quiz output is strict JSON (rendered as an interactive quiz UI).</p>", unsafe_allow_html=True)

    c1, c2 = st.columns([0.65, 0.35], gap="large")

    with c1:
        st.markdown("<div class='card-strong'><h3>Create quiz</h3></div>", unsafe_allow_html=True)

        st.session_state.quiz_topic = st.text_input("Topic", value=st.session_state.quiz_topic)
        st.session_state.quiz_num_questions = st.slider("Questions", 3, 10, int(st.session_state.quiz_num_questions))

        a, b = st.columns([0.6, 0.4])
        with a:
            if st.button("Generate quiz", use_container_width=True, type="primary"):
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_obj = None
                st.session_state.quiz_raw_output = None

                with st.spinner("Generating quiz JSON..."):
                    crew = get_crew()
                    raw = crew.generate_quiz(
                        topic=st.session_state.quiz_topic,
                        num_questions=int(st.session_state.quiz_num_questions),
                    )
                    st.session_state.quiz_raw_output = raw

                obj, err = parse_quiz_json(
                    st.session_state.quiz_raw_output or "",
                    expected_n=int(st.session_state.quiz_num_questions),
                )
                if err:
                    st.error(err)
                else:
                    st.session_state.quiz_obj = obj
                    st.session_state.total_quizzes += 1
                    st.success("Quiz generated and validated.")

        with b:
            if st.button("Clear", use_container_width=True):
                st.session_state.quiz_raw_output = None
                st.session_state.quiz_obj = None
                st.session_state.quiz_answers = {}
                st.session_state.quiz_submitted = False
                st.session_state.quiz_score = None
                st.rerun()

        st.markdown("---")

        if st.session_state.quiz_obj:
            quiz = st.session_state.quiz_obj
            st.markdown("<div class='card'><h3>Take the quiz</h3></div>", unsafe_allow_html=True)

            for q in quiz["questions"]:
                qid = q["id"]
                st.markdown(f"**Q{qid}.** {q['question']}")

                options = ["A", "B", "C", "D"]
                labels = [f"{k}) {q['choices'][k]}" for k in options]
                current = st.session_state.quiz_answers.get(str(qid))
                idx = options.index(current) if current in options else 0

                picked_label = st.radio(
                    label="",
                    options=labels,
                    index=idx,
                    key=f"quiz_pick_{qid}",
                    horizontal=False,
                )
                picked_key = picked_label.split(")")[0].strip()
                st.session_state.quiz_answers[str(qid)] = picked_key
                st.divider()

            s1, s2 = st.columns([0.5, 0.5])
            with s1:
                if st.button("Submit quiz", use_container_width=True, type="primary"):
                    score = 0
                    for q in quiz["questions"]:
                        qid = str(q["id"])
                        if st.session_state.quiz_answers.get(qid) == q["answer"]:
                            score += 1
                    st.session_state.quiz_score = score
                    st.session_state.quiz_submitted = True

            with s2:
                st.download_button(
                    "Download quiz (.json)",
                    data=json.dumps(quiz, indent=2).encode("utf-8"),
                    file_name=f"quiz_{quiz['topic'].replace(' ', '_')}.json",
                    use_container_width=True,
                )

            if st.session_state.quiz_submitted:
                st.markdown("---")
                st.markdown("<div class='card-strong'><h3>Results</h3></div>", unsafe_allow_html=True)
                n = int(quiz["num_questions"])
                score = int(st.session_state.quiz_score or 0)
                st.metric("Score", f"{score}/{n}")

                with st.expander("Review answers"):
                    for q in quiz["questions"]:
                        qid = str(q["id"])
                        chosen = st.session_state.quiz_answers.get(qid)
                        correct = q["answer"]
                        st.write(f"Q{qid}: selected={chosen} • correct={correct}")
                        st.caption(q["explanation"])

            with st.expander("Raw JSON (debug)"):
                st.code(json.dumps(quiz, indent=2), language="json")

        elif st.session_state.quiz_raw_output:
            st.markdown("<div class='card'><h3>Raw output (not valid JSON)</h3></div>", unsafe_allow_html=True)
            st.code(st.session_state.quiz_raw_output, language="text")
        else:
            st.markdown("<div class='card'><p class='muted'>Generate a quiz to display it here.</p></div>", unsafe_allow_html=True)

    with c2:
        st.markdown("<div class='card-strong'><h3>Quality tips</h3></div>", unsafe_allow_html=True)
        st.markdown(
            """
<div class="card">
  <ul>
    <li>Use a specific topic (e.g., “Attention mechanism in Transformers”).</li>
    <li>Index slides/PDF first so RAG can answer from your materials.</li>
    <li>If Chroma is empty, Upload first or confirm CHROMA_COLLECTION is correct.</li>
  </ul>
</div>
""",
            unsafe_allow_html=True,
        )


def render_stats_page():
    st.markdown("## 📊 Stats")
    st.markdown("<p class='muted'>Usage counters + ChromaDB summary.</p>", unsafe_allow_html=True)

    a, b, c, d = st.columns(4)
    a.metric("Questions", st.session_state.total_questions)
    b.metric("Quizzes", st.session_state.total_quizzes)
    c.metric("Uploaded PDFs", len(st.session_state.uploaded_docs))
    chroma = get_chroma_index_summary()
    d.metric("Indexed chunks", chroma["count"] if chroma["ok"] else "—")

    st.markdown("---")
    st.markdown("<div class='card-strong'><h3>Environment</h3></div>", unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="card">
  <p class="muted">ACTIVE_LLM_PROVIDER: <code>{os.getenv("ACTIVE_LLM_PROVIDER","ollama")}</code></p>
  <p class="muted">OLLAMA_BASE_URL: <code>{os.getenv("OLLAMA_BASE_URL","http://localhost:11434")}</code></p>
  <p class="muted">CHROMA_HOST: <code>{os.getenv("CHROMA_HOST","localhost")}</code></p>
  <p class="muted">CHROMA_COLLECTION: <code>{os.getenv("CHROMA_COLLECTION","ml_materials")}</code></p>
  <p class="muted">CREWAI_STORAGE_DIR: <code>{os.getenv("CREWAI_STORAGE_DIR")}</code></p>
  <p class="muted">CREWAI_TRACING_ENABLED: <code>{os.getenv("CREWAI_TRACING_ENABLED","false")}</code></p>
</div>
""",
        unsafe_allow_html=True,
    )


def render_debug_panel():
    if not st.session_state.debug_show:
        return

    st.markdown("---")
    st.markdown("## 🧪 Debug Panel")
    st.markdown("<p class='muted'>Shows recent test artifacts + tool trace logs (best effort).</p>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("<div class='card-strong'><h3>Recent test runs</h3></div>", unsafe_allow_html=True)

        runs = list_recent_files(TEST_LOGS_DIR_B, suffix=".log", limit=6) + list_recent_files(TEST_LOGS_DIR_A, suffix=".log", limit=6)
        runs = sorted(list({p.resolve() for p in runs}), key=lambda p: p.stat().st_mtime, reverse=True)[:8]

        if not runs:
            st.write("No test logs found.")
        else:
            chosen = st.selectbox("Pick a run log", [p.name for p in runs], index=0)
            # find chosen path
            p = next((x for x in runs if x.name == chosen), runs[0])
            st.markdown("<div class='codebox'>", unsafe_allow_html=True)
            st.code(read_tail(p, 6000), language="text")
            st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='card-strong'><h3>Tool trace</h3></div>", unsafe_allow_html=True)
        if TOOL_TRACE_PATH.exists():
            st.caption(f"Showing tail of: {TOOL_TRACE_PATH}")
            st.markdown("<div class='codebox'>", unsafe_allow_html=True)
            st.code(read_tail(TOOL_TRACE_PATH, 6000), language="json")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("No tool trace found yet.")
            st.caption("Optional: implement tool-call tracing in your crew or point APP_TOOL_LOG_PATH to your trace file.")

        st.markdown("<div class='card-strong'><h3>Latest outputs</h3></div>", unsafe_allow_html=True)
        outs = list_recent_files(TEST_OUTPUTS_DIR_B, limit=6) + list_recent_files(TEST_OUTPUTS_DIR_A, limit=6)
        outs = sorted(list({p.resolve() for p in outs}), key=lambda p: p.stat().st_mtime, reverse=True)[:10]

        if outs:
            chosen2 = st.selectbox("Pick output file", [p.name for p in outs], index=0, key="outpick")
            p2 = next((x for x in outs if x.name == chosen2), outs[0])
            st.markdown("<div class='codebox'>", unsafe_allow_html=True)
            st.code(read_tail(p2, 6000), language="text")
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("No outputs found.")


# -----------------------------
# Main
# -----------------------------
def main():
    init_session_state()
    render_sidebar()

    if st.session_state.page == "chat":
        render_chat_page()
    elif st.session_state.page == "upload":
        render_upload_page()
    elif st.session_state.page == "quiz":
        render_quiz_page()
    elif st.session_state.page == "stats":
        render_stats_page()

    render_debug_panel()


if __name__ == "__main__":
    main()
