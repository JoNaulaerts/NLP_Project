"""
Streamlit Application - ML Learning Assistant
Main web interface for the multi-agent learning platform.
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="ML Learning Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stChat {
        padding: 20px;
    }
    .source-citation {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 10px;
    }
    .quiz-question {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "crew" not in st.session_state:
        st.session_state.crew = None
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False


def load_crew():
    """Load the ML Learning Crew."""
    # TODO: Initialize LLM and crew
    # This will be implemented when we configure Llama 3.2
    pass


def display_chat_history():
    """Display chat message history."""
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])


def process_user_input(user_input: str):
    """Process user input and generate response."""
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # TODO: Integrate with crew
            # For now, placeholder response
            response = f"I received your query: '{user_input}'. The multi-agent system will process this once fully configured."
            st.markdown(response)
    
    # Add assistant response to history
    st.session_state.messages.append({
        "role": "assistant",
        "content": response
    })


def sidebar():
    """Render sidebar with settings and options."""
    with st.sidebar:
        st.title("🤖 ML Learning Assistant")
        st.markdown("---")
        
        # Mode selection
        st.subheader("Mode")
        mode = st.radio(
            "Select interaction mode:",
            ["💬 Chat & Learn", "📝 Quiz Mode"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        
        # Document upload (nice-to-have feature)
        st.subheader("📄 Upload Documents")
        uploaded_file = st.file_uploader(
            "Upload ML documents",
            type=["pdf", "txt", "md"],
            help="Upload PDF, TXT, or MD files to add to the knowledge base"
        )
        
        if uploaded_file:
            # Save uploaded file
            save_path = Path("data/user_uploads") / uploaded_file.name
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")
        
        st.markdown("---")
        
        # Actions
        st.subheader("Actions")
        if st.button("🗑️ Clear Conversation"):
            st.session_state.messages = []
            st.rerun()
        
        if st.button("📚 Reload Documents"):
            st.session_state.documents_loaded = False
            st.info("Documents will be reloaded on next query.")
        
        st.markdown("---")
        
        # Info
        st.subheader("ℹ️ About")
        st.markdown("""
        This is a multi-agent ML learning assistant powered by:
        - **CrewAI** for agent orchestration
        - **Llama 3.2** as the LLM
        - **LlamaIndex** for RAG
        - **ChromaDB** for vector storage
        """)
        
        return mode


def main():
    """Main application entry point."""
    init_session_state()
    
    # Render sidebar and get mode
    mode = sidebar()
    
    # Main content area
    st.title("🎓 Machine Learning Learning Assistant")
    st.markdown("Ask questions about Machine Learning concepts, request explanations, or take quizzes!")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    if mode == "💬 Chat & Learn":
        placeholder = "Ask me anything about Machine Learning..."
    else:
        placeholder = "Enter a topic for a quiz (e.g., 'Neural Networks', 'Gradient Descent')..."
    
    if user_input := st.chat_input(placeholder):
        process_user_input(user_input)


if __name__ == "__main__":
    main()
