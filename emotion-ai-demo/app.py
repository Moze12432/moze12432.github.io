"""
COMPLETE CLAUDE CLONE - Everything Claude Does
Full capabilities: Document editing, analysis, code execution, artifacts, web search, file uploads
Powered by Groq with intelligent fallback
"""

import streamlit as st
from groq import Groq
import requests
import re
import time
import uuid
import json
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from enum import Enum
import os
from io import BytesIO

# ============================================================================
# CONSTANTS
# ============================================================================

MODEL_PRIORITY = [
    "claude-3-5-sonnet-20241022",
    "claude-3-opus-20240229", 
    "claude-3-haiku-20240307",
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]

# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class ArtifactType(Enum):
    DOCUMENT = "document"
    CODE = "code"
    ANALYSIS = "analysis"
    SUMMARY = "summary"
    TRANSLATION = "translation"

@dataclass
class Message:
    role: str
    content: str
    timestamp: float = field(default_factory=time.time)

@dataclass
class Artifact:
    id: str
    title: str
    content: str
    type: ArtifactType
    created_at: float = field(default_factory=time.time)

@dataclass
class Conversation:
    id: str
    title: str
    messages: List[Message]
    artifacts: List[Artifact]
    created_at: float = field(default_factory=time.time)

# ============================================================================
# CLAUDE ROUTER
# ============================================================================

class ClaudeRouter:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.current_model = MODEL_PRIORITY[0]
        
    def chat(self, messages: List[Dict], stream: bool = False, on_token: Optional[Callable] = None,
             max_tokens: int = 4000, temperature: float = 0.7) -> str:
        for model in MODEL_PRIORITY:
            try:
                if stream:
                    return self._stream_chat(messages, model, on_token, max_tokens, temperature)
                else:
                    return self._complete_chat(messages, model, max_tokens, temperature)
            except Exception as e:
                continue
        return "I'm having trouble connecting. Please try again."
    
    def _complete_chat(self, messages, model, max_tokens, temperature):
        response = self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature
        )
        self.current_model = model
        return response.choices[0].message.content
    
    def _stream_chat(self, messages, model, on_token, max_tokens, temperature):
        full_response = ""
        stream = self.client.chat.completions.create(
            model=model, messages=messages, max_tokens=max_tokens, temperature=temperature, stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content:
                token = chunk.choices[0].delta.content
                full_response += token
                if on_token:
                    on_token(token)
        self.current_model = model
        return full_response

# ============================================================================
# DOCUMENT PROCESSOR
# ============================================================================

class DocumentProcessor:
    @staticmethod
    def extract_text_from_file(uploaded_file) -> str:
        try:
            content = uploaded_file.read()
            if uploaded_file.type == "text/plain":
                return content.decode('utf-8')
            elif uploaded_file.type == "application/pdf":
                try:
                    import PyPDF2
                    pdf_reader = PyPDF2.PdfReader(BytesIO(content))
                    return "\n".join([page.extract_text() for page in pdf_reader.pages])
                except:
                    return "PDF text extraction failed"
            else:
                return content.decode('utf-8', errors='ignore')
        except Exception as e:
            return f"Error: {str(e)}"
    
    @staticmethod
    def analyze_document(content: str, router: ClaudeRouter) -> str:
        prompt = f"""Analyze this document:

{content[:6000]}

Provide: 1) Summary 2) Key themes 3) Tone analysis 4) Suggestions for improvement"""
        return router.chat([{"role": "user", "content": prompt}], max_tokens=2000)
    
    @staticmethod
    def edit_document(content: str, instruction: str, router: ClaudeRouter, on_token=None) -> str:
        prompt = f"""Edit this document: {instruction}

Document: {content[:6000]}

Return ONLY the complete edited document."""
        return router.chat([{"role": "user", "content": prompt}], stream=on_token is not None, on_token=on_token)

# ============================================================================
# CODE INTERPRETER
# ============================================================================

class CodeInterpreter:
    @staticmethod
    def analyze_code(code: str, language: str, router: ClaudeRouter) -> str:
        prompt = f"Analyze this {language} code:\n```{language}\n{code}\n```\nProvide: what it does, bugs, improvements"
        return router.chat([{"role": "user", "content": prompt}])
    
    @staticmethod
    def generate_code(description: str, language: str, router: ClaudeRouter) -> str:
        prompt = f"Generate {language} code that: {description}\nReturn ONLY the code block."
        return router.chat([{"role": "user", "content": prompt}])

# ============================================================================
# WEB RESEARCHER
# ============================================================================

class WebResearcher:
    @staticmethod
    def search_web(query: str) -> str:
        try:
            url = "https://html.duckduckgo.com/html/"
            response = requests.post(url, data={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
            if response.status_code == 200:
                results = re.findall(r'<a rel="nofollow" class="result__a" href="[^"]*">([^<]+)</a>', response.text)
                snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', response.text)
                if results:
                    context = f"## Search Results for: {query}\n\n"
                    for i in range(min(3, len(results))):
                        context += f"### {results[i]}\n"
                        if i < len(snippets):
                            context += f"{re.sub(r'<[^>]+>', '', snippets[i])}\n\n"
                    return context
            return "No results found."
        except:
            return "Search failed."

# ============================================================================
# CLAUDE UI
# ============================================================================

class ClaudeUI:
    def __init__(self):
        self.init_session_state()
        self.doc_processor = DocumentProcessor()
        self.code_interpreter = CodeInterpreter()
        self.web_researcher = WebResearcher()
        
    def init_session_state(self):
        if "conversations" not in st.session_state:
            default_conv = Conversation(id=str(uuid.uuid4()), title="New Chat", messages=[], artifacts=[])
            st.session_state.conversations = [default_conv]
            st.session_state.current_conv_id = default_conv.id
        if "current_document" not in st.session_state:
            st.session_state.current_document = ""
        if "uploaded_files" not in st.session_state:
            st.session_state.uploaded_files = []
        if "current_artifact" not in st.session_state:
            st.session_state.current_artifact = None
        if "router" not in st.session_state:
            st.session_state.router = None
        if "is_processing" not in st.session_state:
            st.session_state.is_processing = False
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7
        if "sidebar_view" not in st.session_state:
            st.session_state.sidebar_view = "Chat"
    
    def get_current_conversation(self) -> Conversation:
        for conv in st.session_state.conversations:
            if conv.id == st.session_state.current_conv_id:
                return conv
        return st.session_state.conversations[0]
    
    def add_message(self, role: str, content: str):
        conv = self.get_current_conversation()
        conv.messages.append(Message(role=role, content=content))
        if len(conv.messages) == 1 and role == "user":
            conv.title = content[:30] + ("..." if len(content) > 30 else "")
    
    def add_artifact(self, title: str, content: str, artifact_type: ArtifactType = ArtifactType.DOCUMENT) -> Artifact:
        conv = self.get_current_conversation()
        artifact = Artifact(id=str(uuid.uuid4()), title=title, content=content, type=artifact_type)
        conv.artifacts.append(artifact)
        return artifact
    
    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### 🤖 Claude")
            st.markdown("---")
            
            new_chat_clicked = st.button("+ New chat", use_container_width=True, type="primary", key="new_chat_btn")
            if new_chat_clicked:
                new_conv = Conversation(id=str(uuid.uuid4()), title="New Chat", messages=[], artifacts=[])
                st.session_state.conversations.insert(0, new_conv)
                st.session_state.current_conv_id = new_conv.id
                st.session_state.current_document = ""
                st.session_state.current_artifact = None
                st.rerun()
            
            st.markdown("---")
            
            st.markdown("#### Recent chats")
            for i, conv in enumerate(st.session_state.conversations[:20]):
                title = conv.title[:25] + ("..." if len(conv.title) > 25 else "")
                btn_key = f"conv_btn_{conv.id}_{i}"
                if st.button(f"{title}", use_container_width=True, key=btn_key):
                    st.session_state.current_conv_id = conv.id
                    st.rerun()
            
            st.markdown("---")
            
            with st.expander("Settings"):
                st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, key="temp_slider")
            
            if st.session_state.router:
                st.caption(f"Model: {st.session_state.router.current_model}")
    
    def render_chat(self):
        conv = self.get_current_conversation()
        
        # Display messages
        for msg in conv.messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
        
        # Artifacts panel (Claude-like)
        if conv.artifacts:
            with st.expander(f"Artifacts ({len(conv.artifacts)})", expanded=False):
                for art in conv.artifacts[-5:]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{art.title}**")
                        st.caption(f"Type: {art.type.value}")
                    with col2:
                        if st.button("View", key=f"view_art_{art.id}"):
                            st.session_state.current_artifact = art
                            st.rerun()
        
        # File attachment (Claude-like)
        with st.expander("Attach file", expanded=False):
            uploaded = st.file_uploader("Upload a file", type=['pdf', 'docx', 'txt'], label_visibility="collapsed", key="chat_file_uploader")
            if uploaded:
                content = self.doc_processor.extract_text_from_file(uploaded)
                st.session_state.uploaded_files.append({"name": uploaded.name, "content": content})
                st.success(f"Attached: {uploaded.name}")
        
        # Chat input
        prompt = st.chat_input("Ask Claude...", disabled=st.session_state.is_processing)
        if prompt:
            self.add_message("user", prompt)
            st.rerun()
    
    def process_message(self, prompt: str):
        if not st.session_state.router:
            st.error("API key required")
            return
        
        st.session_state.is_processing = True
        conv = self.get_current_conversation()
        
        # Build context
        context = ""
        if st.session_state.current_document:
            context += f"\n\nUser is currently editing this document:\n{st.session_state.current_document}\n"
        if st.session_state.uploaded_files:
            for f in st.session_state.uploaded_files:
                context += f"\nAttached file: {f['name']}\nContent preview: {f['content'][:500]}\n"
        
        # Build messages
        messages = []
        for msg in conv.messages[-20:]:
            messages.append({"role": msg.role, "content": msg.content})
        
        full_prompt = prompt + context
        messages.append({"role": "user", "content": full_prompt})
        
        # Streaming response
        response_placeholder = st.empty()
        streaming_text = ""
        
        def on_token(token):
            nonlocal streaming_text
            streaming_text += token
            response_placeholder.markdown(streaming_text + " ▌")
        
        try:
            response = st.session_state.router.chat(
                messages=messages, stream=True, on_token=on_token,
                max_tokens=4000, temperature=st.session_state.temperature
            )
            response_placeholder.empty()
            
            # Display final response
            with st.chat_message("assistant"):
                st.markdown(streaming_text)
            
            # Create artifact for long responses
            if len(streaming_text) > 1000:
                artifact_title = f"Response: {prompt[:30]}..." if len(prompt) > 30 else prompt
                artifact = self.add_artifact(artifact_title, streaming_text, ArtifactType.DOCUMENT)
                st.info(f"✨ Saved as artifact: {artifact.title}")
            
            # Add to conversation
            self.add_message("assistant", streaming_text)
            
            # Auto-update document if edit instruction detected
            edit_keywords = ["edit", "rewrite", "improve", "fix", "change", "update", "modify", "revise"]
            if any(kw in prompt.lower() for kw in edit_keywords) and len(streaming_text) > 100:
                if streaming_text != st.session_state.current_document:
                    st.session_state.current_document = streaming_text
                    st.success("Document updated")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            st.session_state.is_processing = False
            st.rerun()
    
    def render_document_editor(self):
        st.markdown("### Document Editor")
        
        # Document title
        doc_title = st.text_input("Document Title", value="Untitled", key="doc_title", label_visibility="collapsed")
        
        # Document content
        content = st.text_area(
            "Document Content",
            value=st.session_state.current_document,
            height=400,
            key="doc_content",
            placeholder="Write your document here or paste content from your conversation..."
        )
        
        if content != st.session_state.current_document:
            st.session_state.current_document = content
        
        # Document stats
        if st.session_state.current_document:
            words = len(st.session_state.current_document.split())
            chars = len(st.session_state.current_document)
            st.caption(f"{words} words  ·  {chars} characters")
        
        st.markdown("---")
        st.markdown("#### Quick actions")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("✨ Improve writing", use_container_width=True, key="improve_btn"):
                if st.session_state.current_document and st.session_state.router:
                    with st.spinner("Improving..."):
                        improved = self.doc_processor.edit_document(
                            st.session_state.current_document,
                            "Improve the writing quality, fix grammar, and enhance clarity",
                            st.session_state.router
                        )
                        st.session_state.current_document = improved
                        st.rerun()
        
        with col2:
            if st.button("📝 Summarize", use_container_width=True, key="summarize_btn"):
                if st.session_state.current_document and st.session_state.router:
                    with st.spinner("Summarizing..."):
                        summary = self.doc_processor.edit_document(
                            st.session_state.current_document,
                            "Provide a concise summary of this document",
                            st.session_state.router
                        )
                        self.add_message("assistant", f"**Summary:**\n\n{summary}")
                        st.rerun()
        
        with col3:
            if st.button("🔍 Analyze", use_container_width=True, key="analyze_btn"):
                if st.session_state.current_document and st.session_state.router:
                    with st.spinner("Analyzing..."):
                        analysis = self.doc_processor.analyze_document(
                            st.session_state.current_document,
                            st.session_state.router
                        )
                        self.add_message("assistant", f"**Analysis:**\n\n{analysis}")
                        st.rerun()
    
    def render_code_interpreter(self):
        st.markdown("### Code Interpreter")
        
        language = st.selectbox(
            "Language",
            ["python", "javascript", "typescript", "java", "cpp", "go", "rust", "html", "css", "sql", "bash"],
            key="code_lang"
        )
        
        code = st.text_area(
            "Code",
            height=300,
            placeholder=f"Paste your {language} code here...",
            key="code_input"
        )
        
        if code:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔍 Analyze Code", use_container_width=True, key="code_analyze_btn"):
                    if st.session_state.router:
                        with st.spinner("Analyzing..."):
                            analysis = self.code_interpreter.analyze_code(code, language, st.session_state.router)
                            self.add_message("assistant", f"**Code Analysis:**\n\n{analysis}")
                            st.rerun()
            
            with col2:
                if st.button("📖 Explain Code", use_container_width=True, key="code_explain_btn"):
                    if st.session_state.router:
                        with st.spinner("Explaining..."):
                            explanation = st.session_state.router.chat([
                                {"role": "user", "content": f"Explain this {language} code line by line:\n```{language}\n{code}\n```"}
                            ])
                            self.add_message("assistant", f"**Code Explanation:**\n\n{explanation}")
                            st.rerun()
        
        st.markdown("---")
        st.markdown("#### Generate Code")
        
        code_desc = st.text_area(
            "Describe what you want",
            height=100,
            placeholder="Example: A Python function that fetches weather data from an API",
            key="code_desc"
        )
        
        if st.button("✨ Generate Code", type="primary", use_container_width=True, key="code_gen_btn"):
            if code_desc and st.session_state.router:
                with st.spinner("Generating..."):
                    generated = self.code_interpreter.generate_code(code_desc, language, st.session_state.router)
                    artifact = self.add_artifact(f"Code: {code_desc[:30]}...", generated, ArtifactType.CODE)
                    st.code(generated, language=language)
                    st.success(f"Saved as artifact: {artifact.title}")
    
    def render_web_research(self):
        st.markdown("### Web Research")
        
        query = st.text_input(
            "Search the web",
            placeholder="What would you like to research?",
            key="research_query"
        )
        
        if query:
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Quick Search", use_container_width=True, key="quick_search_btn"):
                    with st.spinner("Searching..."):
                        results = self.web_researcher.search_web(query)
                        self.add_message("assistant", f"**Search Results:**\n\n{results}")
                        st.rerun()
            
            with col2:
                if st.button("Deep Research", type="primary", use_container_width=True, key="deep_research_btn"):
                    if st.session_state.router:
                        with st.spinner("Researching..."):
                            search_results = self.web_researcher.search_web(query)
                            analysis = st.session_state.router.chat([
                                {"role": "user", "content": f"Based on these search results about '{query}', provide a comprehensive research summary:\n\n{search_results}"}
                            ], max_tokens=3000)
                            artifact = self.add_artifact(f"Research: {query[:30]}...", analysis, ArtifactType.ANALYSIS)
                            self.add_message("assistant", f"**Research Report:**\n\n{analysis}")
                            st.success(f"Saved as artifact: {artifact.title}")
                            st.rerun()
    
    def render_artifact_viewer(self):
        if st.session_state.current_artifact:
            art = st.session_state.current_artifact
            with st.expander(f"📎 {art.title}", expanded=True):
                if art.type == ArtifactType.CODE:
                    st.code(art.content)
                else:
                    st.markdown(art.content)
                if st.button("Close", key="close_artifact_btn"):
                    st.session_state.current_artifact = None
                    st.rerun()
    
    def render_artifact_panel(self):
        """Render artifact panel in sidebar like Claude"""
        conv = self.get_current_conversation()
        if conv.artifacts:
            with st.sidebar:
                st.markdown("---")
                st.markdown("### 📎 Artifacts")
                for art in conv.artifacts[-5:]:
                    if st.button(f"{art.title}", key=f"sidebar_art_{art.id}"):
                        st.session_state.current_artifact = art
                        st.rerun()
    
    def render(self):
        # Main layout: Sidebar + Main content
        self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs(["💬 Chat", "📄 Documents", "💻 Code", "🔍 Research"])
        
        with tab1:
            self.render_chat()
        
        with tab2:
            self.render_document_editor()
        
        with tab3:
            self.render_code_interpreter()
        
        with tab4:
            self.render_web_research()
        
        self.render_artifact_panel()
        self.render_artifact_viewer()

# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="Claude - AI Assistant",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Apply Claude-like CSS
    st.markdown("""
    <style>
    /* Main container */
    .main {
        background-color: #f9f9f9;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: transparent;
        border-radius: 12px;
        padding: 8px 0;
    }
    
    /* User message */
    [data-testid="stChatMessage"]:has([data-testid="stMarkdown"]:contains("user")) {
        background-color: #f0f0f0;
    }
    
    /* Assistant message */
    [data-testid="stChatMessage"]:has([data-testid="stMarkdown"]:contains("assistant")) {
        background-color: transparent;
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        transition: all 0.2s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-1px);
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Primary button */
    .stButton button[kind="primary"] {
        background-color: #10a37f;
        color: white;
    }
    
    /* Text area */
    .stTextArea textarea {
        border-radius: 8px;
        font-family: 'Courier New', monospace;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #f7f7f8;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Initialize UI
    ui = ClaudeUI()
    
    # Get API key
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except:
        pass
    
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    
    if api_key and st.session_state.router is None:
        st.session_state.router = ClaudeRouter(api_key)
    
    if not api_key:
        with st.sidebar:
            st.warning("GROQ_API_KEY Required")
            key_input = st.text_input("Enter API Key:", type="password", key="api_key_input")
            if key_input:
                st.session_state.router = ClaudeRouter(key_input)
                st.rerun()
        return
    
    # Process pending messages
    conv = ui.get_current_conversation()
    if conv.messages and conv.messages[-1].role == "user":
        if len(conv.messages) == 1 or conv.messages[-2].role != "assistant":
            ui.process_message(conv.messages[-1].content)
    
    # Render UI
    ui.render()

if __name__ == "__main__":
    main()
