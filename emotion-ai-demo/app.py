"""
GROQ AI DOCUMENT STUDIO - Complete Fixed Implementation
Valid Groq models, proper streaming, no duplicate messages, error-resilient
"""

import streamlit as st
from groq import Groq
import re
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Generator
from dataclasses import dataclass, field
from enum import Enum
import os
from io import BytesIO
import difflib

# ============================================================================
# CONSTANTS - Valid Groq Models Only
# ============================================================================

VALID_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

MODEL_PRIORITY = VALID_GROQ_MODELS

MAX_HISTORY = 20
MAX_DOC_TOKENS = 4000
MAX_DOC_CHARS = MAX_DOC_TOKENS * 3  # ~12000 chars

# ============================================================================
# SYSTEM PROMPT - Crucial for Groq models to behave like a proper assistant
# ============================================================================

SYSTEM_PROMPT = """You are GroqAI, a helpful AI writing assistant integrated into a document editor.

CRITICAL RULES FOR DOCUMENT EDITING:
1. When asked to EDIT a document, wrap the COMPLETE edited document in <document>...</document> tags.
2. Do NOT add any preamble like "Sure!" or "Here's your edited document:" before the <document> tag.
3. If you're just answering a question (not editing), respond normally without <document> tags.
4. Be concise and helpful in regular responses.
5. Preserve the original meaning unless instructed otherwise.

EXAMPLE EDIT RESPONSE:
<document>The completely edited document content goes here...</document>

EXAMPLE QUESTION RESPONSE:
Yes, you can improve passive voice by using active voice. For example, change "The ball was hit by John" to "John hit the ball"."""

# ============================================================================
# ENUMS & DATA CLASSES
# ============================================================================

class ArtifactType(Enum):
    DOCUMENT = "document"
    CODE = "code"
    ANALYSIS = "analysis"
    SUMMARY = "summary"

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
    current_document: str = ""
    document_title: str = "Untitled Document"
    document_versions: list = field(default_factory=list)
    seen_files: set = field(default_factory=set)
    active_diff: str = ""
    created_at: float = field(default_factory=time.time)

# ============================================================================
# GROQ ROUTER - Proper Groq API implementation
# ============================================================================

class GroqRouter:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.current_model = MODEL_PRIORITY[0]
        self.last_error = None
        
    def stream_chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 4000) -> Generator[str, None, None]:
        """Stream chat responses - proper generator for st.write_stream"""
        last_error = None
        for model in MODEL_PRIORITY:
            try:
                stream = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=True
                )
                self.current_model = model
                self.last_error = None
                for chunk in stream:
                    if chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
                return
            except Exception as e:
                last_error = f"{model}: {str(e)}"
                self.last_error = last_error
                continue
        raise RuntimeError(f"All models failed. Last error: {last_error}")
    
    def complete_chat(self, messages: List[Dict], temperature: float = 0.7, max_tokens: int = 4000) -> str:
        """Non-streaming chat completion with streaming fallback"""
        last_error = None
        for model in MODEL_PRIORITY:
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stream=False
                )
                self.current_model = model
                self.last_error = None
                return response.choices[0].message.content
            except Exception as e:
                last_error = f"{model}: {str(e)}"
                self.last_error = last_error
                continue
        raise RuntimeError(f"All models failed. Last error: {last_error}")

# ============================================================================
# DOCUMENT UTILITIES
# ============================================================================

def extract_document_from_response(response: str) -> tuple[str, bool]:
    """Extract document from <document> tags if present"""
    doc_match = re.search(r'<document>(.*?)</document>', response, re.DOTALL)
    if doc_match:
        return doc_match.group(1).strip(), True
    return response, False

def truncate_for_context(text: str, max_chars: int = MAX_DOC_CHARS) -> str:
    """Token-aware truncation with context preservation"""
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... content truncated due to length ...]\n\n" + text[-half:]

def show_diff(old_content: str, new_content: str) -> str:
    """Generate unified diff for track changes"""
    old_lines = old_content.splitlines()
    new_lines = new_content.splitlines()
    diff = difflib.unified_diff(old_lines, new_lines, lineterm='', n=2)
    return "\n".join(diff)

def extract_file_text(uploaded_file) -> str:
    """Properly extract text from uploaded files"""
    try:
        if uploaded_file.type == "application/pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(BytesIO(uploaded_file.getvalue()))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return "PDF support requires pypdf. Install with: pip install pypdf"
        
        elif "wordprocessingml" in uploaded_file.type or uploaded_file.name.endswith('.docx'):
            try:
                import docx
                doc = docx.Document(BytesIO(uploaded_file.getvalue()))
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                return "DOCX support requires python-docx. Install with: pip install python-docx"
        
        elif uploaded_file.type == "text/plain":
            return uploaded_file.getvalue().decode('utf-8')
        
        else:
            return uploaded_file.getvalue().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Error extracting text: {str(e)}"

# ============================================================================
# GROQ UI
# ============================================================================

class GroqUI:
    def __init__(self):
        self.init_session_state()
        
    def init_session_state(self):
        if "conversations" not in st.session_state:
            default_conv = Conversation(id=str(uuid.uuid4()), title="New Chat", messages=[], artifacts=[])
            st.session_state.conversations = [default_conv]
            st.session_state.current_conv_id = default_conv.id
        if "router" not in st.session_state:
            st.session_state.router = None
        if "track_changes" not in st.session_state:
            st.session_state.track_changes = False
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7
    
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
    
    def create_artifact(self, title: str, content: str, artifact_type: ArtifactType = ArtifactType.DOCUMENT) -> Artifact:
        conv = self.get_current_conversation()
        artifact = Artifact(id=str(uuid.uuid4()), title=title, content=content, type=artifact_type)
        conv.artifacts.append(artifact)
        return artifact
    
    def save_version(self, content: str, description: str = ""):
        conv = self.get_current_conversation()
        version = {
            "id": str(uuid.uuid4()),
            "content": content,
            "timestamp": time.time(),
            "description": description
        }
        conv.document_versions.append(version)
        if len(conv.document_versions) > 50:
            conv.document_versions = conv.document_versions[-50:]
    
    def process_edit(self, instruction: str):
        conv = self.get_current_conversation()
        
        if not st.session_state.router:
            st.error("No API key configured. Please add GROQ_API_KEY.")
            return
        
        if not conv.current_document:
            st.warning("No document to edit. Please create or upload a document first.")
            return
        
        try:
            with st.spinner("Editing..."):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Edit this document: {instruction}\n\nDocument:\n{truncate_for_context(conv.current_document)}"}
                ]
                
                response = st.session_state.router.complete_chat(
                    messages=messages,
                    temperature=st.session_state.temperature
                )
                
                doc_content, is_document = extract_document_from_response(response)
                if is_document and doc_content != conv.current_document and len(doc_content) > 100:
                    if st.session_state.track_changes:
                        self.save_version(conv.current_document, f"AI: {instruction[:30]}")
                    conv.current_document = doc_content
                    st.toast("Document updated!", icon="✅")
                    st.rerun()
                else:
                    st.info("No changes were made to the document.")
        except Exception as e:
            st.error(f"Edit failed: {str(e)}")
    
    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### 🤖 GroqAI")
            st.markdown("---")
            
            if st.button("+ New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
                new_conv = Conversation(id=str(uuid.uuid4()), title="New Chat", messages=[], artifacts=[])
                st.session_state.conversations.insert(0, new_conv)
                st.session_state.current_conv_id = new_conv.id
                st.rerun()
            
            st.markdown("---")
            st.markdown("#### Recent Chats")
            for i, conv in enumerate(st.session_state.conversations[:20]):
                title = conv.title[:25] + ("..." if len(conv.title) > 25 else "")
                if st.button(f"💬 {title}", use_container_width=True, key=f"conv_{conv.id}_{i}"):
                    st.session_state.current_conv_id = conv.id
                    st.rerun()
            
            st.markdown("---")
            with st.expander("Settings"):
                st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, key="temp_slider")
                st.session_state.track_changes = st.checkbox("Track Changes", st.session_state.track_changes, key="track_checkbox")
            
            if st.session_state.router:
                st.caption(f"Model: {st.session_state.router.current_model}")
                if st.session_state.router.last_error:
                    st.caption(f"⚠️ Last error: {st.session_state.router.last_error[:50]}")
    
    def render_artifacts_panel(self):
        conv = self.get_current_conversation()
        
        if conv.artifacts:
            with st.expander(f"📎 Artifacts ({len(conv.artifacts)})", expanded=False):
                for art in reversed(conv.artifacts[-10:]):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.caption(f"📄 {art.title}")
                    with col2:
                        if st.button("Load", key=f"load_art_{art.id}"):
                            conv.current_document = art.content
                            st.toast(f"Loaded artifact: {art.title}", icon="✅")
                            st.rerun()
                    with col3:
                        st.download_button(
                            label="📥",
                            data=art.content,
                            file_name=f"{art.title}.txt",
                            mime="text/plain",
                            key=f"download_art_{art.id}",
                            use_container_width=True
                        )
    
    def render_document_editor(self):
        conv = self.get_current_conversation()
        
        st.markdown("### 📄 Document Editor")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            if st.button("📁 New", use_container_width=True, key="new_doc_btn"):
                conv.current_document = ""
                conv.document_title = "Untitled Document"
                st.rerun()
        with col2:
            if st.button("💾 Save Version", use_container_width=True, key="save_version_btn"):
                self.save_version(conv.current_document, "Manual save")
                st.toast("Version saved!", icon="✅")
        with col3:
            # Fixed: Direct download button - no wrapper button
            st.download_button(
                label="📥 Download",
                data=conv.current_document if conv.current_document else " ",
                file_name=f"{conv.document_title}.txt",
                mime="text/plain",
                key="download_file_btn",
                use_container_width=True,
                disabled=not conv.current_document
            )
        with col4:
            if st.button("📊 Stats", use_container_width=True, key="stats_doc_btn"):
                words = len(conv.current_document.split())
                chars = len(conv.current_document)
                st.toast(f"{words} words, {chars} characters", icon="📊")
        with col5:
            if st.button("🗑️ Clear", use_container_width=True, key="clear_doc_btn"):
                conv.current_document = ""
                st.rerun()
        
        st.markdown("---")
        
        title = st.text_input("Title", value=conv.document_title, key="doc_title_input")
        if title != conv.document_title:
            conv.document_title = title
        
        content = st.text_area(
            "Content",
            value=conv.current_document,
            height=400,
            key="doc_content_area",
            placeholder="Write or paste your document here...",
            label_visibility="collapsed"
        )
        
        if content != conv.current_document:
            conv.current_document = content
        
        words = len(conv.current_document.split())
        chars = len(conv.current_document)
        st.caption(f"📝 {words} words  ·  🔤 {chars} characters")
        
        self.render_artifacts_panel()
        
        if conv.active_diff:
            with st.expander("📊 Document Diff", expanded=True):
                st.code(conv.active_diff, language="diff")
                if st.button("Close Diff", key="close_diff_btn"):
                    conv.active_diff = ""
                    st.rerun()
        
        if conv.document_versions:
            with st.expander("📜 Version History", expanded=False):
                for version in reversed(conv.document_versions[-10:]):
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        time_str = datetime.fromtimestamp(version["timestamp"]).strftime("%H:%M:%S")
                        st.caption(f"{time_str} - {version['description']}")
                    with col2:
                        if st.button("Diff", key=f"diff_{version['id']}"):
                            conv.active_diff = show_diff(version["content"], conv.current_document)
                            st.rerun()
                    with col3:
                        if st.button("Restore", key=f"restore_{version['id']}"):
                            conv.current_document = version["content"]
                            st.rerun()
    
    def render_ai_panel(self):
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🎯 AI Actions")
            
            quick_edits = [
                ("✨ Improve Writing", "Improve grammar, clarity, and flow"),
                ("🎨 Make Professional", "Make the tone professional and polished"),
                ("📝 Add Structure", "Add headings and organize content"),
                ("🔤 Fix Grammar Only", "Fix only grammar and spelling"),
                ("📊 More Concise", "Make more concise while preserving meaning"),
                ("🎓 Academic Tone", "Rewrite in academic tone"),
            ]
            
            for label, instruction in quick_edits:
                if st.button(label, use_container_width=True, key=f"quick_{label[:10]}"):
                    self.process_edit(instruction)
            
            st.markdown("---")
            st.markdown("#### Custom Edit")
            
            custom_instruction = st.text_area(
                "Instructions",
                placeholder="Example: Add a conclusion about the key findings...",
                height=80,
                key="custom_edit_input"
            )
            
            if st.button("Apply Edit", type="primary", use_container_width=True, key="apply_edit_btn"):
                if custom_instruction:
                    self.process_edit(custom_instruction)
    
    def render_chat(self):
        conv = self.get_current_conversation()
        
        for msg in conv.messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
        
        with st.expander("Attach file", expanded=False):
            uploaded = st.file_uploader("Upload", type=['pdf', 'docx', 'txt'], label_visibility="collapsed", key="chat_uploader")
            if uploaded and uploaded.name not in conv.seen_files:
                conv.seen_files.add(uploaded.name)
                text = extract_file_text(uploaded)
                conv.current_document = text
                self.add_message("user", f"[Attached file: {uploaded.name}]\n\nDocument content loaded.")
                st.success(f"✅ Loaded: {uploaded.name}")
                st.rerun()
        
        prompt = st.chat_input("Ask GroqAI to edit your document...")
        if prompt:
            if not st.session_state.router:
                st.error("No API key configured. Please add GROQ_API_KEY.")
                return
            
            self.add_message("user", prompt)
            
            with st.chat_message("assistant"):
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                
                for msg in conv.messages[:-1][-19:]:
                    messages.append({"role": msg.role, "content": msg.content})
                
                full_prompt = prompt
                if conv.current_document:
                    full_prompt = f"{prompt}\n\nUser is currently editing this document:\n{truncate_for_context(conv.current_document)}"
                messages.append({"role": "user", "content": full_prompt})
                
                try:
                    response = st.write_stream(
                        st.session_state.router.stream_chat(
                            messages=messages,
                            temperature=st.session_state.temperature
                        )
                    )
                except RuntimeError as e:
                    st.error(str(e))
                    self.add_message("assistant", f"Error: {str(e)}")
                    return
            
            doc_content, is_document = extract_document_from_response(response)
            
            document_updated = False
            if is_document and doc_content != conv.current_document and len(doc_content) > 100:
                if st.session_state.track_changes:
                    self.save_version(conv.current_document, f"AI Edit: {prompt[:50]}")
                conv.current_document = doc_content
                document_updated = True
            
            if is_document and len(doc_content) > 500:
                artifact_title = f"Edit: {prompt[:30]}..." if len(prompt) > 30 else prompt
                self.create_artifact(artifact_title, doc_content, ArtifactType.DOCUMENT)
            
            self.add_message("assistant", response)
            
            if document_updated:
                st.rerun()
    
    def render(self):
        self.render_sidebar()
        self.render_ai_panel()
        
        col1, col2 = st.columns([3, 2])
        
        with col1:
            self.render_document_editor()
        
        with col2:
            st.markdown("### 💬 Chat with GroqAI")
            self.render_chat()

# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="GroqAI Document Studio",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.markdown("""
    <style>
    .stTextArea textarea {
        font-family: 'Courier New', monospace;
        font-size: 14px;
        line-height: 1.6;
    }
    .stButton button {
        border-radius: 6px;
        transition: all 0.2s;
    }
    </style>
    """, unsafe_allow_html=True)
    
    ui = GroqUI()
    
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except:
        pass
    
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")
    
    if api_key and st.session_state.router is None:
        st.session_state.router = GroqRouter(api_key)
    
    if not api_key:
        with st.sidebar:
            st.error("🔑 GROQ_API_KEY Required")
            st.markdown("Get your key from [Groq Console](https://console.groq.com)")
            key_input = st.text_input("Enter API Key:", type="password", key="api_input")
            if key_input:
                st.session_state.router = GroqRouter(key_input)
                st.rerun()
        return
    
    ui.render()

if __name__ == "__main__":
    main()
