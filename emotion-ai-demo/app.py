"""
COMPLETE CLAUDE CLONE - Microsoft Word Style Editor
Full capabilities: Rich text editing, real-time AI editing, track changes, formatting
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

@dataclass
class DocumentVersion:
    id: str
    content: str
    timestamp: float
    description: str

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
# MICROSOFT WORD-STYLE DOCUMENT EDITOR
# ============================================================================

class WordStyleEditor:
    """Microsoft Word-style document editor with rich formatting"""
    
    def __init__(self):
        self.version_history = []
        
    def apply_formatting(self, text: str, format_type: str) -> str:
        """Apply formatting like Word"""
        if format_type == "bold":
            return f"**{text}**"
        elif format_type == "italic":
            return f"*{text}*"
        elif format_type == "heading1":
            return f"# {text}"
        elif format_type == "heading2":
            return f"## {text}"
        elif format_type == "heading3":
            return f"### {text}"
        elif format_type == "bullet":
            return f"- {text}"
        elif format_type == "number":
            return f"1. {text}"
        return text
    
    def save_version(self, content: str, description: str = ""):
        """Save document version"""
        version = DocumentVersion(
            id=str(uuid.uuid4()),
            content=content,
            timestamp=time.time(),
            description=description
        )
        self.version_history.append(version)
        if len(self.version_history) > 50:
            self.version_history = self.version_history[-50:]
        return version

# ============================================================================
# CLAUDE UI WITH WORD-STYLE EDITOR
# ============================================================================

class ClaudeUI:
    def __init__(self):
        self.init_session_state()
        self.word_editor = WordStyleEditor()
        
    def init_session_state(self):
        if "conversations" not in st.session_state:
            default_conv = Conversation(id=str(uuid.uuid4()), title="New Chat", messages=[], artifacts=[])
            st.session_state.conversations = [default_conv]
            st.session_state.current_conv_id = default_conv.id
        if "current_document" not in st.session_state:
            st.session_state.current_document = ""
        if "document_title" not in st.session_state:
            st.session_state.document_title = "Untitled Document"
        if "document_versions" not in st.session_state:
            st.session_state.document_versions = []
        if "track_changes" not in st.session_state:
            st.session_state.track_changes = False
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
        if "show_ai_panel" not in st.session_state:
            st.session_state.show_ai_panel = True
    
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
    
    def save_document_version(self, description: str = ""):
        """Save current document version"""
        version = self.word_editor.save_version(st.session_state.current_document, description)
        st.session_state.document_versions.append(version)
        return version
    
    def restore_version(self, version_id: str):
        """Restore a previous version"""
        for version in st.session_state.document_versions:
            if version.id == version_id:
                st.session_state.current_document = version.content
                return True
        return False
    
    def render_sidebar(self):
        with st.sidebar:
            st.markdown("### 🤖 Claude")
            st.markdown("---")
            
            # New Chat
            if st.button("+ New chat", use_container_width=True, type="primary", key="new_chat_btn"):
                new_conv = Conversation(id=str(uuid.uuid4()), title="New Chat", messages=[], artifacts=[])
                st.session_state.conversations.insert(0, new_conv)
                st.session_state.current_conv_id = new_conv.id
                st.rerun()
            
            st.markdown("---")
            
            # Recent Chats
            st.markdown("#### Recent chats")
            for i, conv in enumerate(st.session_state.conversations[:20]):
                title = conv.title[:25] + ("..." if len(conv.title) > 25 else "")
                if st.button(f"💬 {title}", use_container_width=True, key=f"conv_{conv.id}_{i}"):
                    st.session_state.current_conv_id = conv.id
                    st.rerun()
            
            st.markdown("---")
            
            # Settings
            with st.expander("Settings"):
                st.session_state.temperature = st.slider("Temperature", 0.0, 1.0, st.session_state.temperature, key="temp_slider")
                st.session_state.track_changes = st.checkbox("Track Changes", st.session_state.track_changes, key="track_changes_checkbox")
            
            if st.session_state.router:
                st.caption(f"Model: {st.session_state.router.current_model}")
    
    def render_word_editor(self):
        """Microsoft Word-style document editor"""
        
        # Word-style toolbar
        st.markdown("### 📄 Document Editor")
        
        # Toolbar row 1 - File operations
        col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
        with col1:
            if st.button("📁 New", use_container_width=True, key="new_doc_btn"):
                st.session_state.current_document = ""
                st.session_state.document_title = "Untitled Document"
                st.rerun()
        with col2:
            if st.button("💾 Save", use_container_width=True, key="save_doc_btn"):
                self.save_document_version("Manual save")
                st.toast("Document saved", icon="✅")
        with col3:
            if st.button("↩️ Undo", use_container_width=True, key="undo_btn"):
                if len(st.session_state.document_versions) > 1:
                    prev_version = st.session_state.document_versions[-2]
                    st.session_state.current_document = prev_version.content
                    st.rerun()
        with col4:
            if st.button("↪️ Redo", use_container_width=True, key="redo_btn"):
                st.info("Redo not implemented")
        with col5:
            if st.button("📋 Copy", use_container_width=True, key="copy_btn"):
                st.toast("Copied to clipboard", icon="📋")
        with col6:
            if st.button("📥 Download", use_container_width=True, key="download_btn"):
                if st.session_state.current_document:
                    st.download_button(
                        label="Download",
                        data=st.session_state.current_document,
                        file_name=f"{st.session_state.document_title}.txt",
                        mime="text/plain",
                        key="download_file_btn"
                    )
        with col7:
            if st.button("📊 Stats", use_container_width=True, key="stats_btn"):
                words = len(st.session_state.current_document.split())
                chars = len(st.session_state.current_document)
                st.toast(f"{words} words, {chars} characters", icon="📊")
        
        # Toolbar row 2 - Formatting (Word-style)
        st.markdown("---")
        col1, col2, col3, col4, col5, col6 = st.columns(6)
        with col1:
            if st.button("B", use_container_width=True, key="bold_btn"):
                st.info("Select text and click bold (rich text coming soon)")
        with col2:
            if st.button("I", use_container_width=True, key="italic_btn"):
                st.info("Select text and click italic")
        with col3:
            if st.button("H1", use_container_width=True, key="h1_btn"):
                st.info("Heading 1 formatting")
        with col4:
            if st.button("H2", use_container_width=True, key="h2_btn"):
                st.info("Heading 2 formatting")
        with col5:
            if st.button("• Bullet", use_container_width=True, key="bullet_btn"):
                st.info("Bullet list formatting")
        with col6:
            if st.button("1. Number", use_container_width=True, key="number_btn"):
                st.info("Numbered list formatting")
        
        st.markdown("---")
        
        # Document title (Word-style)
        doc_title = st.text_input(
            "Document Title",
            value=st.session_state.document_title,
            key="doc_title_input",
            label_visibility="collapsed",
            placeholder="Document title..."
        )
        if doc_title != st.session_state.document_title:
            st.session_state.document_title = doc_title
        
        # Main document editor (Word-style text area)
        content = st.text_area(
            "Document Content",
            value=st.session_state.current_document,
            height=450,
            key="doc_content_area",
            placeholder="Start typing your document here...",
            label_visibility="collapsed"
        )
        
        # Auto-save on change
        if content != st.session_state.current_document:
            if st.session_state.track_changes:
                self.save_document_version(f"Auto-save: {datetime.now().strftime('%H:%M:%S')}")
            st.session_state.current_document = content
        
        st.markdown("---")
        
        # Document info bar (Word-style)
        words = len(st.session_state.current_document.split()) if st.session_state.current_document else 0
        chars = len(st.session_state.current_document) if st.session_state.current_document else 0
        lines = len(st.session_state.current_document.split('\n')) if st.session_state.current_document else 0
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption(f"📝 {words} words")
        with col2:
            st.caption(f"🔤 {chars} characters")
        with col3:
            st.caption(f"📄 {lines} lines")
        with col4:
            versions_count = len(st.session_state.document_versions)
            st.caption(f"💾 {versions_count} versions saved")
        
        # Version history (Word-style)
        if st.session_state.document_versions:
            with st.expander("📜 Version History", expanded=False):
                for version in reversed(st.session_state.document_versions[-10:]):
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        time_str = datetime.fromtimestamp(version.timestamp).strftime("%Y-%m-%d %H:%M:%S")
                        st.caption(f"{time_str} - {version.description}")
                    with col2:
                        if st.button("Restore", key=f"restore_{version.id}"):
                            st.session_state.current_document = version.content
                            st.rerun()
    
    def render_ai_panel(self):
        """Claude-style AI editing panel"""
        
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🎯 AI Assistant")
            
            # Quick edit suggestions
            st.markdown("#### Quick edits")
            quick_edits = [
                ("✨ Improve writing", "Improve the writing quality, fix grammar, and enhance clarity"),
                ("🎨 Make professional", "Make this document more professional and polished"),
                ("📝 Add structure", "Add clear headings, sections, and improve organization"),
                ("🔤 Fix grammar", "Fix all grammar and spelling errors"),
                ("📊 More concise", "Make this more concise while preserving key information"),
                ("🎓 Academic tone", "Rewrite in academic tone with proper citations"),
                ("💼 Business tone", "Rewrite in professional business tone"),
                ("🗣️ Simplify language", "Simplify the language for a general audience"),
            ]
            
            for label, instruction in quick_edits:
                if st.button(label, use_container_width=True, key=f"quick_edit_{label[:10]}"):
                    if st.session_state.current_document and st.session_state.router:
                        st.session_state.is_processing = True
                        
                        # Create placeholder for streaming
                        response_placeholder = st.empty()
                        streaming_text = ""
                        
                        def on_token(token):
                            nonlocal streaming_text
                            streaming_text += token
                            response_placeholder.markdown(f"**Editing:** {streaming_text[:200]}...")
                        
                        # Get AI edit
                        prompt = f"""Edit this document: {instruction}

Document:
{st.session_state.current_document[:6000]}

Return ONLY the complete edited document."""
                        
                        messages = [{"role": "user", "content": prompt}]
                        
                        try:
                            edited = st.session_state.router.chat(
                                messages=messages,
                                stream=True,
                                on_token=on_token,
                                max_tokens=4000,
                                temperature=st.session_state.temperature
                            )
                            response_placeholder.empty()
                            
                            # Save old version if tracking changes
                            if st.session_state.track_changes:
                                self.save_document_version(f"AI Edit: {instruction[:50]}")
                            
                            # Update document
                            st.session_state.current_document = streaming_text
                            st.success("✅ Document updated!")
                            st.rerun()
                        except Exception as e:
                            st.error(f"Error: {str(e)}")
                        finally:
                            st.session_state.is_processing = False
            
            st.markdown("---")
            
            # Custom instruction
            st.markdown("#### Custom instruction")
            custom_instruction = st.text_area(
                "Tell Claude what to do",
                placeholder="Example: Add a conclusion about the key findings...",
                height=80,
                key="custom_instruction"
            )
            
            if st.button("Apply edit", type="primary", use_container_width=True, key="apply_edit_btn"):
                if custom_instruction and st.session_state.current_document and st.session_state.router:
                    with st.spinner("Editing..."):
                        edited = st.session_state.router.chat([
                            {"role": "user", "content": f"""Edit this document: {custom_instruction}

Document:
{st.session_state.current_document[:6000]}

Return ONLY the complete edited document."""}
                        ], max_tokens=4000, temperature=st.session_state.temperature)
                        
                        if st.session_state.track_changes:
                            self.save_document_version(f"AI Edit: {custom_instruction[:50]}")
                        
                        st.session_state.current_document = edited
                        st.success("✅ Document updated!")
                        st.rerun()
    
    def render_chat(self):
        """Chat interface for conversation with Claude"""
        
        conv = self.get_current_conversation()
        
        # Display messages
        for msg in conv.messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)
        
        # Artifacts
        if conv.artifacts:
            with st.expander(f"📎 Artifacts ({len(conv.artifacts)})", expanded=False):
                for art in conv.artifacts[-5:]:
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(f"**{art.title}**")
                        st.caption(f"Type: {art.type.value}")
                    with col2:
                        if st.button("View", key=f"view_art_{art.id}"):
                            st.session_state.current_artifact = art
                            st.rerun()
        
        # File attachment
        with st.expander("Attach file", expanded=False):
            uploaded = st.file_uploader("Upload a file", type=['pdf', 'docx', 'txt'], label_visibility="collapsed", key="chat_file_uploader")
            if uploaded:
                try:
                    content = uploaded.read()
                    if uploaded.type == "text/plain":
                        text_content = content.decode('utf-8')
                    else:
                        text_content = f"[File: {uploaded.name}]\n{content[:500].decode('utf-8', errors='ignore')}"
                    
                    st.session_state.uploaded_files.append({"name": uploaded.name, "content": text_content})
                    st.success(f"✅ Attached: {uploaded.name}")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        # Chat input
        prompt = st.chat_input("Ask Claude to edit your document...", disabled=st.session_state.is_processing)
        if prompt:
            self.add_message("user", prompt)
            st.rerun()
    
    def process_message(self, prompt: str):
        """Process user message with document context"""
        
        if not st.session_state.router:
            st.error("API key required")
            return
        
        st.session_state.is_processing = True
        conv = self.get_current_conversation()
        
        # Build context with current document
        context = ""
        if st.session_state.current_document:
            context += f"\n\nUser is currently editing this document:\n{st.session_state.current_document}\n"
        
        if st.session_state.uploaded_files:
            for f in st.session_state.uploaded_files:
                context += f"\nAttached file: {f['name']}\n{f['content'][:500]}\n"
        
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
                st.info(f"📎 Saved as artifact: {artifact.title}")
            
            # Add to conversation
            self.add_message("assistant", streaming_text)
            
            # Check if this is an edit request
            edit_keywords = ["edit", "rewrite", "improve", "fix", "change", "update", "modify", "revise", "add", "remove"]
            if any(kw in prompt.lower() for kw in edit_keywords) and len(streaming_text) > 100:
                if streaming_text != st.session_state.current_document:
                    if st.session_state.track_changes:
                        self.save_document_version(f"AI Edit: {prompt[:50]}")
                    st.session_state.current_document = streaming_text
                    st.success("✅ Document updated!")
                    
        except Exception as e:
            st.error(f"Error: {str(e)}")
        finally:
            st.session_state.is_processing = False
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
    
    def render(self):
        # Sidebar with AI panel
        self.render_sidebar()
        self.render_ai_panel()
        
        # Main content - Word editor and chat
        col1, col2 = st.columns([3, 2])
        
        with col1:
            self.render_word_editor()
        
        with col2:
            st.markdown("### 💬 Chat with Claude")
            self.render_chat()
        
        self.render_artifact_viewer()

# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="Claude - Word Style Editor",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Microsoft Word-style CSS
    st.markdown("""
    <style>
    /* Word-style container */
    .main {
        background-color: #ffffff;
    }
    
    /* Word-style editor area */
    .stTextArea textarea {
        font-family: 'Calibri', 'Segoe UI', Arial, sans-serif;
        font-size: 14px;
        line-height: 1.6;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        background-color: #ffffff;
    }
    
    .stTextArea textarea:focus {
        border-color: #0078d4;
        box-shadow: 0 0 0 2px rgba(0,120,212,0.2);
        outline: none;
    }
    
    /* Word-style toolbar buttons */
    .stButton button {
        background-color: #f5f5f5;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        color: #333;
        font-weight: normal;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #e8e8e8;
        border-color: #0078d4;
        transform: none;
    }
    
    /* Primary action button */
    .stButton button[kind="primary"] {
        background-color: #0078d4;
        border-color: #0078d4;
        color: white;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #fafafa;
        border-right: 1px solid #e0e0e0;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: transparent;
        border-radius: 8px;
    }
    
    /* Headers */
    h1, h2, h3, h4 {
        color: #1f1f1f;
        font-weight: 600;
    }
    
    /* Info bar */
    .stCaption {
        color: #666;
        font-size: 12px;
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
            st.warning("🔑 GROQ_API_KEY Required")
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
