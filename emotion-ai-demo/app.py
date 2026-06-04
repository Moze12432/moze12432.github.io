"""
MOZEAI DOCUMENT STUDIO - Claude-Style Interface
Valid Groq models, proper streaming, Claude-identical UI/UX
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
# CONSTANTS
# ============================================================================

VALID_GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

MODEL_PRIORITY = VALID_GROQ_MODELS
MAX_DOC_CHARS = 12000

SYSTEM_PROMPT = """You are MozeAI, a helpful AI writing assistant integrated into a document editor.

CRITICAL RULES FOR DOCUMENT EDITING:
1. When asked to EDIT a document, wrap the COMPLETE edited document in <document>...</document> tags.
2. Do NOT add any preamble like "Sure!" or "Here's your edited document:" before the <document> tag.
3. If you're just answering a question (not editing), respond normally without <document> tags.
4. Be concise and helpful in regular responses.
5. Preserve the original meaning unless instructed otherwise.

EXAMPLE EDIT RESPONSE:
<document>The completely edited document content goes here...</document>

EXAMPLE QUESTION RESPONSE:
Yes, you can improve passive voice by using active voice."""

# ============================================================================
# DATA CLASSES
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
# GROQ ROUTER
# ============================================================================

class GroqRouter:
    def __init__(self, api_key: str):
        self.client = Groq(api_key=api_key)
        self.current_model = MODEL_PRIORITY[0]
        self.last_error = None

    def stream_chat(self, messages, temperature=0.7, max_tokens=4000):
        last_error = None
        for model in MODEL_PRIORITY:
            try:
                stream = self.client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature, stream=True
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
        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def complete_chat(self, messages, temperature=0.7, max_tokens=4000):
        last_error = None
        for model in MODEL_PRIORITY:
            try:
                response = self.client.chat.completions.create(
                    model=model, messages=messages,
                    max_tokens=max_tokens, temperature=temperature, stream=False
                )
                self.current_model = model
                self.last_error = None
                return response.choices[0].message.content
            except Exception as e:
                last_error = f"{model}: {str(e)}"
                self.last_error = last_error
        raise RuntimeError(f"All models failed. Last error: {last_error}")

# ============================================================================
# UTILITIES
# ============================================================================

def extract_document_from_response(response: str):
    doc_match = re.search(r'<document>(.*?)</document>', response, re.DOTALL)
    if doc_match:
        return doc_match.group(1).strip(), True
    return response, False

def truncate_for_context(text: str, max_chars: int = MAX_DOC_CHARS):
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return text[:half] + "\n\n[... content truncated ...]\n\n" + text[-half:]

def show_diff(old_content: str, new_content: str):
    diff = difflib.unified_diff(old_content.splitlines(), new_content.splitlines(), lineterm='', n=2)
    return "\n".join(diff)

def extract_file_text(uploaded_file):
    try:
        if uploaded_file.type == "application/pdf":
            try:
                import pypdf
                reader = pypdf.PdfReader(BytesIO(uploaded_file.getvalue()))
                return "\n".join(page.extract_text() or "" for page in reader.pages)
            except ImportError:
                return "PDF support requires pypdf."
        elif "wordprocessingml" in uploaded_file.type or uploaded_file.name.endswith('.docx'):
            try:
                import docx
                doc = docx.Document(BytesIO(uploaded_file.getvalue()))
                return "\n".join(p.text for p in doc.paragraphs)
            except ImportError:
                return "DOCX support requires python-docx."
        elif uploaded_file.type == "text/plain":
            return uploaded_file.getvalue().decode('utf-8')
        else:
            return uploaded_file.getvalue().decode('utf-8', errors='ignore')
    except Exception as e:
        return f"Error: {str(e)}"

# ============================================================================
# CLAUDE-STYLE CSS
# ============================================================================

CLAUDE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Styrene+A:wght@400;500&display=swap');

/* ── Reset & Base ── */
* { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #f5f0e8 !important;
    font-family: 'Tiempos Text', Georgia, serif;
    color: #1a1a1a;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header, [data-testid="stToolbar"],
[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}

/* ── Main content padding ── */
.main .block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* ── SIDEBAR — Claude dark panel ── */
[data-testid="stSidebar"] {
    background-color: #2d2d2d !important;
    border-right: none !important;
    width: 260px !important;
}

[data-testid="stSidebar"] * {
    color: #e8e0d0 !important;
}

[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 {
    color: #e8e0d0 !important;
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif;
}

[data-testid="stSidebar"] hr {
    border-color: #444 !important;
    margin: 8px 0 !important;
}

/* Sidebar New Chat button — Claude orange/coral */
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
    background-color: #c96442 !important;
    color: #fff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 8px 12px !important;
    width: 100% !important;
    cursor: pointer !important;
    transition: background 0.15s !important;
}

[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover {
    background-color: #b55538 !important;
}

/* Sidebar conversation buttons */
[data-testid="stSidebar"] .stButton > button:not([kind="primary"]) {
    background-color: transparent !important;
    border: none !important;
    color: #c8bfb0 !important;
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    font-size: 13px !important;
    text-align: left !important;
    padding: 6px 8px !important;
    border-radius: 6px !important;
    width: 100% !important;
    transition: background 0.1s !important;
}

[data-testid="stSidebar"] .stButton > button:not([kind="primary"]):hover {
    background-color: #3d3d3d !important;
    color: #fff !important;
}

/* Sidebar caption/small text */
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] small {
    color: #888 !important;
    font-size: 11px !important;
}

/* Sidebar expander */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: transparent !important;
    border: 1px solid #444 !important;
    border-radius: 6px !important;
}

/* Sidebar slider */
[data-testid="stSidebar"] .stSlider [data-testid="stSliderThumb"] {
    background-color: #c96442 !important;
}

/* ── MAIN AREA — cream/paper ── */
[data-testid="stMain"] {
    background-color: #f5f0e8 !important;
}

/* ── Column layout ── */
[data-testid="column"] {
    padding: 0 !important;
}

/* ── Headings ── */
h1, h2, h3 {
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    color: #1a1a1a !important;
    font-weight: 600 !important;
}

/* ── Document editor area ── */
.doc-panel {
    background: #fff;
    border-right: 1px solid #e5ddd0;
    min-height: 100vh;
    padding: 24px;
}

/* Text area — Word-style white canvas */
.stTextArea textarea {
    background-color: #ffffff !important;
    border: 1px solid #e0d8cc !important;
    border-radius: 6px !important;
    font-family: 'Tiempos Text', Georgia, serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
    color: #1a1a1a !important;
    padding: 16px !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06) !important;
    resize: vertical !important;
}

.stTextArea textarea:focus {
    border-color: #c96442 !important;
    box-shadow: 0 0 0 2px rgba(201,100,66,0.15) !important;
    outline: none !important;
}

/* ── Text input (title) ── */
.stTextInput input {
    background-color: transparent !important;
    border: none !important;
    border-bottom: 1px solid #ddd5c8 !important;
    border-radius: 0 !important;
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    color: #1a1a1a !important;
    padding: 4px 0 8px 0 !important;
}

.stTextInput input:focus {
    border-bottom-color: #c96442 !important;
    box-shadow: none !important;
}

/* ── Toolbar buttons ── */
.stButton > button {
    background-color: #f5f0e8 !important;
    border: 1px solid #ddd5c8 !important;
    border-radius: 6px !important;
    color: #4a4a4a !important;
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
    transition: all 0.15s !important;
    cursor: pointer !important;
}

.stButton > button:hover {
    background-color: #ebe5d8 !important;
    border-color: #c8bfb0 !important;
    color: #1a1a1a !important;
}

/* Apply Edit — primary action, Claude orange */
.stButton > button[kind="primary"] {
    background-color: #c96442 !important;
    border-color: #c96442 !important;
    color: #fff !important;
}

.stButton > button[kind="primary"]:hover {
    background-color: #b55538 !important;
}

/* Download button */
.stDownloadButton > button {
    background-color: #f5f0e8 !important;
    border: 1px solid #ddd5c8 !important;
    border-radius: 6px !important;
    color: #4a4a4a !important;
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    font-size: 12px !important;
    font-weight: 500 !important;
    padding: 6px 10px !important;
}

.stDownloadButton > button:hover {
    background-color: #ebe5d8 !important;
}

/* ── CHAT PANEL ── */
.chat-panel {
    background-color: #f5f0e8;
    padding: 0;
    display: flex;
    flex-direction: column;
    height: 100vh;
}

/* Chat messages */
[data-testid="stChatMessage"] {
    background-color: transparent !important;
    border: none !important;
    padding: 12px 0 !important;
    font-family: 'Tiempos Text', Georgia, serif !important;
    font-size: 15px !important;
    line-height: 1.7 !important;
}

/* User message bubble */
[data-testid="stChatMessage"][data-testid*="user"] {
    background-color: #ece6da !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    margin: 4px 0 !important;
}

/* Avatar — Claude style */
[data-testid="stChatMessageAvatar"] {
    background-color: #c96442 !important;
    border-radius: 50% !important;
    width: 28px !important;
    height: 28px !important;
}

/* Chat input */
[data-testid="stChatInput"] {
    background-color: #fff !important;
    border: 1px solid #ddd5c8 !important;
    border-radius: 24px !important;
    padding: 12px 20px !important;
    font-family: 'Tiempos Text', Georgia, serif !important;
    font-size: 15px !important;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06) !important;
}

[data-testid="stChatInput"]:focus-within {
    border-color: #c96442 !important;
    box-shadow: 0 2px 12px rgba(201,100,66,0.12) !important;
}

/* ── Divider ── */
hr {
    border: none !important;
    border-top: 1px solid #e5ddd0 !important;
    margin: 12px 0 !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background-color: transparent !important;
    border: 1px solid #e5ddd0 !important;
    border-radius: 8px !important;
}

[data-testid="stExpander"] summary {
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    font-size: 13px !important;
    color: #5a5a5a !important;
    font-weight: 500 !important;
}

/* ── Caption / small text ── */
.stCaption, small, [data-testid="stCaption"] {
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    color: #7a7a7a !important;
    font-size: 12px !important;
}

/* ── Success / info / warning ── */
[data-testid="stAlert"] {
    border-radius: 8px !important;
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    font-size: 13px !important;
}

/* ── Checkbox ── */
.stCheckbox label {
    font-family: 'Söhne', ui-sans-serif, system-ui, sans-serif !important;
    font-size: 13px !important;
    color: #c8bfb0 !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #d0c8bc; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #b8b0a4; }

/* ── Column divider ── */
[data-testid="column"]:first-child {
    border-right: 1px solid #e5ddd0;
}

/* ── Spinner ── */
.stSpinner > div {
    border-top-color: #c96442 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background-color: #fff !important;
    border: 1px dashed #ddd5c8 !important;
    border-radius: 8px !important;
}
</style>
"""

# ============================================================================
# MOZEAI UI
# ============================================================================

class MozeUI:
    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        if "conversations" not in st.session_state:
            default_conv = Conversation(id=str(uuid.uuid4()), title="New conversation", messages=[], artifacts=[])
            st.session_state.conversations = [default_conv]
            st.session_state.current_conv_id = default_conv.id
        if "router" not in st.session_state:
            st.session_state.router = None
        if "track_changes" not in st.session_state:
            st.session_state.track_changes = False
        if "temperature" not in st.session_state:
            st.session_state.temperature = 0.7

    def get_conv(self) -> Conversation:
        for conv in st.session_state.conversations:
            if conv.id == st.session_state.current_conv_id:
                return conv
        return st.session_state.conversations[0]

    def add_message(self, role: str, content: str):
        conv = self.get_conv()
        conv.messages.append(Message(role=role, content=content))
        if len(conv.messages) == 1 and role == "user":
            conv.title = content[:35] + ("…" if len(content) > 35 else "")

    def create_artifact(self, title, content, artifact_type=ArtifactType.DOCUMENT):
        conv = self.get_conv()
        art = Artifact(id=str(uuid.uuid4()), title=title, content=content, type=artifact_type)
        conv.artifacts.append(art)
        return art

    def save_version(self, content: str, description: str = ""):
        conv = self.get_conv()
        version = {"id": str(uuid.uuid4()), "content": content,
                   "timestamp": time.time(), "description": description}
        conv.document_versions.append(version)
        if len(conv.document_versions) > 50:
            conv.document_versions = conv.document_versions[-50:]

    def process_edit(self, instruction: str):
        conv = self.get_conv()
        if not st.session_state.router:
            st.error("No API key configured.")
            return
        if not conv.current_document:
            st.warning("No document to edit.")
            return
        try:
            with st.spinner("Editing…"):
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Edit this document: {instruction}\n\nDocument:\n{truncate_for_context(conv.current_document)}"}
                ]
                response = st.session_state.router.complete_chat(messages, temperature=st.session_state.temperature)
                doc_content, is_document = extract_document_from_response(response)
                if is_document and doc_content != conv.current_document and len(doc_content) > 100:
                    if st.session_state.track_changes:
                        self.save_version(conv.current_document, f"AI: {instruction[:30]}")
                    conv.current_document = doc_content
                    st.toast("Document updated!", icon="✅")
                    st.rerun()
                else:
                    st.info("No changes made.")
        except Exception as e:
            st.error(f"Edit failed: {str(e)}")

    # ── SIDEBAR ──────────────────────────────────────────────────────────────

    def render_sidebar(self):
        with st.sidebar:
            # Logo / brand
            st.markdown("""
            <div style="padding: 16px 8px 8px 8px;">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:16px;">
                    <div style="width:32px;height:32px;background:#c96442;border-radius:8px;
                                display:flex;align-items:center;justify-content:center;
                                font-size:16px;">🌿</div>
                    <span style="font-family:'Söhne',sans-serif;font-size:16px;
                                 font-weight:600;color:#e8e0d0;">MozeAI</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            if st.button("✏️  New conversation", use_container_width=True, type="primary", key="new_chat_btn"):
                new_conv = Conversation(id=str(uuid.uuid4()), title="New conversation", messages=[], artifacts=[])
                st.session_state.conversations.insert(0, new_conv)
                st.session_state.current_conv_id = new_conv.id
                st.rerun()

            st.markdown("---")

            # Conversation list — Claude style: just titles, no icons
            if st.session_state.conversations:
                st.markdown('<p style="font-size:11px;color:#666;font-family:sans-serif;'
                            'text-transform:uppercase;letter-spacing:0.08em;'
                            'margin:0 0 6px 4px;">Recent</p>', unsafe_allow_html=True)
                for i, conv in enumerate(st.session_state.conversations[:30]):
                    is_active = conv.id == st.session_state.current_conv_id
                    label = conv.title[:28] + ("…" if len(conv.title) > 28 else "")
                    btn_style = "background:#3d3d3d!important;" if is_active else ""
                    if st.button(label, use_container_width=True, key=f"conv_{conv.id}_{i}"):
                        st.session_state.current_conv_id = conv.id
                        st.rerun()

            st.markdown("---")

            # AI Actions — Claude-style quick edits
            st.markdown('<p style="font-size:11px;color:#666;font-family:sans-serif;'
                        'text-transform:uppercase;letter-spacing:0.08em;'
                        'margin:0 0 6px 4px;">Edit document</p>', unsafe_allow_html=True)

            quick_edits = [
                ("Improve writing", "Improve grammar, clarity, and flow"),
                ("Make professional", "Make the tone professional and polished"),
                ("Add structure", "Add headings and organize content"),
                ("Fix grammar", "Fix only grammar and spelling"),
                ("Make concise", "Make more concise while preserving meaning"),
                ("Academic tone", "Rewrite in academic tone"),
            ]
            for label, instruction in quick_edits:
                if st.button(label, use_container_width=True, key=f"quick_{label[:12]}"):
                    self.process_edit(instruction)

            st.markdown("---")

            # Custom instruction
            st.markdown('<p style="font-size:11px;color:#666;font-family:sans-serif;'
                        'text-transform:uppercase;letter-spacing:0.08em;'
                        'margin:0 0 6px 4px;">Custom edit</p>', unsafe_allow_html=True)
            custom_instruction = st.text_area(
                "Custom", placeholder="Tell MozeAI what to change…",
                height=72, key="custom_edit_input", label_visibility="collapsed"
            )
            if st.button("Apply edit", type="primary", use_container_width=True, key="apply_edit_btn"):
                if custom_instruction:
                    self.process_edit(custom_instruction)

            st.markdown("---")

            # Settings
            with st.expander("Settings"):
                st.session_state.temperature = st.slider(
                    "Temperature", 0.0, 1.0, st.session_state.temperature, key="temp_slider"
                )
                st.session_state.track_changes = st.checkbox(
                    "Track changes", st.session_state.track_changes, key="track_checkbox"
                )

            # Model info
            if st.session_state.router:
                model_short = st.session_state.router.current_model.split("-")[0]
                st.caption(f"⚡ {st.session_state.router.current_model}")
                if st.session_state.router.last_error:
                    st.caption(f"⚠ {st.session_state.router.last_error[:50]}")

    # ── DOCUMENT EDITOR ──────────────────────────────────────────────────────

    def render_document_editor(self):
        conv = self.get_conv()

        # Top bar — title + toolbar
        st.markdown('<div style="padding: 20px 24px 0 24px;">', unsafe_allow_html=True)

        # Document title
        title = st.text_input(
            "doc_title", value=conv.document_title,
            key="doc_title_input", label_visibility="collapsed",
            placeholder="Untitled document"
        )
        if title != conv.document_title:
            conv.document_title = title

        # Toolbar
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1.4, 1, 1])
        with col1:
            if st.button("New", use_container_width=True, key="new_doc_btn"):
                conv.current_document = ""
                conv.document_title = "Untitled Document"
                st.rerun()
        with col2:
            if st.button("Save version", use_container_width=True, key="save_version_btn"):
                self.save_version(conv.current_document, "Manual save")
                st.toast("Version saved!", icon="💾")
        with col3:
            st.download_button(
                label="Download .txt",
                data=conv.current_document if conv.current_document else " ",
                file_name=f"{conv.document_title}.txt",
                mime="text/plain",
                key="download_file_btn",
                use_container_width=True,
                disabled=not conv.current_document
            )
        with col4:
            if st.button("Stats", use_container_width=True, key="stats_doc_btn"):
                words = len(conv.current_document.split())
                st.toast(f"{words:,} words · {len(conv.current_document):,} chars", icon="📊")
        with col5:
            if st.button("Clear", use_container_width=True, key="clear_doc_btn"):
                conv.current_document = ""
                st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div style="padding: 8px 24px 24px 24px;">', unsafe_allow_html=True)

        # Document content
        content = st.text_area(
            "content", value=conv.current_document,
            height=460, key="doc_content_area",
            placeholder="Start writing, or paste text to edit with AI…",
            label_visibility="collapsed"
        )
        if content != conv.current_document:
            conv.current_document = content

        # Word count bar — Claude style subtle
        words = len(conv.current_document.split()) if conv.current_document else 0
        chars = len(conv.current_document) if conv.current_document else 0
        st.caption(f"{words:,} words · {chars:,} characters")

        # Artifacts
        if conv.artifacts:
            with st.expander(f"Saved versions ({len(conv.artifacts)})", expanded=False):
                for art in reversed(conv.artifacts[-10:]):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        st.caption(f"📄 {art.title}")
                    with c2:
                        if st.button("Load", key=f"load_art_{art.id}"):
                            conv.current_document = art.content
                            st.toast("Loaded!", icon="✅")
                            st.rerun()
                    with c3:
                        st.download_button(
                            "↓", data=art.content,
                            file_name=f"{art.title}.txt", mime="text/plain",
                            key=f"dl_art_{art.id}", use_container_width=True
                        )

        # Diff view
        if conv.active_diff:
            with st.expander("Changes", expanded=True):
                st.code(conv.active_diff, language="diff")
                if st.button("Dismiss", key="close_diff_btn"):
                    conv.active_diff = ""
                    st.rerun()

        # Version history
        if conv.document_versions:
            with st.expander("Version history", expanded=False):
                for version in reversed(conv.document_versions[-10:]):
                    c1, c2, c3 = st.columns([3, 1, 1])
                    with c1:
                        t = datetime.fromtimestamp(version["timestamp"]).strftime("%H:%M:%S")
                        st.caption(f"{t} — {version['description']}")
                    with c2:
                        if st.button("Diff", key=f"diff_{version['id']}"):
                            conv.active_diff = show_diff(version["content"], conv.current_document)
                            st.rerun()
                    with c3:
                        if st.button("Restore", key=f"restore_{version['id']}"):
                            conv.current_document = version["content"]
                            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    # ── CHAT PANEL ───────────────────────────────────────────────────────────

    def render_chat(self):
        conv = self.get_conv()

        # Header — "MozeAI" with model badge, just like Claude shows model
        st.markdown(f"""
        <div style="padding:16px 20px 8px 20px; border-bottom:1px solid #e5ddd0;
                    display:flex; align-items:center; gap:10px;">
            <div style="width:28px;height:28px;background:#c96442;border-radius:50%;
                        display:flex;align-items:center;justify-content:center;font-size:14px;">🌿</div>
            <span style="font-family:'Söhne',sans-serif;font-size:14px;font-weight:600;color:#1a1a1a;">MozeAI</span>
            <span style="font-family:'Söhne',sans-serif;font-size:11px;color:#999;
                         background:#f0ebe3;padding:2px 8px;border-radius:10px;border:1px solid #e0d8cc;">
                llama-3.3-70b
            </span>
        </div>
        """, unsafe_allow_html=True)

        # Empty state — Claude-style centered greeting
        if not conv.messages:
            st.markdown("""
            <div style="display:flex;flex-direction:column;align-items:center;
                        justify-content:center;padding:60px 24px;text-align:center;">
                <div style="width:56px;height:56px;background:#c96442;border-radius:16px;
                            display:flex;align-items:center;justify-content:center;
                            font-size:28px;margin-bottom:20px;">🌿</div>
                <h2 style="font-family:'Söhne',sans-serif;font-size:22px;
                           font-weight:600;color:#1a1a1a;margin:0 0 8px 0;">
                    How can I help?
                </h2>
                <p style="font-family:'Tiempos Text',Georgia,serif;font-size:14px;
                          color:#7a7a7a;margin:0;max-width:280px;line-height:1.6;">
                    Write or paste a document on the left, then ask me to edit, improve, or discuss it.
                </p>
            </div>
            """, unsafe_allow_html=True)

        # Messages
        for msg in conv.messages:
            with st.chat_message(msg.role):
                st.markdown(msg.content)

        # File attach
        with st.expander("Attach file", expanded=False):
            uploaded = st.file_uploader(
                "file", type=['pdf', 'docx', 'txt'],
                label_visibility="collapsed", key="chat_uploader"
            )
            if uploaded and uploaded.name not in conv.seen_files:
                conv.seen_files.add(uploaded.name)
                text = extract_file_text(uploaded)
                conv.current_document = text
                self.add_message("user", f"[Attached: {uploaded.name}]")
                st.success(f"Loaded {uploaded.name}")
                st.rerun()

        # Chat input
        prompt = st.chat_input("Message MozeAI…")
        if prompt:
            if not st.session_state.router:
                st.error("No API key configured.")
                return

            self.add_message("user", prompt)

            with st.chat_message("assistant"):
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                for msg in conv.messages[:-1][-19:]:
                    messages.append({"role": msg.role, "content": msg.content})

                full_prompt = prompt
                if conv.current_document:
                    full_prompt = f"{prompt}\n\nUser's document:\n{truncate_for_context(conv.current_document)}"
                messages.append({"role": "user", "content": full_prompt})

                try:
                    response = st.write_stream(
                        st.session_state.router.stream_chat(messages, temperature=st.session_state.temperature)
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
                self.create_artifact(f"Edit: {prompt[:30]}", doc_content)

            self.add_message("assistant", response)

            if document_updated:
                st.rerun()

    # ── RENDER ───────────────────────────────────────────────────────────────

    def render(self):
        self.render_sidebar()
        col1, col2 = st.columns([3, 2])
        with col1:
            self.render_document_editor()
        with col2:
            self.render_chat()


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.set_page_config(
        page_title="MozeAI",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.markdown(CLAUDE_CSS, unsafe_allow_html=True)

    ui = MozeUI()

    # API key resolution
    api_key = None
    try:
        if "GROQ_API_KEY" in st.secrets:
            api_key = st.secrets["GROQ_API_KEY"]
    except Exception:
        pass

    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY")

    if api_key and st.session_state.router is None:
        st.session_state.router = GroqRouter(api_key)

    if not api_key:
        with st.sidebar:
            st.markdown("""
            <div style="padding:16px 8px 8px;">
                <div style="display:flex;align-items:center;gap:10px;margin-bottom:20px;">
                    <div style="width:32px;height:32px;background:#c96442;border-radius:8px;
                                font-size:16px;display:flex;align-items:center;justify-content:center;">🌿</div>
                    <span style="font-size:16px;font-weight:600;color:#e8e0d0;">MozeAI</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            st.error("Groq API key required")
            st.markdown("Get yours at [console.groq.com](https://console.groq.com)")
            key_input = st.text_input("API Key", type="password", key="api_input", label_visibility="collapsed",
                                      placeholder="gsk_…")
            if key_input:
                st.session_state.router = GroqRouter(key_input)
                st.rerun()
        # Show empty main area
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;height:80vh;">
            <div style="text-align:center;">
                <div style="font-size:48px;margin-bottom:16px;">🌿</div>
                <h2 style="font-family:'Söhne',sans-serif;color:#1a1a1a;">Welcome to MozeAI</h2>
                <p style="color:#7a7a7a;font-family:Georgia,serif;">Add your Groq API key in the sidebar to get started.</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        return

    ui.render()


if __name__ == "__main__":
    main()
