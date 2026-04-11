# ============================================
# COMPLETE WORKING CODE - NO HIDING ANYTHING
# ============================================

import streamlit as st
from groq import Groq
import requests
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import PyPDF2
import docx
from io import StringIO
import csv
import json
from datetime import datetime
import pytz

# ============================================
# FILE PROCESSING FUNCTIONS
# ============================================

def extract_text_from_pdf(file):
    try:
        file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text.strip() + "\n"
        return text[:5000] if text.strip() else "No text found in PDF"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_docx(file):
    try:
        file.seek(0)
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                text += para.text.strip() + "\n\n"
        return text[:5000] if text.strip() else "No text found in document"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_txt(file):
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        return content[:5000] if content.strip() else "File is empty"
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_csv(file):
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        csv_reader = csv.reader(StringIO(content))
        text = "CSV Data:\n\n"
        rows = list(csv_reader)
        if rows:
            text += "Headers: " + " | ".join(rows[0]) + "\n\n"
            for i, row in enumerate(rows[1:11], 1):
                text += f"Row {i}: " + " | ".join(row) + "\n"
        return text[:5000]
    except Exception as e:
        return f"Error: {str(e)}"

def extract_text_from_json(file):
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        data = json.loads(content)
        return json.dumps(data, indent=2)[:5000]
    except Exception as e:
        return f"Error: {str(e)}"

def process_uploaded_file(uploaded_file):
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()
    
    if file_type == "application/pdf" or file_name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    elif file_type == "text/plain" or file_name.endswith('.txt'):
        return extract_text_from_txt(uploaded_file)
    elif file_type == "text/csv" or file_name.endswith('.csv'):
        return extract_text_from_csv(uploaded_file)
    elif file_type == "application/json" or file_name.endswith('.json'):
        return extract_text_from_json(uploaded_file)
    else:
        return f"Unsupported file type"

# ============================================
# CONFIG
# ============================================

MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0
MAX_TOKENS = 800

st.set_page_config(page_title="Mukiibi Moses AI", page_icon="🧠")

# ============================================
# SIMPLE CSS - ONLY POSITION CHAT INPUT
# ============================================

st.markdown("""
<style>
    /* ONLY position the chat input - nothing else */
    .stChatInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: white !important;
        padding: 10px !important;
        z-index: 999 !important;
    }
    
    /* Add padding to bottom of main content */
    .main .block-container {
        padding-bottom: 80px !important;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# GROQ CLIENT
# ============================================

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

def get_current_datetime():
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    return f"Current: {now.strftime('%B %d, %Y - %I:%M %p')}"

def llm(messages):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=messages
        )
        return completion.choices[0].message.content.strip()
    except:
        return "AI service temporarily unavailable."

# ============================================
# SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are MozeAI, an AI assistant created by Mukiibi Moses.

RULES:
- ONLY use uploaded file content if user explicitly asks about "the file", "the document", or "my upload"
- For normal conversation (greetings, "who are you", "what is your purpose"), ignore files completely
- Answer clearly and conversationally
"""

# ============================================
# SESSION STATE
# ============================================

if "memory_store" not in st.session_state:
    st.session_state.memory_store = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "file_context" not in st.session_state:
    st.session_state.file_context = ""

# ============================================
# SIMPLE FUNCTIONS
# ============================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def store_memory(text):
    if len(text) > 30:
        vec = embedder.encode(text)
        st.session_state.memory_store.append((text, vec))

def retrieve_memory(query):
    if not st.session_state.memory_store:
        return ""
    qvec = embedder.encode(query)
    scores = [(np.dot(qvec, vec), text) for text, vec in st.session_state.memory_store]
    scores.sort(reverse=True)
    return "\n".join([t[1][:300] for t in scores[:2]])

def internet_search(query):
    try:
        url = "https://html.duckduckgo.com/html/"
        response = requests.post(url, data={"q": query}, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if response.status_code == 200:
            results = re.findall(r'<a rel="nofollow" class="result__a" href="[^"]*">([^<]+)</a>', response.text)
            if results:
                return f"Search results: {', '.join(results[:3])}"
    except:
        pass
    return ""

def calculator(query):
    try:
        expression = re.findall(r"[0-9\+\-\*\/\.\(\) ]+", query.lower().replace("×", "*").replace("x", "*"))
        if expression:
            return str(eval(expression[0]))
    except:
        return None

def route(query):
    q = query.lower()
    if any(x in q for x in ["file", "document", "upload", "summary of"]):
        return "file_task"
    if any(x in q for x in ["+", "-", "*", "/", "calculate"]):
        return "calculator"
    if any(x in q for x in ["time", "date"]):
        return "datetime"
    if any(x in q for x in ["news", "headlines"]):
        return "news"
    if any(x in q for x in ["who is", "what is", "search"]):
        return "search"
    return "reason"

def clean_answer(text):
    return text.split("🧠")[0].split("Plan:")[0].strip()

def reason(question, context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context: {context[:1500]}\n\nQuestion: {question}\nAnswer:"}
    ]
    return clean_answer(llm(messages))

def analyze_uploaded_files(query, file_context, filenames):
    prompt = f"Based ONLY on these files:\n{filenames}\n\nContent:\n{file_context[:3000]}\n\nQuestion: {query}\nAnswer based only on the file content."
    messages = [{"role": "system", "content": "You analyze files. Answer only from the content provided."}, {"role": "user", "content": prompt}]
    return clean_answer(llm(messages))

def run_agent(query):
    reset_phrases = ["leave the document", "clear context", "forget the file", "start fresh"]
    if any(phrase in query.lower() for phrase in reset_phrases):
        st.session_state.file_context = ""
        st.session_state.uploaded_files = {}
        return "Context cleared! How can I help you today?"
    
    tool = route(query)
    context = ""
    
    if tool == "file_task" and st.session_state.file_context:
        filenames = "\n".join(st.session_state.uploaded_files.keys())
        return analyze_uploaded_files(query, st.session_state.file_context, filenames)
    
    if tool == "calculator":
        result = calculator(query)
        if result:
            return result
    
    if tool == "datetime":
        context += get_current_datetime()
    
    if tool == "news":
        news = internet_search(query + " news")
        if news:
            context += news
    
    if tool == "search":
        search_result = internet_search(query)
        if search_result:
            context += search_result
    
    mem = retrieve_memory(query)
    if mem:
        context += "\n" + mem
    
    if not context:
        context = get_current_datetime()
    
    answer = reason(query, context)
    store_memory(answer)
    return answer

# ============================================
# UI - MAIN DISPLAY
# ============================================

st.markdown('<h1 style="text-align: center;">🧠 Mukiibi-Moses AI</h1>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("### MozeAI")
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.memory_store = []
        st.session_state.chat_history = []
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.rerun()
    
    if st.button("🗑️ Clear Files", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.rerun()
    
    st.markdown("---")
    st.markdown("**Upload Files**")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'csv', 'json'],
        accept_multiple_files=True,
        key="sidebar_uploader"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                content = process_uploaded_file(file)
                if content and not content.startswith("Error"):
                    st.session_state.uploaded_files[file.name] = content
                    st.success(f"✅ {file.name}")
        
        if st.session_state.uploaded_files:
            st.session_state.file_context = "\n\n".join([
                f"FILE: {name}\n{content}" for name, content in st.session_state.uploaded_files.items()
            ])
    
    if st.session_state.uploaded_files:
        st.info(f"📄 {len(st.session_state.uploaded_files)} file(s) loaded")

# ============================================
# CHAT DISPLAY
# ============================================

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# Chat input (this will be fixed at bottom by CSS)
query = st.chat_input("Ask me anything...")

# Process query
if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)
    
    response = run_agent(query)
    
    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()
