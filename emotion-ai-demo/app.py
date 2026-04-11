import streamlit as st
from groq import Groq
import requests
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from bs4 import BeautifulSoup
from urllib.parse import urlparse

import PyPDF2
import docx
from io import StringIO
import csv
import json
from datetime import datetime
import pytz

# ================= FILE PROCESSING =================
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        return "".join([p.extract_text() for p in pdf_reader.pages])[:5000]
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file):
    try:
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])[:5000]
    except Exception as e:
        return f"Error reading DOCX: {str(e)}"

def extract_text_from_txt(file):
    try:
        return file.read().decode("utf-8")[:5000]
    except Exception as e:
        return f"Error reading TXT: {str(e)}"

def extract_text_from_csv(file):
    try:
        content = file.read().decode("utf-8")
        return content[:5000]
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

def extract_text_from_json(file):
    try:
        content = file.read().decode("utf-8")
        return json.dumps(json.loads(content), indent=2)[:5000]
    except Exception as e:
        return f"Error reading JSON: {str(e)}"

def process_uploaded_file(uploaded_file):
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"): return extract_text_from_pdf(uploaded_file)
    if name.endswith(".docx"): return extract_text_from_docx(uploaded_file)
    if name.endswith(".txt"): return extract_text_from_txt(uploaded_file)
    if name.endswith(".csv"): return extract_text_from_csv(uploaded_file)
    if name.endswith(".json"): return extract_text_from_json(uploaded_file)
    return "Unsupported file type"

# ================= CONFIG =================
MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0
MAX_TOKENS = 400

st.set_page_config(page_title="MozeAI", page_icon="🧠")

# ================= CSS (FIXED UI WITH TOOLTIP) =================
st.markdown("""
<style>
/* Chat input fixed at bottom */
.stChatInputContainer {
    position: fixed;
    bottom: 20px;
    left: 0;
    right: 0;
    display: flex;
    justify-content: center;
    z-index: 1000;
}

.stChatInputContainer > div {
    width: 100%;
    max-width: 800px;
    background: #40414f;
    border-radius: 25px;
    padding: 10px 60px 10px 15px;
    border: 1px solid #555;
}

.stChatInputContainer textarea {
    background: transparent !important;
    color: white !important;
    border: none !important;
}

/* Hide default file uploader text */
div[data-testid="stFileUploader"] > div:first-child {
    display: none;
}

div[data-testid="stFileUploader"] {
    position: fixed;
    bottom: 32px;
    right: calc(50% - 360px);
    z-index: 1001;
}

div[data-testid="stFileUploader"] button {
    background: transparent;
    border: none;
    font-size: 22px;
    padding: 0;
    margin: 0;
    width: 36px;
    height: 36px;
    cursor: pointer;
    color: #888;
    transition: all 0.3s ease;
}

div[data-testid="stFileUploader"] button:hover {
    color: #667eea;
    transform: scale(1.1);
}

/* Tooltip on hover */
.upload-icon {
    position: fixed;
    bottom: 32px;
    right: calc(50% - 360px);
    font-size: 22px;
    cursor: pointer;
    z-index: 1002;
    color: #888;
    transition: all 0.3s ease;
}

.upload-icon:hover {
    color: #667eea;
    transform: scale(1.1);
}

/* Hide the actual file uploader display */
.main > div {
    padding-bottom: 120px;
}

/* Active files indicator */
.active-files {
    position: fixed;
    bottom: 80px;
    left: 50%;
    transform: translateX(-50%);
    background: rgba(64, 65, 79, 0.9);
    color: white;
    padding: 5px 15px;
    border-radius: 20px;
    font-size: 12px;
    z-index: 999;
    backdrop-filter: blur(10px);
    white-space: nowrap;
}

/* Sidebar styling */
.css-1d391kg {
    background: linear-gradient(180deg, #1e1e2f 0%, #2d2d44 100%);
}

.stButton button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 10px;
    font-weight: bold;
    transition: transform 0.2s;
}

.stButton button:hover {
    transform: scale(1.05);
}
</style>
""", unsafe_allow_html=True)

# ================= CLIENT =================
client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

def llm(messages):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=messages
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return "AI service temporarily unavailable."

# ================= SYSTEM PROMPT =================
SYSTEM_PROMPT = """
You are MozeAI, an AI with REAL-TIME information access and file analysis capabilities.

Created by Mukiibi Moses, a Computer Engineering student at Kyungdong University.
He is an AI builder focused on designing intelligent autonomous agents, language model applications, and practical AI systems that solve real-world problems such as education, automation, and decision support.

CAPABILITIES:
- Read and analyze uploaded files (PDF, DOCX, TXT, CSV, JSON)
- Access to current date/time
- Real-time web search
- Latest news headlines
- Calculator for math problems
- Memory of past conversations

INSTRUCTIONS:
- When users upload files, analyze them thoroughly
- Provide summaries, answer questions, evaluate content based on user requests
- For evaluation tasks, provide constructive feedback with strengths, improvements, and scores
- When a user provides a URL, focus on answering based on that webpage's content
- Answer clearly and factually
- If information isn't in context, say "I don't have that information"
- Do not hallucinate or make up dates/events
"""

# ================= SESSION STATE =================
if "memory_store" not in st.session_state:
    st.session_state.memory_store = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}

if "file_context" not in st.session_state:
    st.session_state.file_context = ""

# ================= EMBEDDINGS =================
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ================= CURRENT DATE/TIME =================
def get_current_datetime():
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    return f"""Current Information:
• Date: {now.strftime('%B %d, %Y')}
• Time: {now.strftime('%I:%M %p')}
• Day: {now.strftime('%A')}"""

# ================= MEMORY FUNCTIONS =================
def store_memory(text):
    if len(text) < 30:
        return
    vec = embedder.encode(text)
    st.session_state.memory_store.append((text, vec))

def retrieve_memory(query):
    if not st.session_state.memory_store:
        return ""
    qvec = embedder.encode(query)
    scores = []
    for text, vec in st.session_state.memory_store:
        sim = np.dot(qvec, vec)
        scores.append((sim, text))
    scores.sort(reverse=True)
    return "\n".join([t[1][:300] for t in scores[:2]])

# ================= WEB SEARCH =================
def internet_search(query):
    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.post(url, data=params, headers=headers)
        
        if response.status_code == 200:
            results = re.findall(r'<a rel="nofollow" class="result__a" href="[^"]*">([^<]+)</a>', response.text)
            snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', response.text)
            
            if results and snippets:
                context = f"Recent search results for '{query}':\n\n"
                for i in range(min(3, len(results))):
                    context += f"• {results[i]}\n"
                    if i < len(snippets):
                        context += f"  {snippets[i]}\n\n"
                return context[:1500]
        return wikipedia_fallback(query)
    except:
        return wikipedia_fallback(query)

def wikipedia_fallback(query):
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        q = query.strip().replace(" ", "_")
        r = requests.get(url + q)
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")[:1000]
            if extract:
                return f"Wikipedia information:\n{extract}"
    except:
        pass
    return ""

def get_current_news(topic="latest"):
    try:
        api_key = st.secrets.get("NEWS_API_KEY", "")
        if not api_key:
            url = "https://rss2json.com/api.json?rss_url=https://feeds.bbci.co.uk/news/rss.xml"
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])[:3]
                news_text = "Latest news headlines:\n\n"
                for item in items:
                    news_text += f"• {item.get('title', '')}\n"
                    news_text += f"  {item.get('description', '')[:150]}...\n\n"
                return news_text[:1000]
    except:
        pass
    return ""

# ================= CALCULATOR =================
def calculator(query):
    try:
        expression = query.lower()
        expression = expression.replace("×", "*").replace("x", "*")
        expression = re.findall(r"[0-9\+\-\*\/\.\(\) ]+", expression)
        if expression:
            return str(eval(expression[0]))
    except:
        return None

# ================= WEB SCRAPING =================
def scrape_with_requests(url):
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    ]
    for user_agent in user_agents:
        try:
            headers = {'User-Agent': user_agent}
            response = requests.get(url, headers=headers, timeout=15)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                for element in soup(["script", "style", "nav", "footer", "header"]):
                    element.decompose()
                text = soup.get_text()
                text = ' '.join(text.split())
                if len(text) > 200:
                    return text[:3000]
        except:
            continue
    return None

def scrape_webpage(url):
    content = scrape_with_requests(url)
    if content:
        return content
    return f"Unable to read {url}. Please open it in your browser."

def extract_urls_from_query(query):
    return re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*', query)

# ================= ROUTER =================
def route(query):
    q = query.lower()
    if extract_urls_from_query(query):
        return "scrape_url"
    if any(x in q for x in ["summarize", "analyze", "from the file", "in the document", "based on the file"]):
        return "file_task"
    if any(x in q for x in ["+", "-", "*", "/", "calculate"]):
        return "calculator"
    if any(x in q for x in ["time", "date", "today"]):
        return "datetime"
    if any(x in q for x in ["news", "headlines"]):
        return "news"
    if any(x in q for x in ["evaluate", "assess", "grade", "review", "score"]):
        return "evaluate"
    if any(x in q for x in ["capital", "population", "who is", "what is", "tell me about"]):
        return "search"
    return "reason"

# ================= CLEAN OUTPUT =================
def clean_answer(text):
    if "🧠" in text:
        text = text.split("🧠")[0]
    if "Plan:" in text:
        text = text.split("Plan:")[0]
    return text.strip()

# ================= REASONING =================
def reason(question, context):
    context = context[:1500]
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{question}\n\nAnswer clearly."}
    ]
    return clean_answer(llm(messages))

# ================= EVALUATION =================
def evaluate_work(question, file_content):
    prompt = f"""
Analyze the following file content and answer the user's evaluation request.

File Content:
{file_content[:3000]}

User Request:
{question}

Provide:
1. Overall assessment
2. Strengths (2-3 points)
3. Areas for improvement (2-3 points)
4. Score out of 100
5. Specific recommendations

Be constructive and actionable.
"""
    messages = [
        {"role": "system", "content": "You are an expert evaluator providing constructive feedback."},
        {"role": "user", "content": prompt}
    ]
    return clean_answer(llm(messages))

# ================= AGENT =================
def run_agent(query):
    tool = route(query)
    context = ""
    
    if tool == "file_task" and st.session_state.file_context:
        context = f"\n\nUploaded Files Content:\n{st.session_state.file_context}\n"
    
    elif tool == "evaluate" and st.session_state.file_context:
        return evaluate_work(query, st.session_state.file_context)
    
    elif tool == "scrape_url":
        urls = extract_urls_from_query(query)
        scraped = ""
        for url in urls:
            content = scrape_webpage(url)
            if content:
                scraped += f"\n\nContent from {url}:\n{content}\n"
        if scraped:
            context = scraped
        else:
            return "I couldn't read that link. Try asking me to search instead."
    
    if tool == "calculator":
        result = calculator(query)
        if result:
            return result
    
    if tool == "datetime":
        context += get_current_datetime()
    
    if tool == "news":
        news = get_current_news(query)
        if news:
            context += news
    
    if tool == "search":
        web = internet_search(query)
        if web:
            context += web
    
    mem = retrieve_memory(query)
    if mem:
        context += "\n\n" + mem
    
    if not context and not st.session_state.file_context:
        context = get_current_datetime()
    
    answer = reason(query, context)
    store_memory(answer)
    return answer

# ================= SIDEBAR =================
with st.sidebar:
    st.markdown("### 🧠 MozeAI")
    st.markdown("---")
    
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.memory_store = []
        st.session_state.chat_history = []
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📤 Tips")
    st.markdown("**Supported:** PDF, DOCX, TXT, CSV, JSON")
    st.markdown("**Try:** Summarize, Evaluate, Analyze")
    st.markdown("---")
    st.markdown("### About")
    st.markdown("**Mukiibi Moses**")
    st.markdown("Computer Engineering @ Kyungdong University")
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("✅ File reading & analysis")
    st.markdown("✅ Document summarization")
    st.markdown("✅ Work evaluation & grading")
    st.markdown("✅ Web search & news")
    st.markdown("✅ Calculator & memory")

# ================= UI =================
st.title("🧠 Mukiibi Moses AI")
st.markdown('<p style="text-align: center; color: #667eea;">Your Intelligent Autonomous Agent</p>', unsafe_allow_html=True)
st.markdown("---")

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# Chat input
query = st.chat_input("Message MozeAI...")

# Hidden file uploader with tooltip
uploaded_file = st.file_uploader("", label_visibility="collapsed", type=['pdf', 'docx', 'txt', 'csv', 'json'])

# Process uploaded file
if uploaded_file:
    if uploaded_file.name not in st.session_state.uploaded_files:
        with st.spinner(f"Reading {uploaded_file.name}..."):
            content = process_uploaded_file(uploaded_file)
            if content and not content.startswith("Error"):
                st.session_state.uploaded_files[uploaded_file.name] = content
                st.session_state.file_context = "\n\n".join([
                    f"=== {name} ===\n{content}" 
                    for name, content in st.session_state.uploaded_files.items()
                ])
                st.success(f"✅ Loaded: {uploaded_file.name}")
                st.rerun()
            else:
                st.error(f"❌ Failed: {uploaded_file.name}")

# Show active files
if st.session_state.uploaded_files:
    file_list = ', '.join(list(st.session_state.uploaded_files.keys())[:2])
    extra = f" +{len(st.session_state.uploaded_files)-2}" if len(st.session_state.uploaded_files) > 2 else ""
    st.markdown(f'<div class="active-files">📎 {file_list}{extra}</div>', unsafe_allow_html=True)
    
    if st.button("🗑️ Clear all", key="clear_files"):
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.rerun()

# Process chat query
if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)
    
    response = run_agent(query)
    
    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()
