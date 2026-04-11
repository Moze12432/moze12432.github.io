import streamlit as st
from groq import Groq
import requests
import re
import numpy as np
from sentence_transformers import SentenceTransformer

from bs4 import BeautifulSoup
import requests
from urllib.parse import urlparse

import PyPDF2
import docx
from io import StringIO
import csv
import json

# ============================================
# FILE PROCESSING FUNCTIONS - IMPROVED
# ============================================

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text.strip() + "\n"
        
        if not text.strip():
            return "The PDF appears to be empty or contains only scanned images (no extractable text)."
        
        return text[:5000]
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file):
    """Extract text from Word document"""
    try:
        file.seek(0)
        doc = docx.Document(file)
        text = ""
        
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                text += para.text.strip() + "\n\n"
        
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text and cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text += " | ".join(row_text) + "\n"
            text += "\n"
        
        if not text.strip():
            return "The Word document appears to be empty or contains only images/tables that couldn't be extracted."
        
        return text[:5000]
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def extract_text_from_txt(file):
    """Extract text from text file"""
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        if not content.strip():
            return "The text file is empty."
        return content[:5000]
    except UnicodeDecodeError:
        try:
            file.seek(0)
            content = file.read().decode('latin-1')
            return content[:5000]
        except Exception as e:
            return f"Error reading text file: {str(e)}"
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def extract_text_from_csv(file):
    """Extract text from CSV file"""
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        csv_reader = csv.reader(StringIO(content))
        text = "CSV Data:\n\n"
        
        rows = list(csv_reader)
        if rows:
            headers = rows[0]
            text += "Headers: " + " | ".join(headers) + "\n\n"
            text += "Data rows:\n"
            
            for i, row in enumerate(rows[1:11], 1):
                text += f"Row {i}: " + " | ".join(row) + "\n"
            
            if len(rows) > 11:
                text += f"\n... and {len(rows) - 11} more rows"
        
        if not text.strip():
            return "The CSV file appears to be empty."
        
        return text[:5000]
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"

def extract_text_from_json(file):
    """Extract text from JSON file"""
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        data = json.loads(content)
        
        formatted_json = json.dumps(data, indent=2)
        
        if len(formatted_json) > 3000:
            text = "JSON Data Summary:\n\n"
            text += f"Type: {type(data).__name__}\n"
            
            if isinstance(data, dict):
                text += f"Top-level keys: {', '.join(list(data.keys())[:10])}\n"
                if len(data.keys()) > 10:
                    text += f"... and {len(data.keys()) - 10} more keys\n"
                text += "\nFull JSON (truncated):\n"
                text += formatted_json[:3000]
            elif isinstance(data, list):
                text += f"Number of items: {len(data)}\n"
                if len(data) > 0:
                    text += f"First item preview: {json.dumps(data[0], indent=2)[:500]}\n"
                text += "\nFull JSON (truncated):\n"
                text += formatted_json[:3000]
            else:
                text += f"Value: {str(data)[:500]}\n"
        else:
            text = f"JSON Data:\n\n{formatted_json}"
        
        return text[:5000]
    except json.JSONDecodeError as e:
        return f"Error parsing JSON: Invalid JSON format - {str(e)}"
    except Exception as e:
        return f"Error reading JSON file: {str(e)}"

def process_uploaded_file(uploaded_file):
    """Route file to appropriate processor based on type"""
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
        return f"Unsupported file type: {file_type}. Supported: PDF, DOCX, TXT, CSV, JSON"

# ============================================
# CONFIG
# ============================================

MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0
MAX_TOKENS = 800

# ============================================
# STREAMLIT SETTINGS
# ============================================

st.set_page_config(page_title="Mukiibi Moses AI", page_icon="🧠", layout="wide")

# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .stChatMessage [data-testid="stChatMessageContent"]:has(div:first-child) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 12px;
    }
    
    .stChatMessage [data-testid="stChatMessageContent"]:has(div:last-child) {
        background: #f0f2f6;
        color: #1e1e2f;
        border-radius: 15px;
        padding: 12px;
        border-left: 4px solid #764ba2;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: scale(1.05);
        background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    }
    
    /* Fixed bottom chat bar */
    .fixed-chat-bar {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
        z-index: 999;
    }
    
    .stFileUploader > div:first-child {
        display: none;
    }
    
    .stFileUploader button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 8px 16px;
        font-size: 20px;
        font-weight: bold;
        transition: transform 0.2s;
        width: 100%;
        min-width: 60px;
        height: 46px;
    }
    
    .stFileUploader button:hover {
        transform: scale(1.05);
    }
    
    .file-count-badge {
        position: fixed;
        bottom: 90px;
        right: 30px;
        background: #ff4444;
        color: white;
        border-radius: 50%;
        width: 32px;
        height: 32px;
        font-size: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-weight: bold;
        cursor: pointer;
        z-index: 1000;
        box-shadow: 0 2px 10px rgba(0,0,0,0.2);
    }
    
    .main .block-container {
        padding-bottom: 100px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# GROQ CLIENT
# ============================================

client = Groq(
    api_key=st.secrets.get("GROQ_API_KEY")
)

# ============================================
# CURRENT DATE & TIME
# ============================================

from datetime import datetime
import pytz

def get_current_datetime():
    """Get current date and time"""
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    
    return f"""Current Information:
• Date: {now.strftime('%B %d, %Y')}
• Time: {now.strftime('%I:%M %p')}
• Day: {now.strftime('%A')}
• Timezone: Asia/Seoul"""
    
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
        return f"AI service temporarily unavailable."

# ============================================
# UPDATED SYSTEM PROMPT - Clear file usage rules
# ============================================

SYSTEM_PROMPT = """
You are MozeAI, an AI assistant created by Mukiibi Moses, a Computer Engineering student at Kyungdong University.

CAPABILITIES:
- Access to current date/time
- Real-time web search
- Latest news headlines
- Calculator for math problems
- Memory of past conversations
- File analysis (when user explicitly asks about uploaded files)

CRITICAL RULES:
1. ONLY use uploaded file content if the user SPECIFICALLY asks about "the file", "the document", "my upload", or similar explicit references
2. For normal conversation (greetings, "who are you", "what is your purpose", general questions), IGNORE any uploaded files completely
3. Treat each conversation as starting fresh - do not assume the user wants to discuss their files
4. When the user asks general questions like "what is your purpose", answer based on YOUR capabilities as an AI, not based on uploaded files
5. If the user says "leave the document" or "back to normal conversation", completely ignore file context for all subsequent responses

EXAMPLE BEHAVIOR:
- User: "who are you?" → Answer: "I am MozeAI, an AI assistant created by Mukiibi Moses..."
- User: "what is your purpose?" → Answer: "My purpose is to assist you with information, answer questions, and help with tasks..."
- User: "what does my file say?" → Answer: Use file content

Remember: Normal conversation = ignore files. Explicit file questions = use files only when asked.
"""

# ============================================
# SESSION MEMORY & FILES
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
# EMBEDDINGS
# ============================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

# ============================================
# MEMORY FUNCTIONS
# ============================================

def store_memory(text):
    if len(text) < 30:
        return
    vec = embedder.encode(text)
    st.session_state.memory_store.append((text, vec))

def retrieve_memory(query):
    memory_store = st.session_state.memory_store
    if not memory_store:
        return ""
    qvec = embedder.encode(query)
    scores = []
    for text, vec in memory_store:
        sim = np.dot(qvec, vec)
        scores.append((sim, text))
    scores.sort(reverse=True)
    top = scores[:2]
    return "\n".join([t[1][:300] for t in top])

# ============================================
# FILE ANALYSIS FUNCTION
# ============================================

def analyze_uploaded_files(query, file_context, filenames):
    """Analyze uploaded files and answer questions about them"""
    
    analysis_prompt = f"""
You are analyzing uploaded files. Answer the user's question based ONLY on the actual file contents provided below.

**Uploaded Files:**
{filenames}

**ACTUAL FILE CONTENT:**
{file_context}

**User Question:** {query}

**INSTRUCTIONS:**
1. Base your answer on the ACTUAL file content above
2. Quote specific sentences or data from the content
3. If the content shows an error message, inform the user
4. Be specific and detailed

Now analyze and answer.
"""
    
    messages = [
        {"role": "system", "content": "You are a document analysis assistant. Answer based ONLY on the actual file content provided."},
        {"role": "user", "content": analysis_prompt}
    ]
    
    response = llm(messages)
    return clean_answer(response)

# ============================================
# EVALUATION FUNCTION
# ============================================

def evaluate_work(question, file_content):
    """Evaluate user's work from uploaded files"""
    evaluation_prompt = f"""
You are an expert evaluator. Analyze the following file content and answer the user's evaluation request.

**File Content to Evaluate:**
{file_content[:3000]}

**User Request:**
{question}

Please provide:
1. Overall assessment based on actual content
2. Strengths (2-3 points with specific examples)
3. Areas for improvement (2-3 points)
4. Score out of 100 (if applicable)
5. Specific recommendations
"""
    
    messages = [
        {"role": "system", "content": "You are an expert evaluator providing constructive feedback based solely on the provided content."},
        {"role": "user", "content": evaluation_prompt}
    ]
    
    return clean_answer(llm(messages))

# ============================================
# WEB SEARCH FUNCTIONS
# ============================================

def internet_search(query):
    """Search the web for current information"""
    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        
        response = requests.post(url, data=params, headers=headers, timeout=10)
        
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
    except Exception as e:
        return wikipedia_fallback(query)

def wikipedia_fallback(query):
    """Wikipedia as backup search"""
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        q = query.strip().replace(" ", "_")
        r = requests.get(url + q, timeout=10)
        
        if r.status_code == 200:
            data = r.json()
            extract = data.get("extract", "")[:1000]
            if extract:
                return f"Wikipedia information:\n{extract}"
    except:
        pass
    return ""

def get_current_news(topic="latest"):
    """Get current news"""
    try:
        url = f"https://rss2json.com/api.json?rss_url=https://feeds.bbci.co.uk/news/rss.xml"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])[:3]
            
            news_text = "Latest news headlines:\n\n"
            for item in items:
                news_text += f"• {item.get('title', '')}\n"
                news_text += f"  {item.get('description', '')[:150]}...\n\n"
            return news_text[:1000]
    except Exception as e:
        pass
    return ""

# ============================================
# CALCULATOR
# ============================================

def calculator(query):
    try:
        expression = query.lower()
        expression = expression.replace("×","*")
        expression = expression.replace("x","*")
        expression = re.findall(r"[0-9\+\-\*\/\.\(\) ]+", expression)
        if expression:
            result = eval(expression[0])
            return str(result)
    except:
        return None

# ============================================
# WEB PAGE SCRAPER
# ============================================

def scrape_with_requests(url):
    """Try to scrape with different request methods"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
    ]
    
    for user_agent in user_agents:
        try:
            headers = {'User-Agent': user_agent, 'Accept': 'text/html'}
            response = requests.get(url, headers=headers, timeout=15)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                for element in soup(["script", "style", "nav", "footer"]):
                    element.decompose()
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                text = ' '.join(line for line in lines if line)
                if len(text) > 200:
                    return text[:3000]
        except:
            continue
    return None

def scrape_webpage(url):
    """Universal webpage reader"""
    content = scrape_with_requests(url)
    if content:
        return content
    return "Unable to read this website. It may block automated access."

def extract_urls_from_query(query):
    """Extract URLs from user query"""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    return re.findall(url_pattern, query)

# ============================================
# ROUTER FUNCTION
# ============================================

def route(query):
    q = query.lower()
    
    urls = extract_urls_from_query(query)
    if urls:
        return "scrape_url"
    
    file_keywords = [
        "summarize", "analyze this file", "what does the file say", "from the file", 
        "in the document", "based on the file", "tell me about this file", "what is this file",
        "what's in this file", "describe the file", "file content", "document says", 
        "uploaded file", "my file", "what does my file", "in my file"
    ]
    
    if any(x in q for x in file_keywords):
        return "file_task"
    
    if any(x in q for x in ["evaluate", "assess", "grade", "review my", "check my", "score"]):
        return "evaluate"
    
    if any(x in q for x in ["+","-","*","/","×","calculate"]):
        return "calculator"
    
    if any(x in q for x in ["time", "date", "today", "current time", "what day"]):
        return "datetime"
    
    if any(x in q for x in ["news", "headlines", "current events", "breaking"]):
        return "news"
    
    if any(x in q for x in ["who is", "what is", "when did", "where is", "search"]):
        return "search"
    
    return "reason"

# ============================================
# CLEAN OUTPUT
# ============================================

def clean_answer(text):
    if "🧠" in text:
        text = text.split("🧠")[0]
    if "Plan:" in text:
        text = text.split("Plan:")[0]
    return text.strip()

# ============================================
# REASONING
# ============================================

def reason(question, context):
    context = context[:2000]
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"""
Context:
{context}

Question:
{question}

Answer clearly based on the context. For normal conversation, ignore any file content.
"""}
    ]
    return clean_answer(llm(messages))

# ============================================
# UPDATED AGENT FUNCTION WITH RESET CAPABILITY
# ============================================

# ============================================
# UPDATED AGENT FUNCTION - NO RERUNS
# ============================================

def run_agent(query):
    # Check if user wants to reset/ignore files
    reset_phrases = ["leave the document", "back to normal conversation", "ignore the file", "forget the file", "clear context", "start fresh"]
    if any(phrase in query.lower() for phrase in reset_phrases):
        st.session_state.file_context = ""
        st.session_state.uploaded_files = {}
        return "✅ Context cleared! I'll now have a normal conversation with you without referencing any files. How can I help you today?"
    
    tool = route(query)
    context = ""
    
    # FILE TASK - Only if user explicitly asks about files AND files exist
    if tool == "file_task" and st.session_state.file_context:
        filenames = "\n".join([f"- {name}" for name in st.session_state.uploaded_files.keys()])
        with st.spinner("📖 Reading your document..."):
            return analyze_uploaded_files(query, st.session_state.file_context, filenames)
    
    # EVALUATION task
    elif tool == "evaluate" and st.session_state.file_context:
        with st.spinner("📝 Evaluating your work..."):
            return evaluate_work(query, st.session_state.file_context)
    
    # URL Scraping
    elif tool == "scrape_url":
        urls = extract_urls_from_query(query)
        scraped_content = ""
        for url in urls:
            with st.spinner(f"Reading {url}..."):
                content = scrape_webpage(url)
                if content:
                    scraped_content += f"\n\nContent from {url}:\n{content}\n"
        if scraped_content:
            context = scraped_content
        else:
            return "I couldn't read that link."
    
    # Calculator
    if tool == "calculator":
        result = calculator(query)
        if result:
            return result
    
    # Date/Time
    if tool == "datetime":
        context += get_current_datetime()
    
    # News
    if tool == "news":
        news_context = get_current_news(query)
        if news_context:
            context += news_context
    
    # Web Search
    if tool == "search":
        web_context = internet_search(query)
        if web_context:
            context += web_context
    
    # Memory retrieval
    mem = retrieve_memory(query)
    if mem:
        context += "\n\n" + mem
    
    # If no context found, add current date
    if not context:
        context = get_current_datetime()
    
    answer = reason(query, context)
    store_memory(answer)
    
    return answer

# ============================================
# UI - MAIN PAGE
# ============================================

st.markdown('<h1>🧠 Mukiibi-Moses AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #667eea;">Your Intelligent Autonomous Agent</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("### 🧠 MozeAI")
    st.markdown("---")
    
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.memory_store = []
        st.session_state.chat_history = []
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.rerun()
    
    # Clear file context button
    if st.button("🗑️ Clear File Context", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.toast("✅ File context cleared! Now having normal conversation.", icon="🧹")
        st.rerun()
    
    st.markdown("---")
    st.markdown("### 📤 File Upload")
    st.markdown("""
    **Supported files:**
    - PDF, DOCX, TXT, CSV, JSON
    
    **Ask about files:**
    - "What does this file say?"
    - "Summarize my document"
    - "Analyze this data"
    
    **To clear files:**
    - Say "clear context" or click the button above
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Created by **Mukiibi Moses**")
    st.markdown("Computer Engineering @ Kyungdong University")

# ============================================
# CHAT INTERFACE WITH FIXED BOTTOM BAR - FIXED
# ============================================

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# Fixed bottom bar
st.markdown('<div class="fixed-chat-bar">', unsafe_allow_html=True)

col1, col2 = st.columns([6, 1])

with col1:
    query = st.chat_input("Ask me anything...", key="chat_input")

with col2:
    uploaded_files = st.file_uploader(
        "📎",
        type=['pdf', 'docx', 'txt', 'csv', 'json'],
        accept_multiple_files=True,
        key="inline_uploader",
        label_visibility="collapsed",
        help="Upload PDF, DOCX, TXT, CSV, or JSON files"
    )
    
    # Process files if uploaded - REMOVED st.rerun()
    if uploaded_files:
        files_processed = False
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    file_content = process_uploaded_file(file)
                    if file_content and not file_content.startswith("Error"):
                        st.session_state.uploaded_files[file.name] = file_content
                        st.toast(f"✅ Loaded: {file.name}", icon="📎")
                        files_processed = True
                    else:
                        st.toast(f"❌ Failed: {file.name}", icon="⚠️")
        
        # Update file context if any files were processed
        if files_processed and st.session_state.uploaded_files:
            st.session_state.file_context = "\n\n" + ("="*50) + "\n".join([
                f"\n📄 FILE: {name}\n{'-'*40}\n{content}\n" 
                for name, content in st.session_state.uploaded_files.items()
            ])
            # Don't call st.rerun() - let Streamlit handle it naturally

st.markdown('</div>', unsafe_allow_html=True)

# File count badge
if len(st.session_state.uploaded_files) > 0:
    file_count = len(st.session_state.uploaded_files)
    st.markdown(f"""
    <div class="file-count-badge" title="{file_count} file(s) loaded - Click sidebar button to clear">
        📎 {file_count}
    </div>
    """, unsafe_allow_html=True)

# Process chat input
if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)
    
    response = run_agent(query)
    
    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()
