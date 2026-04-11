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

# ============================================
# FILE PROCESSING FUNCTIONS
# ============================================

def extract_text_from_pdf(file):
    """Extract text from PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text[:5000]
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file):
    """Extract text from Word document"""
    try:
        doc = docx.Document(file)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text[:5000]
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def extract_text_from_txt(file):
    """Extract text from text file"""
    try:
        content = file.read().decode('utf-8')
        return content[:5000]
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def extract_text_from_csv(file):
    """Extract text from CSV file"""
    try:
        content = file.read().decode('utf-8')
        csv_reader = csv.reader(StringIO(content))
        text = ""
        for row in csv_reader:
            text += " | ".join(row) + "\n"
        return f"CSV Data:\n{text[:5000]}"
    except Exception as e:
        return f"Error reading CSV file: {str(e)}"

def extract_text_from_json(file):
    """Extract text from JSON file"""
    try:
        content = file.read().decode('utf-8')
        data = json.loads(content)
        return f"JSON Data:\n{json.dumps(data, indent=2)[:5000]}"
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
MAX_TOKENS = 400

# ============================================
# STREAMLIT SETTINGS
# ============================================

st.set_page_config(page_title="Mukiibi Moses AI", page_icon="🧠")

# ============================================
# CUSTOM CSS FOR PERSONAL STYLING
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
    
    .css-1d391kg {
        background: linear-gradient(180deg, #1e1e2f 0%, #2d2d44 100%);
    }
    
    .stChatInputContainer {
        border-radius: 20px;
        border: 2px solid #667eea;
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
# LLM CALL
# ============================================

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

# ============================================
# SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are MozeAI, an AI with REAL-TIME information access and file analysis capabilities.

Created by Mukiibi Moses, a Computer Engineering student at Kyungdong University.
He is an AI builder focused on designing intelligent autonomous agents, language model applications, and practical AI systems that solve real-world problems such as education, automation, and decision support.
He is an active researcher on researchGate, Aademia and other research Platforms.

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
- Provide summaries, answer questions, or extract specific information from webpages
- Use the provided context which includes file content or scraped webpage content
- For time/date questions, use the current information provided
- For news/events, rely on the search results given
- Answer clearly and factually
- If information isn't in context, say "I don't have that information"
- Do not hallucinate or make up dates/events
- Do not show internal reasoning
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
# CURRENT DATE & TIME
# ============================================

def get_current_datetime():
    """Get current date and time"""
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    
    return f"""Current Information:
• Date: {now.strftime('%B %d, %Y')}
• Time: {now.strftime('%I:%M %p')}
• Day: {now.strftime('%A')}
• Timezone: Asia/Seoul"""

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
# WEB SEARCH FUNCTIONS
# ============================================

def internet_search(query):
    """Search the web for current information"""
    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
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
        
    except Exception as e:
        return wikipedia_fallback(query)

def wikipedia_fallback(query):
    """Wikipedia as backup search"""
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
    """Get current news"""
    try:
        api_key = st.secrets.get("NEWS_API_KEY", "")
        
        if not api_key:
            url = f"https://rss2json.com/api.json?rss_url=https://feeds.bbci.co.uk/news/rss.xml"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                items = data.get("items", [])[:3]
                
                news_text = "Latest news headlines:\n\n"
                for item in items:
                    news_text += f"• {item.get('title', '')}\n"
                    news_text += f"  {item.get('description', '')[:150]}...\n\n"
                return news_text[:1000]
        
        else:
            url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&apiKey={api_key}&pageSize=3"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get("articles", [])
                
                news_text = f"Current news about '{topic}':\n\n"
                for article in articles[:3]:
                    news_text += f"• {article.get('title', '')}\n"
                    news_text += f"  {article.get('description', '')[:150]}...\n\n"
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
# WEB SCRAPING FUNCTIONS
# ============================================

def get_site_info_from_search(url):
    """Get information about any website using search engines"""
    try:
        domain = urlparse(url).netloc.replace('www.', '')
        search_url = f"https://www.google.com/search?q={domain}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(search_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            snippets = soup.find_all('div', class_='VwiC3b')
            if snippets:
                return f"Information about {domain} (from web search):\n{snippets[0].get_text()[:500]}"
    except:
        pass
    return None

def scrape_with_requests(url):
    """Try to scrape with different request methods"""
    user_agents = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    ]
    
    for user_agent in user_agents:
        try:
            headers = {
                'User-Agent': user_agent,
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
            }
            
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                
                for element in soup(["script", "style", "nav", "footer", "header", "iframe"]):
                    element.decompose()
                
                text = soup.get_text()
                lines = (line.strip() for line in text.splitlines())
                chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                text = ' '.join(chunk for chunk in chunks if chunk)
                text = ' '.join(text.split())
                
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
    
    content = get_site_info_from_search(url)
    if content:
        return content + "\n\n(Note: Direct access blocked. Showing search results instead.)"
    
    domain = urlparse(url).netloc
    return f"""Unable to directly read this website ({domain}).

Please try:
1. Opening the link directly in your browser
2. Copying and pasting the relevant text here
3. Asking me to search for the information instead"""

def extract_urls_from_query(query):
    """Extract URLs from user query"""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    return re.findall(url_pattern, query)

# ============================================
# ROUTER
# ============================================

def route(query):
    q = query.lower()
    
    urls = extract_urls_from_query(query)
    if urls:
        return "scrape_url"
    
    if any(x in q for x in ["summarize", "analyze this file", "what does the file say", "from the file", "in the document", "based on the file"]):
        return "file_task"
    
    if any(x in q for x in ["+","-","*","/","×","calculate"]):
        return "calculator"
    
    if any(x in q for x in ["time", "date", "today", "current time", "what day"]):
        return "datetime"
    
    if any(x in q for x in ["news", "headlines", "current events", "breaking"]):
        return "news"
    
    if any(x in q for x in ["evaluate", "assess", "grade", "review my", "check my", "score"]):
        return "evaluate"
    
    if any(x in q for x in [
        "capital", "population", "leader", "history", "tell me about",
        "who is", "what is", "when did", "where is", "current"
    ]):
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
    context = context[:1500]
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"""
Context:
{context}

Question:
{question}

Answer clearly.
"""}
    ]
    return clean_answer(llm(messages))

# ============================================
# EVALUATION FUNCTION
# ============================================

def evaluate_work(question, file_content):
    """Evaluate user's work from uploaded files"""
    evaluation_prompt = f"""
You are an expert evaluator. Analyze the following file content and answer the user's evaluation request.

File Content:
{file_content[:3000]}

User Request:
{question}

Please provide:
1. Overall assessment
2. Strengths (2-3 points)
3. Areas for improvement (2-3 points)
4. Score out of 100 (if applicable)
5. Specific recommendations

Be constructive, specific, and actionable.
"""
    
    messages = [
        {"role": "system", "content": "You are an expert evaluator providing constructive feedback."},
        {"role": "user", "content": evaluation_prompt}
    ]
    
    return clean_answer(llm(messages))

# ============================================
# AGENT
# ============================================

def run_agent(query):
    tool = route(query)
    context = ""
    
    # Handle file-related tasks
    if tool == "file_task" and st.session_state.file_context:
        context = f"\n\nUploaded Files Content:\n{st.session_state.file_context}\n"
        context += "\nAnswer the user's question based on these files.\n"
    
    # Handle evaluation tasks
    elif tool == "evaluate" and st.session_state.file_context:
        return evaluate_work(query, st.session_state.file_context)
    
    # Handle URL scraping
    elif tool == "scrape_url":
        urls = extract_urls_from_query(query)
        scraped_content = ""
        for url in urls:
            with st.spinner(f"Reading {url}..."):
                content = scrape_webpage(url)
                if content and not content.startswith("Unable"):
                    scraped_content += f"\n\nContent from {url}:\n{content}\n"
        
        if scraped_content:
            context = scraped_content
        else:
            return "I couldn't read that link. Try asking me to search for the information instead."
    
    # Handle calculator
    if tool == "calculator":
        result = calculator(query)
        if result:
            return result
    
    # Handle date/time
    if tool == "datetime":
        context += get_current_datetime()
    
    # Handle news
    if tool == "news":
        news_context = get_current_news(query)
        if news_context:
            context += news_context
    
    # Handle web search
    if tool == "search":
        web_context = internet_search(query)
        if web_context:
            context += web_context
    
    # Retrieve memory
    mem = retrieve_memory(query)
    if mem:
        context += "\n\n" + mem
    
    # Add default context if needed
    if not context and not st.session_state.file_context:
        context = get_current_datetime()
    
    answer = reason(query, context)
    store_memory(answer)
    
    return answer

# ============================================
# UI
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
    
    st.markdown("---")
    st.markdown("### 📤 File Upload Tips")
    st.markdown("""
    **Supported files:**
    - PDF, DOCX, TXT, CSV, JSON
    
    **Example tasks:**
    - "Summarize this document"
    - "What are the key points?"
    - "Evaluate my essay"
    - "Analyze this data"
    - "Check my work for errors"
    - "Give me a score out of 100"
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Created by **Mukiibi Moses**")
    st.markdown("Computer Engineering @ Kyungdong University")
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("✅ File reading & analysis")
    st.markdown("✅ Document summarization")
    st.markdown("✅ Work evaluation & grading")
    st.markdown("✅ Access to current date/time")
    st.markdown("✅ Real-time web search")
    st.markdown("✅ Latest news headlines")
    st.markdown("✅ Calculator for math problems")
    st.markdown("✅ Memory of past conversations")

# ============================================
# CHAT HISTORY & INPUT WITH STATIONARY BOTTOM BAR
# ============================================

# Create a container for the chat history (scrollable)
chat_container = st.container()

# Create a fixed bottom container for the input
st.markdown("""
<style>
    /* Fixed bottom input bar */
    .fixed-bottom {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: linear-gradient(135deg, #1e1e2f 0%, #2d2d44 100%);
        padding: 15px 20px;
        z-index: 1000;
        border-top: 1px solid rgba(102, 126, 234, 0.3);
    }
    
    /* Chat input wrapper with upload button inside */
    .chat-input-wrapper {
        max-width: 800px;
        margin: 0 auto;
        position: relative;
    }
    
    /* Style the chat input */
    .stChatInputContainer {
        border-radius: 25px;
        border: 2px solid #667eea;
        background: white;
    }
    
    /* Position upload button INSIDE the chat input */
    div[data-testid="stFileUploader"] {
        position: absolute;
        right: 15px;
        top: 50%;
        transform: translateY(-50%);
        z-index: 1001;
        width: auto !important;
    }
    
    /* Hide all file uploader text */
    div[data-testid="stFileUploader"] > div:first-child {
        display: none;
    }
    
    div[data-testid="stFileUploader"] > div:first-child + div {
        display: none;
    }
    
    /* Style the upload button */
    div[data-testid="stFileUploader"] button {
        background: transparent;
        border: none;
        font-size: 22px;
        padding: 0;
        margin: 0;
        width: 36px;
        height: 36px;
        cursor: pointer;
        transition: all 0.3s ease;
        color: #667eea;
    }
    
    div[data-testid="stFileUploader"] button:hover {
        transform: scale(1.1);
        color: #764ba2;
        background: transparent;
    }
    
    /* Add padding to chat input to make room for the button */
    .stChatInputContainer textarea {
        padding-right: 50px !important;
    }
    
    /* Hide Streamlit's default bottom padding */
    .main > div {
        padding-bottom: 100px;
    }
    
    /* Active files indicator */
    .active-files-indicator {
        text-align: center;
        font-size: 12px;
        color: #667eea;
        margin-top: 5px;
        max-width: 800px;
        margin-left: auto;
        margin-right: auto;
    }
</style>
""", unsafe_allow_html=True)

# Display chat history in scrollable container
with chat_container:
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

# Fixed bottom bar with chat input and upload button
with st.container():
    st.markdown('<div class="fixed-bottom">', unsafe_allow_html=True)
    
    # Create the chat input wrapper
    st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)
    
    # Chat input (will be styled with padding for the button)
    query = st.chat_input("Ask anything...", key="main_chat_input")
    
    # Upload button (appears INSIDE the chat input)
    uploaded_file = st.file_uploader(
        "📎",
        type=['pdf', 'docx', 'txt', 'csv', 'json'],
        label_visibility="collapsed",
        key="inline_uploader"
    )
    
    if uploaded_file:
        if uploaded_file.name not in st.session_state.uploaded_files:
            with st.spinner(f"📖 Reading {uploaded_file.name}..."):
                file_content = process_uploaded_file(uploaded_file)
                if file_content and not file_content.startswith("Error"):
                    st.session_state.uploaded_files[uploaded_file.name] = file_content
                    st.session_state.file_context = "\n\n".join([
                        f"=== FILE: {name} ===\n{content}" 
                        for name, content in st.session_state.uploaded_files.items()
                    ])
                    st.success(f"✅ Loaded: {uploaded_file.name}")
                    st.rerun()
                else:
                    st.error(f"❌ Failed to load: {uploaded_file.name}")
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Show active files indicator below the input
    if st.session_state.uploaded_files:
        col1, col2, col3 = st.columns([0.7, 0.2, 0.1])
        with col1:
            file_names = ', '.join(list(st.session_state.uploaded_files.keys())[:2])
            st.markdown(f'<div class="active-files-indicator">📎 {file_names}</div>', unsafe_allow_html=True)
            if len(st.session_state.uploaded_files) > 2:
                st.caption(f"   +{len(st.session_state.uploaded_files) - 2} more")
        with col3:
            if st.button("🗑️", key="clear_files_simple", help="Clear all files"):
                st.session_state.uploaded_files = {}
                st.session_state.file_context = ""
                st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

# Process the query
if query:
    st.session_state.chat_history.append(("user", query))
    response = run_agent(query)
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()
