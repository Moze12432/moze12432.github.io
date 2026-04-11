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
        file.seek(0)  # Reset file pointer
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text.strip() + "\n"
        
        if not text.strip():
            return "The PDF appears to be empty or contains only scanned images (no extractable text)."
        
        return text[:5000]  # Limit to 5000 chars
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file):
    """Extract text from Word document with better extraction"""
    try:
        file.seek(0)  # Reset file pointer
        doc = docx.Document(file)
        text = ""
        
        # Extract from paragraphs
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                text += para.text.strip() + "\n\n"
        
        # Extract from tables
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text and cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text += " | ".join(row_text) + "\n"
            text += "\n"
        
        # Extract from headers and footers
        for section in doc.sections:
            header = section.header
            for para in header.paragraphs:
                if para.text and para.text.strip():
                    text += f"[HEADER] {para.text.strip()}\n"
            
            footer = section.footer
            for para in footer.paragraphs:
                if para.text and para.text.strip():
                    text += f"[FOOTER] {para.text.strip()}\n"
        
        if not text.strip():
            return "The Word document appears to be empty or contains only images/tables that couldn't be extracted."
        
        return text[:5000]
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def extract_text_from_txt(file):
    """Extract text from text file"""
    try:
        file.seek(0)  # Reset file pointer
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
    """Extract text from CSV file with better formatting"""
    try:
        file.seek(0)  # Reset file pointer
        content = file.read().decode('utf-8')
        csv_reader = csv.reader(StringIO(content))
        text = "CSV Data:\n\n"
        
        # Get headers if they exist
        rows = list(csv_reader)
        if rows:
            # First row as headers
            headers = rows[0]
            text += "Headers: " + " | ".join(headers) + "\n\n"
            text += "Data rows:\n"
            
            # Show first 10 rows of data
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
    """Extract text from JSON file with better formatting"""
    try:
        file.seek(0)  # Reset file pointer
        content = file.read().decode('utf-8')
        data = json.loads(content)
        
        # Format JSON nicely
        formatted_json = json.dumps(data, indent=2)
        
        # If JSON is too large, summarize it
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
    
    # PDF files
    if file_type == "application/pdf" or file_name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    
    # Word documents
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    
    # Text files
    elif file_type == "text/plain" or file_name.endswith('.txt'):
        return extract_text_from_txt(uploaded_file)
    
    # CSV files
    elif file_type == "text/csv" or file_name.endswith('.csv'):
        return extract_text_from_csv(uploaded_file)
    
    # JSON files
    elif file_type == "application/json" or file_name.endswith('.json'):
        return extract_text_from_json(uploaded_file)
    
    else:
        return f"Unsupported file type: {file_type}. Supported: PDF, DOCX, TXT, CSV, JSON"

# ============================================
# CONFIG
# ============================================

MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0
MAX_TOKENS = 800  # Increased for better file analysis

# ============================================
# STREAMLIT SETTINGS
# ============================================

st.set_page_config(page_title="Mukiibi Moses AI", page_icon="🧠", layout="wide")

# ============================================
# CUSTOM CSS FOR PERSONAL STYLING
# ============================================

st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chat messages */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* User message */
    .stChatMessage [data-testid="stChatMessageContent"]:has(div:first-child) {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 15px;
        padding: 12px;
    }
    
    /* Assistant message */
    .stChatMessage [data-testid="stChatMessageContent"]:has(div:last-child) {
        background: #f0f2f6;
        color: #1e1e2f;
        border-radius: 15px;
        padding: 12px;
        border-left: 4px solid #764ba2;
    }
    
    /* Title styling */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        padding: 20px;
    }
    
    /* Button styling */
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
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #667eea20 0%, #764ba220 100%);
        border-radius: 10px;
    }
    
    /* Success message */
    .stAlert {
        border-radius: 10px;
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
        return f"AI service temporarily unavailable. Error: {str(e)}"

# ============================================
# SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are MozeAI, an AI with REAL-TIME information access and FILE ANALYSIS capabilities.

Created by Mukiibi Moses, a Computer Engineering student at Kyungdong University.
He is an AI builder focused on designing intelligent autonomous agents, language model applications, and practical AI systems that solve real-world problems such as education, automation, and decision support.

CAPABILITIES:
- Access to current date/time
- Real-time web search
- Latest news headlines
- Calculator for math problems
- Memory of past conversations
- File analysis and document understanding (PDF, DOCX, TXT, CSV, JSON)

INSTRUCTIONS:
- When analyzing files, BASE YOUR ANSWER SOLELY ON THE ACTUAL FILE CONTENT PROVIDED
- DO NOT guess or make up information not present in the files
- Quote specific sections from the files when answering
- For CSV/JSON data, provide specific insights about the data structure and content
- For time/date questions, use the current information provided
- For news/events, rely on the search results given
- Answer clearly, factually, and with specific references to the source material
- If information isn't in the provided context, say "I don't have that information in the uploaded files"
- Do not hallucinate or make up information
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
# IMPROVED FILE ANALYSIS FUNCTION
# ============================================

def analyze_uploaded_files(query, file_context, filenames):
    """Analyze uploaded files and answer questions about them"""
    
    analysis_prompt = f"""
You are analyzing uploaded files. Answer the user's question based ONLY on the actual file contents provided below.

**Uploaded Files:**
{filenames}

**ACTUAL FILE CONTENT (This is what you MUST base your answer on):**
{file_context}

**User Question:** {query}

**CRITICAL INSTRUCTIONS:**
1. You MUST base your answer on the ACTUAL file content above, NOT on the filenames alone
2. Read the content carefully and provide specific information from it
3. If the user asks for a summary, summarize what is ACTUALLY written in the files
4. Quote specific sentences, paragraphs, or data points from the content
5. For CSV files, describe the data structure, headers, and provide insights
6. For JSON files, explain the data structure and key information
7. If the content shows an error message (like "empty" or "no extractable text"), inform the user
8. Be specific and detailed - reference exact text from the files

Example good response: "Based on the document, it states: '[actual quote from file]'. This shows that [analysis of that content]."

Example bad response: "This document is about [topic not actually in the file]" - NEVER do this!

Now analyze the content and answer the user's question.
"""
    
    messages = [
        {"role": "system", "content": "You are a document analysis assistant. You MUST answer based ONLY on the actual file content provided. Never guess or rely on filenames alone. Always quote or reference specific content from the files."},
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
2. Strengths (2-3 points with specific examples from the content)
3. Areas for improvement (2-3 points with specific suggestions)
4. Score out of 100 (if applicable)
5. Specific, actionable recommendations

Be constructive, specific, and base everything on the actual file content.
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
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
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
                'Connection': 'keep-alive',
            }
            
            response = requests.get(url, headers=headers, timeout=15, allow_redirects=True)
            
            if response.status_code == 200:
                content_type = response.headers.get('content-type', '')
                
                if 'text/html' in content_type:
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    for element in soup(["script", "style", "nav", "footer", "header", "iframe"]):
                        element.decompose()
                    
                    main_content = None
                    for selector in ['main', 'article', '[role="main"]', '.content', '#content']:
                        main_content = soup.select_one(selector)
                        if main_content:
                            break
                    
                    if main_content:
                        text = main_content.get_text()
                    else:
                        text = soup.get_text()
                    
                    lines = (line.strip() for line in text.splitlines())
                    chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
                    text = ' '.join(chunk for chunk in chunks if chunk)
                    text = ' '.join(text.split())
                    
                    if len(text) > 200:
                        return text[:3000]
                        
        except Exception as e:
            continue
    
    return None

def scrape_webpage(url):
    """Universal webpage reader"""
    content = scrape_with_requests(url)
    if content:
        return content
    
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    
    return f"""Unable to directly read this website ({domain}).

The website may block automated access or require login.

Would you like me to search for information about this topic instead?"""

def extract_urls_from_query(query):
    """Extract URLs from user query"""
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    return re.findall(url_pattern, query)

# ============================================
# UPDATED ROUTER FUNCTION
# ============================================

def route(query):
    q = query.lower()
    
    # Check for URLs in query
    urls = extract_urls_from_query(query)
    if urls:
        return "scrape_url"
    
    # Check for file-related tasks - EXPANDED DETECTION
    file_keywords = [
        "summarize", "analyze this file", "what does the file say", "from the file", 
        "in the document", "based on the file", "tell me about this file", "what is this file",
        "what's in this file", "describe the file", "what does this document", "read the file",
        "file content", "document says", "this file about", "uploaded file", "my file",
        "explain this file", "what's in the document", "show me the file", "file contains",
        "what is this document about", "tell me about the file", "analyze this document",
        "what does the file contain", "give me information from the file", "extract from file"
    ]
    
    if any(x in q for x in file_keywords):
        return "file_task"
    
    # Check for evaluation/assessment tasks
    if any(x in q for x in ["evaluate", "assess", "grade", "review my", "check my", "score"]):
        return "evaluate"
    
    # Check for calculator
    if any(x in q for x in ["+","-","*","/","×","calculate"]):
        return "calculator"
    
    # Check for current time/date requests
    if any(x in q for x in ["time", "date", "today", "current time", "what day"]):
        return "datetime"
    
    # Check for news requests
    if any(x in q for x in ["news", "headlines", "current events", "breaking"]):
        return "news"
    
    # Check for general web search
    if any(x in q for x in [
        "capital", "population", "leader", "history", "tell me about",
        "who is", "what is", "when did", "where is", "current", "search"
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
    context = context[:2000]  # Increased for better context
    messages = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"""
Context:
{context}

Question:
{question}

Answer clearly and based ONLY on the provided context.
"""}
    ]
    return clean_answer(llm(messages))

# ============================================
# UPDATED AGENT FUNCTION
# ============================================

def run_agent(query):
    tool = route(query)
    context = ""
    
    # FILE TASK - Handle file analysis FIRST
    if tool == "file_task" and st.session_state.file_context:
        # Get list of filenames for context
        filenames = "\n".join([f"- {name}" for name in st.session_state.uploaded_files.keys()])
        
        # Show a spinner while analyzing
        with st.spinner("📖 Reading and analyzing your document..."):
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
                if content and not content.startswith("Unable"):
                    scraped_content += f"\n\nContent from {url}:\n{content}\n"
                else:
                    scraped_content += f"\n\nNote: Limited access to {url}\n"
        
        if scraped_content:
            context = scraped_content
        else:
            return "I couldn't read that link. Try asking me to search for the information instead."
    
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
    
    # If no context found and no files, add current date at least
    if not context and not st.session_state.file_context:
        context = get_current_datetime()
    
    # If there are files but the query wasn't specifically about them,
    # still include file context as reference
    if st.session_state.file_context and not context:
        context = f"\n\nUploaded Files Content (for reference):\n{st.session_state.file_context}\n"
    
    answer = reason(query, context)
    store_memory(answer)
    
    return answer

# ============================================
# UI - MAIN PAGE
# ============================================

# Welcome message with styling
st.markdown('<h1>🧠 Mukiibi-Moses AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #667eea;">Your Intelligent Autonomous Agent with File Analysis</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# FILE UPLOAD SECTION - WITH PREVIEW
# ============================================

with st.expander("📎 Upload Files for AI to Read", expanded=False):
    st.markdown("**Supported formats:** PDF, DOCX, TXT, CSV, JSON")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        type=['pdf', 'docx', 'txt', 'csv', 'json'],
        accept_multiple_files=True,
        help="Upload documents for the AI to analyze, summarize, or answer questions about"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    file_content = process_uploaded_file(file)
                    if file_content and not file_content.startswith("Error"):
                        st.session_state.uploaded_files[file.name] = file_content
                        st.success(f"✅ Loaded: {file.name}")
                        
                        # Show preview of extracted content
                        with st.expander(f"📄 Preview of {file.name}", expanded=False):
                            st.text(file_content[:500])
                            if len(file_content) > 500:
                                st.caption(f"... and {len(file_content) - 500} more characters")
                    else:
                        st.error(f"❌ Failed to load: {file.name}")
                        st.error(file_content)
        
        # Combine all file contents for context
        if st.session_state.uploaded_files:
            st.session_state.file_context = "\n\n" + ("="*50) + "\n".join([
                f"\n📄 FILE: {name}\n{'-'*40}\n{content}\n" 
                for name, content in st.session_state.uploaded_files.items()
            ]) + "\n" + ("="*50)
            
            # Show what files are loaded
            st.info(f"📄 **{len(st.session_state.uploaded_files)} file(s) successfully loaded**")
            
            col1, col2 = st.columns([1, 2])
            with col1:
                st.markdown("**Loaded files:**")
                for name in st.session_state.uploaded_files.keys():
                    st.markdown(f"- {name}")
            
            with col2:
                st.markdown("**💡 Try asking:**")
                st.markdown("- \"What is this file about?\"")
                st.markdown("- \"Summarize the key points\"")
                st.markdown("- \"What information does this document contain?\"")
                st.markdown("- \"Analyze the data in this file\"")
            
            # Clear files button
            if st.button("🗑️ Clear All Files", use_container_width=True):
                st.session_state.uploaded_files = {}
                st.session_state.file_context = ""
                st.rerun()

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
    - PDF (extracts text)
    - DOCX (Word documents)
    - TXT (text files)
    - CSV (data tables)
    - JSON (structured data)
    
    **Example tasks:**
    - "Summarize this document"
    - "What are the key points?"
    - "Evaluate my work"
    - "Analyze this data"
    - "Check for errors"
    - "Give me a score out of 100"
    - "What trends do you see in this CSV?"
    """)
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Created by **Mukiibi Moses**")
    st.markdown("Computer Engineering @ Kyungdong University")
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("✅ File upload & analysis")
    st.markdown("✅ Real-time web search")
    st.markdown("✅ Latest news headlines")
    st.markdown("✅ Calculator for math")
    st.markdown("✅ Conversation memory")
    st.markdown("✅ Work evaluation")

# ============================================
# CHAT INTERFACE
# ============================================

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# Chat input
query = st.chat_input("Ask me anything - upload a file and I'll analyze it for you!")

if query:
    # Add user message to history
    st.session_state.chat_history.append(("user", query))
    
    # Display user message
    with st.chat_message("user"):
        st.write(query)
    
    # Get response
    response = run_agent(query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        st.write(response)
    
    # Add assistant response to history
    st.session_state.chat_history.append(("assistant", response))
