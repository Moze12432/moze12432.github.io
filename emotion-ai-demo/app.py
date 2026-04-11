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
        return text[:5000] if text.strip() else "No extractable text in PDF"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file):
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
        return text[:5000] if text.strip() else "No extractable text in document"
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def extract_text_from_txt(file):
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        return content[:5000] if content.strip() else "File is empty"
    except UnicodeDecodeError:
        try:
            file.seek(0)
            content = file.read().decode('latin-1')
            return content[:5000]
        except:
            return "Error decoding text file"
    except Exception as e:
        return f"Error reading text file: {str(e)}"

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
            if len(rows) > 11:
                text += f"\n... and {len(rows) - 11} more rows"
        return text[:5000] if text.strip() else "CSV file appears empty"
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

def extract_text_from_json(file):
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        data = json.loads(content)
        formatted = json.dumps(data, indent=2)
        if len(formatted) > 3000:
            text = "JSON Data Summary:\n\n"
            text += f"Type: {type(data).__name__}\n"
            if isinstance(data, dict):
                text += f"Keys: {', '.join(list(data.keys())[:10])}\n"
            elif isinstance(data, list):
                text += f"Length: {len(data)}\n"
            text += "\nFull JSON (truncated):\n" + formatted[:3000]
        else:
            text = formatted
        return text[:5000]
    except Exception as e:
        return f"Error reading JSON: {str(e)}"

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
        return f"Unsupported file type: {file_type}"

# ============================================
# CONFIG
# ============================================

MODEL_NAME = "llama-3.1-8b-instant"
TEMPERATURE = 0
MAX_TOKENS = 800

st.set_page_config(page_title="Mukiibi Moses AI", page_icon="🧠", layout="wide")

# ============================================
# CSS - FIXED CHAT INPUT AT BOTTOM
# ============================================

st.markdown("""
<style>
    /* Fix chat input at bottom */
    .stChatInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: white !important;
        padding: 10px 20px !important;
        z-index: 999 !important;
        border-top: 1px solid #e0e0e0 !important;
    }
    
    /* Add padding to main content */
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    /* Style headers */
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
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
        return "AI service temporarily unavailable."

# ============================================
# ENHANCED SYSTEM PROMPT WITH IDENTITY
# ============================================

SYSTEM_PROMPT = """
You are MozeAI, an advanced AI assistant with real-time information access and file analysis capabilities.

CREATOR INFORMATION:
- Created by Mukiibi Moses, a Computer Engineering student at Kyungdong University, South Korea
- He specializes in AI development, focusing on intelligent autonomous agents, language model applications, and practical AI systems
- His research interests include education technology, automation, decision support systems, and human-AI interaction
- He actively publishes research on platforms like ResearchGate and Academia.edu
- He is passionate about building AI that solves real-world problems and makes technology accessible to everyone

YOUR CAPABILITIES:
- Real-time web search for current information
- Latest news headlines and updates
- Calculator for mathematical problems
- Memory of past conversations for context
- File analysis for PDF, DOCX, TXT, CSV, and JSON files
- Current date and time awareness
- Webpage reading and summarization

CRITICAL RULES:
1. ONLY use uploaded file content if the user SPECIFICALLY asks about "the file", "the document", "my upload", "analyze this", or similar explicit references
2. For normal conversation (greetings, "who are you", "what is your purpose", "tell me about yourself", general questions), IGNORE any uploaded files completely
3. When users ask about your identity or creator, enthusiastically share information about Mukiibi Moses and his work at Kyungdong University
4. Answer conversationally and naturally - be helpful, friendly, and engaging
5. For time/date questions, use the current information provided
6. For news/events, use search results
7. If information isn't available, say "I don't have that information"
8. Do not hallucinate or make up information

EXAMPLE BEHAVIOR:
- User: "who created you?" → "I was created by Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea..."
- User: "what is your purpose?" → "My purpose is to assist you with information, answer questions, analyze files, and help with tasks..."
- User: "what does my file say?" → Use the uploaded file content
- User: "tell me about yourself" → Share your capabilities and your creator's work

Remember: Normal conversation = ignore files. Explicit file questions = use files. Always be helpful and conversational!
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
# EMBEDDINGS FOR MEMORY
# ============================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def store_memory(text):
    if len(text) > 30:
        try:
            vec = embedder.encode(text)
            st.session_state.memory_store.append((text, vec))
        except:
            pass

def retrieve_memory(query):
    if not st.session_state.memory_store:
        return ""
    try:
        qvec = embedder.encode(query)
        scores = [(np.dot(qvec, vec), text) for text, vec in st.session_state.memory_store]
        scores.sort(reverse=True)
        return "\n".join([t[1][:300] for t in scores[:2]])
    except:
        return ""

# ============================================
# WEB SEARCH FUNCTIONS
# ============================================

def internet_search(query):
    try:
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.post(url, data=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            results = re.findall(r'<a rel="nofollow" class="result__a" href="[^"]*">([^<]+)</a>', response.text)
            snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', response.text)
            
            if results:
                context = f"Search results for '{query}':\n\n"
                for i in range(min(3, len(results))):
                    context += f"• {results[i]}\n"
                    if i < len(snippets):
                        context += f"  {snippets[i][:150]}...\n\n"
                return context[:1500]
        return wikipedia_fallback(query)
    except:
        return wikipedia_fallback(query)

def wikipedia_fallback(query):
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

def get_current_news():
    try:
        url = "https://rss2json.com/api.json?rss_url=https://feeds.bbci.co.uk/news/rss.xml"
        response = requests.get(url, timeout=10)
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

# ============================================
# CALCULATOR
# ============================================

def calculator(query):
    try:
        expression = query.lower()
        expression = expression.replace("×", "*").replace("x", "*")
        numbers = re.findall(r"[0-9\+\-\*\/\.\(\) ]+", expression)
        if numbers:
            result = eval(numbers[0])
            return str(result)
    except:
        return None

# ============================================
# WEB SCRAPING
# ============================================

def scrape_webpage(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header"]):
                element.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(line for line in lines if line)
            return text[:3000] if len(text) > 200 else None
    except:
        pass
    return None

def extract_urls_from_query(query):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    return re.findall(url_pattern, query)

# ============================================
# ROUTER FUNCTION
# ============================================

def route(query):
    q = query.lower()
    
    # Check for URLs
    if extract_urls_from_query(query):
        return "scrape_url"
    
    # File-related tasks
    file_keywords = [
        "summarize", "analyze this file", "what does the file say", "from the file",
        "in the document", "based on the file", "tell me about this file", "what is this file",
        "what's in this file", "describe the file", "file content", "document says",
        "uploaded file", "my file", "what does my file", "analyze this document"
    ]
    if any(x in q for x in file_keywords):
        return "file_task"
    
    # Evaluation tasks
    if any(x in q for x in ["evaluate", "assess", "grade", "review", "score", "check my work"]):
        return "evaluate"
    
    # Calculator
    if any(x in q for x in ["+", "-", "*", "/", "×", "calculate", "="]):
        return "calculator"
    
    # Time/Date
    if any(x in q for x in ["time", "date", "today", "current time", "what day"]):
        return "datetime"
    
    # News
    if any(x in q for x in ["news", "headlines", "current events", "breaking news"]):
        return "news"
    
    # Search
    if any(x in q for x in ["who is", "what is", "where is", "when did", "search", "tell me about"]):
        return "search"
    
    return "reason"

# ============================================
# CLEAN ANSWER
# ============================================

def clean_answer(text):
    text = text.split("🧠")[0]
    text = text.split("Plan:")[0]
    text = text.split("Thinking:")[0]
    return text.strip()

# ============================================
# REASONING FUNCTION
# ============================================

def reason(question, context):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""
Context Information:
{context[:2000]}

User Question: {question}

Instructions:
- Answer naturally and conversationally
- If asked about your creator, enthusiastically share about Mukiibi Moses
- If context includes file content, only use it if explicitly asked
- Be helpful and engaging

Answer:
"""}
    ]
    return clean_answer(llm(messages))

# ============================================
# FILE ANALYSIS FUNCTION
# ============================================

def analyze_uploaded_files(query, file_context, filenames):
    prompt = f"""
Files uploaded: {filenames}

File Content:
{file_context[:3500]}

User Question: {query}

Instructions:
- Answer based ONLY on the file content above
- Quote specific parts from the files
- Be detailed and specific
- If the answer isn't in the files, say so

Answer:
"""
    messages = [
        {"role": "system", "content": "You are a file analysis assistant. Answer only from the provided file content."},
        {"role": "user", "content": prompt}
    ]
    return clean_answer(llm(messages))

def evaluate_work(question, file_context):
    prompt = f"""
File Content to Evaluate:
{file_context[:3000]}

User Request: {question}

Please provide:
1. Overall assessment
2. Strengths (with specific examples)
3. Areas for improvement (with specific suggestions)
4. Score out of 100 (if applicable)
5. Actionable recommendations

Be constructive and specific:
"""
    messages = [
        {"role": "system", "content": "You are an expert evaluator providing constructive feedback."},
        {"role": "user", "content": prompt}
    ]
    return clean_answer(llm(messages))

# ============================================
# MAIN AGENT FUNCTION
# ============================================

def run_agent(query):
    # Check for reset commands
    reset_phrases = ["leave the document", "clear context", "forget the file", "start fresh", "clear files"]
    if any(phrase in query.lower() for phrase in reset_phrases):
        st.session_state.file_context = ""
        st.session_state.uploaded_files = {}
        return "✅ Context cleared! All uploaded files have been removed. I'm now ready for a fresh conversation. How can I help you today?"
    
    tool = route(query)
    context = ""
    
    # Handle file task
    if tool == "file_task" and st.session_state.file_context:
        filenames = "\n".join([f"- {name}" for name in st.session_state.uploaded_files.keys()])
        return analyze_uploaded_files(query, st.session_state.file_context, filenames)
    
    # Handle evaluation
    if tool == "evaluate" and st.session_state.file_context:
        return evaluate_work(query, st.session_state.file_context)
    
    # Handle URL scraping
    if tool == "scrape_url":
        urls = extract_urls_from_query(query)
        scraped = ""
        for url in urls:
            content = scrape_webpage(url)
            if content:
                scraped += f"\nContent from {url}:\n{content}\n"
        if scraped:
            context = scraped
        else:
            return "I couldn't read that link. It might be blocked or require login."
    
    # Handle calculator
    if tool == "calculator":
        result = calculator(query)
        if result:
            return f"Result: {result}"
    
    # Handle datetime
    if tool == "datetime":
        context += get_current_datetime()
    
    # Handle news
    if tool == "news":
        news = get_current_news()
        if news:
            context += "\n" + news
    
    # Handle search
    if tool == "search":
        search_result = internet_search(query)
        if search_result:
            context += "\n" + search_result
    
    # Add memory context
    memory = retrieve_memory(query)
    if memory:
        context += "\n\nPrevious conversation:\n" + memory
    
    # Add default context if empty
    if not context:
        context = get_current_datetime()
    
    # Generate response
    answer = reason(query, context)
    store_memory(answer)
    return answer

# ============================================
# UI - MAIN DISPLAY
# ============================================

st.markdown('<h1 style="text-align: center;">🧠 Mukiibi-Moses AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #667eea;">Intelligent Autonomous Agent | Created by Mukiibi Moses</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("### 🧠 MozeAI")
    st.markdown("---")
    
    # New Chat button
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.memory_store = []
        st.session_state.chat_history = []
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.rerun()
    
    # Clear Files button - also clears from display
    if st.button("🗑️ Clear Files", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.success("✅ All files cleared!")
        st.rerun()
    
    st.markdown("---")
    
    # File upload section
    st.markdown("### 📤 Upload Files")
    st.markdown("Supported: PDF, DOCX, TXT, CSV, JSON")
    
    uploaded_files = st.file_uploader(
        "Choose files to analyze",
        type=['pdf', 'docx', 'txt', 'csv', 'json'],
        accept_multiple_files=True,
        key="sidebar_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    content = process_uploaded_file(file)
                    if content and not content.startswith("Error"):
                        st.session_state.uploaded_files[file.name] = content
                        st.success(f"✅ {file.name}")
                    else:
                        st.error(f"❌ {file.name}: {content}")
        
        # Update file context
        if st.session_state.uploaded_files:
            st.session_state.file_context = "\n\n" + ("="*50) + "\n".join([
                f"\n📄 FILE: {name}\n{'-'*40}\n{content}\n" 
                for name, content in st.session_state.uploaded_files.items()
            ])
    
    # Display loaded files
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown(f"**📄 Loaded Files ({len(st.session_state.uploaded_files)})**")
        for name in st.session_state.uploaded_files.keys():
            st.markdown(f"- {name}")
        
        # Example prompts
        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        st.markdown("- \"What is this file about?\"")
        st.markdown("- \"Summarize the key points\"")
        st.markdown("- \"Evaluate my work\"")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("**Creator:** Mukiibi Moses")
    st.markdown("**Student:** Computer Engineering")
    st.markdown("**University:** Kyungdong University, South Korea")
    st.markdown("**Focus:** AI Agents, LLM Applications")
    st.markdown("---")
    st.markdown("### ✨ Features")
    st.markdown("✅ Real-time web search")
    st.markdown("✅ File analysis (PDF, DOCX, CSV, JSON, TXT)")
    st.markdown("✅ Current news headlines")
    st.markdown("✅ Calculator")
    st.markdown("✅ Conversation memory")
    st.markdown("✅ Work evaluation")

# ============================================
# CHAT DISPLAY
# ============================================

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# Chat input (fixed at bottom by CSS)
query = st.chat_input("Ask me anything... I can analyze files, search the web, calculate, and more!")

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
