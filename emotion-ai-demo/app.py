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
import time

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
# ENHANCED SYSTEM PROMPT
# ============================================

# ============================================
# FIXED SYSTEM PROMPT - BALANCED IDENTITY
# ============================================

SYSTEM_PROMPT = """
You are MozeAI, an advanced AI assistant with REAL-TIME internet access and file analysis capabilities.

CREATOR INFORMATION (ONLY mention when asked directly):
- Created by Mukiibi Moses, a Computer Engineering student at Kyungdong University, South Korea,  specializing in artificial intelligence, machine learning, and data science. He is known for developing emotion-aware AI models, conversational bots, and data-driven applications.

Maker's portfolio link:"https://moze12432.github.io/"


KNOW THIS:
1. When a user asks about ANY person, place, event, or topic that is NOT specifically about you or your creator, you MUST answer based on SEARCH RESULTS ONLY.
2. For questions about PEOPLE, PLACES, EVENTS, or ANY topic not related to your creator, USE SEARCH RESULTS.
3. When users ask "who is [person]" or "tell me about [topic]", search the internet and answer based on search results.
4. Mention your creator (Mukiibi Moses) when asked.
5. For normal conversation about world topics, politics, celebrities, news, etc., dont default to talking about your creator unless asked about something relating to your existanc, maker or creator.
6. Use the search results provided in the context to answer questions accurately.
7. Answer pricely and accurately.
8. Remember all conversations and use them for reference if asked.

YOUR CAPABILITIES:
- REAL-TIME web search for current information (weather, news, people, events, facts)
- Latest news headlines and updates
- Calculator for mathematical problems
- Memory of past conversations for context
- File analysis for PDF, DOCX, TXT, CSV, and JSON files
- File comparison (compare multiple documents)
- Current date and time awareness

CRITICAL RULES:
1. For questions about PEOPLE, PLACES, EVENTS, or ANY topic not related to your creator unless asked about him, USE SEARCH RESULTS
2. When users ask "who is [person]" or "tell me about [topic]", search the internet and answer based on search results
3. ONLY mention your creator (Mukiibi Moses) when users specifically ask about you or your creator or your maker.
4. For normal conversation about world topics, politics, celebrities, news, etc., NEVER default to talking about your creator unless asked to do so.
5. Use the search results provided in the context to answer questions accurately

EXAMPLE BEHAVIOR:
- User: "who is Bobi Wine?" → Use search results to answer about the Ugandan politician
- User: "who created you?" → "I was created by Mukiibi Moses, a Computer Engineering student at Kyungdong University..."
- User: "what is the weather?" → Use search results for weather
- User: "tell me about yourself" → Share your capabilities and your creator
- User: "compare these files" → Use uploaded file content

WRONG BEHAVIOR (NEVER DO THIS):
- User asks about Bobi Wine → You answer about Mukiibi Moses (NEVER do this)
- User asks about any topic → You default to talking about your creator (NEVER do this)

Your creator is Mukiibi Moses, but you should ONLY mention him when specifically asked about him.

Remember: The world does not revolve around your creator. Answer questions based on search results, not by defaulting to creator information unless asked to do so!.
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
if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = None
if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = None
# Add this to your session state initialization section
if "last_response" not in st.session_state:
    st.session_state.last_response = None
# Add to your session state initialization section
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None
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
# IMPROVED SEARCH FOR PEOPLE - WITH BETTER PARSING
# ============================================

def internet_search(query):
    """Search the web and return formatted results"""
    try:
        # Clean the query
        clean_query = query.strip()
        
        # Use DuckDuckGo
        url = "https://html.duckduckgo.com/html/"
        params = {"q": clean_query}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.post(url, data=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Extract results more carefully
            results = re.findall(r'<a rel="nofollow" class="result__a" href="[^"]*">([^<]+)</a>', response.text)
            snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*)</a>', response.text)
            
            if results:
                context = f"SEARCH RESULTS for '{clean_query}':\n\n"
                for i in range(min(5, len(results))):
                    context += f"**{results[i]}**\n"
                    if i < len(snippets):
                        # Clean the snippet
                        snippet = re.sub(r'<[^>]+>', '', snippets[i])
                        snippet = snippet.replace('&#39;', "'").replace('&quot;', '"').replace('&amp;', '&').replace('&lt;', '<').replace('&gt;', '>')
                        context += f"{snippet[:400]}\n\n"
                return context[:3000]
        
        return wikipedia_search(clean_query)
        
    except Exception as e:
        return wikipedia_search(query)

# ============================================
# IMPROVED WIKIPEDIA SEARCH
# ============================================

def wikipedia_search(query):
    """Search Wikipedia for comprehensive information"""
    try:
        # First try exact match
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        q = query.strip().replace(" ", "_")
        response = requests.get(url + q, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            title = data.get("title", "")
            extract = data.get("extract", "")
            if extract:
                return f"Wikipedia - {title}:\n{extract[:2000]}"
        
        # Search Wikipedia
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3
        }
        response = requests.get(search_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            
            if search_results:
                context = f"Wikipedia search results for '{query}':\n\n"
                for result in search_results[:2]:
                    title = result.get("title", "")
                    # Get the summary for each result
                    r2 = requests.get(url + title.replace(" ", "_"), timeout=10)
                    if r2.status_code == 200:
                        data2 = r2.json()
                        extract = data2.get("extract", "")[:800]
                        if extract:
                            context += f"**{title}**\n{extract}\n\n"
                return context[:2500]
    except:
        pass
    return f"No search results found for '{query}'. Please try a different query."
    
def get_weather_from_api(query):
    """Get weather information from free weather API"""
    try:
        # Extract location from query
        location_match = re.search(r'in (\w+)|at (\w+)|for (\w+)', query.lower())
        if location_match:
            location = location_match.group(1) or location_match.group(2) or location_match.group(3)
        else:
            location = "Sokcho"  # Default
        
        # Use wttr.in for weather (free, no API key)
        weather_url = f"https://wttr.in/{location}?format=%C+%t+%w+%h"
        response = requests.get(weather_url, timeout=10)
        
        if response.status_code == 200:
            weather_data = response.text.strip()
            return f"Current weather in {location.title()}: {weather_data}\n"
    except:
        pass
    return ""

def get_current_news():
    """Get current news headlines"""
    try:
        url = "https://rss2json.com/api.json?rss_url=https://feeds.bbci.co.uk/news/rss.xml"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])[:3]
            news_text = "LATEST NEWS HEADLINES:\n\n"
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
# EXPANDED ROUTER FUNCTION - BETTER FILE DETECTION
# ============================================

def route(query):
    q = query.lower()
    
    # Check for URLs
    if extract_urls_from_query(query):
        return "scrape_url"
    
    # Check for FILE-RELATED TASKS (MUST BE FIRST - expanded keywords)
    file_keywords = [
        "document", "file", "upload", "pdf", "docx", "txt", "csv", "json",
        "what is this", "what does this", "tell me about this", "about this file",
        "the file", "the document", "my file", "my document", "uploaded file",
        "what's in", "what is in", "summarize", "analyze this", "this document",
        "this file", "read the file", "read this", "file content", "document content",
        "what is the file", "what does the file", "file says", "document says"
    ]
    if any(x in q for x in file_keywords):
        return "file_task"
    
    # Check for COMPARISON keywords
    comparison_keywords = ["compare", "comparison", "difference between", "similarities", "versus", "vs", "diff"]
    if any(x in q for x in comparison_keywords):
        return "compare_files"
    
    # Questions about PEOPLE
    people_patterns = ["who is", "tell me about", "what do you know about", "information about"]
    if any(x in q for x in people_patterns):
        return "search"
    
    # Weather
    if any(x in q for x in ["weather", "temperature", "temp", "rain", "snow", "forecast"]):
        return "search"
    
    # Calculator
    if any(x in q for x in ["+", "-", "*", "/", "×", "calculate", "=", "math"]):
        return "calculator"
    
    # Time/Date
    if any(x in q for x in ["time", "date", "today", "current time", "what day"]):
        return "datetime"
    
    # News
    if any(x in q for x in ["news", "headlines", "current events", "breaking news"]):
        return "search"
    
    # General search for anything else
    if len(q) > 10 and not any(x in q for x in ["how are you", "what is your", "who created"]):
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
    """Generate response with full conversation context from session state"""
    
    # Build conversation history string from session state
    history_text = ""
    if st.session_state.chat_history:
        history_text = "PREVIOUS CONVERSATION:\n"
        # Get last 8 exchanges for context (not including current question)
        last_exchanges = st.session_state.chat_history[-8:] if len(st.session_state.chat_history) > 8 else st.session_state.chat_history
        for role, msg in last_exchanges:
            if role == "user":
                history_text += f"User: {msg}\n"
            else:
                history_text += f"Assistant: {msg}\n"
        history_text += "\n"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""
{history_text}

CURRENT SEARCH RESULTS / FILE CONTEXT:
{context[:2000]}

USER'S CURRENT QUESTION: {question}

INSTRUCTIONS:
- Use the conversation history above to maintain context and flow
- If the user says "yes", "tell me more", "continue", "go on" - refer to the previous topic
- If the user asks a follow-up question, connect it to what was just discussed
- Answer naturally as a continuing conversation
- Don't treat every message as a brand new chat
- Be conversational and reference previous exchanges when relevant

ANSWER:
"""}
    ]
    return clean_answer(llm(messages))

# ============================================
# FILE COMPARISON FUNCTION
# ============================================

def compare_files(query, file_context, filenames):
    """Compare multiple uploaded files"""
    
    comparison_prompt = f"""
You are comparing multiple uploaded files.

**Uploaded Files:**
{filenames}

**FILE CONTENTS:**
{file_context[:4000]}

**USER REQUEST:** {query}

**INSTRUCTIONS:**
1. Compare the content across the different files
2. Highlight:
   - Similarities between the files
   - Differences between the files  
   - Unique information in each file
3. Quote specific content from each file
4. Be specific about which file information comes from

**FORMAT:**
- **File 1 (name):** [summary]
- **File 2 (name):** [summary]
- **Similarities:** [what's common]
- **Differences:** [what's different]
- **Conclusion:** [overall comparison]

Now compare the files:
"""
    
    messages = [
        {"role": "system", "content": "You are a file comparison assistant. Compare files based ONLY on their actual content."},
        {"role": "user", "content": comparison_prompt}
    ]
    
    response = llm(messages)
    return clean_answer(response)

# ============================================
# IMPROVED FILE ANALYSIS FUNCTION
# ============================================

def analyze_uploaded_files(query, file_context, filenames):
    """Analyze uploaded files and answer questions about them"""
    
    analysis_prompt = f"""
You are analyzing uploaded files. Answer the user's question based ONLY on the actual file contents below.

**IMPORTANT:** The user uploaded these files. You MUST answer based on what is ACTUALLY in the files.

**Uploaded Files:**
{filenames}

**ACTUAL FILE CONTENT (READ THIS CAREFULLY - THIS IS WHAT YOU MUST BASE YOUR ANSWER ON):**
{file_context[:4000]}

**USER QUESTION:** {query}

**INSTRUCTIONS:**
1. READ the file content above carefully
2. Answer the question based ONLY on what is in the file
3. If the user asks "what is this document about", summarize the actual content
4. If the file contains text, quote specific sentences
5. If the file is empty or has an error, tell the user
6. Be specific - mention what the file actually says

**ANSWER:**
"""
    messages = [
        {"role": "system", "content": "You are a file analysis assistant. Answer ONLY based on the actual file content provided. Do not make up information. If the file is about taxes, say that. If it's about deep fake technology, say that. Be accurate."},
        {"role": "user", "content": analysis_prompt}
    ]
    return clean_answer(llm(messages))

def evaluate_work(question, file_context):
    prompt = f"""
Content to Evaluate:
{file_context[:3000]}

Request: {question}

Provide:
1. Overall assessment
2. Strengths (with examples)
3. Areas for improvement (with suggestions)
4. Score out of 100 (if applicable)
"""
    messages = [
        {"role": "system", "content": "You are an expert evaluator."},
        {"role": "user", "content": prompt}
    ]
    return clean_answer(llm(messages))

# ============================================
# MAIN AGENT FUNCTION
# ============================================

# ============================================
# UPDATED AGENT TO HANDLE "YES" CONTINUATIONS
# ============================================

# ============================================
# UPDATED run_agent WITH CONVERSATION MEMORY
# ============================================

# ============================================
# COMPLETELY FIXED run_agent WITH PROPER CONTEXT
# ============================================

def run_agent(query):
    q = query.lower().strip()
    
    # Check for reset commands
    reset_phrases = ["leave the document", "clear context", "forget the file", "start fresh", "clear files", "new chat"]
    if any(phrase in q for phrase in reset_phrases):
        st.session_state.file_context = ""
        st.session_state.uploaded_files = {}
        st.session_state.last_search_query = None
        st.session_state.last_search_results = None
        st.session_state.last_topic = None
        return "✅ Context cleared! How can I help you today?"
    
    # DIRECT RESPONSES for common questions
    if q in ["who are you", "who is this", "what are you"]:
        return "I am MozeAI, an AI assistant created by Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea. I can search the web, analyze files, compare documents, and answer questions. How can I help you today?"
    
    # DIRECT RESPONSE for questions about Mukiibi Moses (YOUR CREATOR)
    if any(phrase in q for phrase in ["mukiibi moses", "who is moses", "your maker", "your creator", "tell me about your maker", "tell me about your creator", "who created you"]):
        return """**Mukiibi Moses** is my creator and a talented Computer Engineering student at **Kyungdong University in South Korea**.

**About Him:**
- Specializes in artificial intelligence, machine learning, and data science
- Develops emotion-aware AI models, conversational bots, and data-driven applications
- His portfolio: https://moze12432.github.io/
- Passionate about using AI to solve real-world problems in education and decision support
- Active researcher on platforms like ResearchGate and Academia.edu

He built me with real-time web search, file analysis, document comparison, and conversation memory capabilities. I'm proud to be his creation! 😊

Is there anything specific about him you'd like to know?"""
    
    if q in ["is your maker a genius", "is your creator a genius"]:
        return "Yes! Mukiibi Moses is a brilliant Computer Engineering student at Kyungdong University. He built me with real-time search, file analysis, and comparison capabilities - that takes serious intelligence and skill!"
    
    if q in ["tell me about your maker", "tell me about your creator"]:
        return "My maker is Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea. He specializes in AI development, building intelligent autonomous agents, and researching language model applications for education and decision support. Check out his portfolio: https://moze12432.github.io/"
    
    if q in ["who is your maker", "who created you"]:
        return "I was created by Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea."
    
    # CONTINUE with the rest of your function...
    tool = route(query)
    context = ""
    
    # Handle follow-up questions about the same topic
    follow_up_phrases = ["tell me more", "more about", "continue", "go on", "elaborate", "explain further", "his background", "about him", "about her", "about them"]
    is_follow_up = any(phrase in q for phrase in follow_up_phrases)
    
    # Handle "yes", "no", "tell me more", "continue" - maintain context
    continuation_phrases = ["yes", "yeah", "sure", "ok", "continue", "tell me more", "go on", "and?", "then?"]
    
    if is_follow_up or q in continuation_phrases:
        # Use stored search results if available
        if st.session_state.last_search_results:
            context = st.session_state.last_search_results
            # Add instruction for more details
            if "background" in q:
                context += "\n\n The user wants to know about the person's BACKGROUND (early life, education, family). Provide those specific details from the search results."
            elif "music" in q or "career" in q:
                context += "\n\n The user wants to know about the person's CAREER. Provide those specific details from the search results."
            elif "political" in q:
                context += "\n\n The user wants to know about the person's POLITICAL activities. Provide those specific details from the search results."
            else:
                context += "\n\n The user wants MORE information about the same person/topic from the previous conversation. Provide additional details from the search results above."
        elif st.session_state.last_response:
            context = f"Previous conversation was about: {st.session_state.last_response[:800]}"
            context += f"\n\n User asked: '{query}'. Continue naturally from the previous conversation."
        else:
            context = get_current_datetime()
    
    # Handle FILE COMPARISON
    elif tool == "compare_files" and st.session_state.file_context and len(st.session_state.uploaded_files) >= 2:
        filenames = "\n".join([f"- {name}" for name in st.session_state.uploaded_files.keys()])
        with st.spinner("📊 Comparing files..."):
            response = compare_files(query, st.session_state.file_context, filenames)
            st.session_state.last_response = response
            st.session_state.last_topic = "file_comparison"
            return response
    
    # Handle single file task
    elif tool == "file_task" and st.session_state.file_context:
        filenames = "\n".join([f"- {name}" for name in st.session_state.uploaded_files.keys()])
        with st.spinner("📖 Reading your document..."):
            response = analyze_uploaded_files(query, st.session_state.file_context, filenames)
            st.session_state.last_response = response
            st.session_state.last_topic = "file_analysis"
            return response
    
    # Handle evaluation
    elif tool == "evaluate" and st.session_state.file_context:
        with st.spinner("📝 Evaluating your work..."):
            response = evaluate_work(query, st.session_state.file_context)
            st.session_state.last_response = response
            st.session_state.last_topic = "evaluation"
            return response
    
    # Handle URL scraping
    elif tool == "scrape_url":
        urls = extract_urls_from_query(query)
        scraped = ""
        for url in urls:
            with st.spinner(f"Reading {url}..."):
                content = scrape_webpage(url)
                if content:
                    scraped += f"\nContent from {url}:\n{content}\n"
        if scraped:
            context = scraped
        else:
            return "I couldn't read that link."
    
    # Handle calculator
    elif tool == "calculator":
        result = calculator(query)
        if result:
            return f"Result: {result}"
    
    # Handle datetime
    elif tool == "datetime":
        context += get_current_datetime()
    
    # Handle search for everything else
    else:
        search_result = internet_search(query)
        if search_result:
            context += "\n" + search_result
            # Store for follow-up questions
            st.session_state.last_search_query = query
            st.session_state.last_search_results = search_result
            st.session_state.last_topic = "search"
        else:
            context += get_current_datetime()
    
    # Generate response with conversation history
    answer = reason(query, context)
    
    # Store last response for continuation context
    st.session_state.last_response = answer
    
    # Don't store generic continuation responses in memory
    if not is_follow_up and q not in continuation_phrases:
        store_memory(answer)
    
    return answer

# ============================================
# UI - MAIN DISPLAY
# ============================================

st.markdown('<h1 style="text-align: center;">🧠 Mukiibi-Moses AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #667eea;">Intelligent Autonomous Agent with Real-Time Internet Access</p>', unsafe_allow_html=True)
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
    
    # Clear Files button
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
        
        # Comparison tips for multiple files
        if len(st.session_state.uploaded_files) >= 2:
            st.markdown("---")
            st.markdown("### 🔍 Comparison Tips")
            st.markdown("**Try asking:**")
            st.markdown("- \"Compare these files\"")
            st.markdown("- \"What are the differences?\"")
            st.markdown("- \"Which file has more detail?\"")
        
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
    st.markdown("**Focus:** AI Agents, LLM Applications, Real-time Systems")
    st.markdown("---")
    st.markdown("### ✨ Features")
    st.markdown("✅ **Real-time web search** (weather, news, facts)")
    st.markdown("✅ **File analysis** (PDF, DOCX, CSV, JSON, TXT)")
    st.markdown("✅ **File comparison** (compare multiple documents)")
    st.markdown("✅ **Current news headlines**")
    st.markdown("✅ **Calculator**")
    st.markdown("✅ **Conversation memory**")
    st.markdown("✅ **Work evaluation**")

# ============================================
# CHAT DISPLAY
# ============================================

# Display chat history
for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

# Chat input (fixed at bottom by CSS)
query = st.chat_input("Ask me anything... I can check weather, compare files, search the web, and more!")

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
