import streamlit as st
from groq import Groq
import requests
import re
import numpy as np
from sentence_transformers import SentenceTransformer

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
    
    /* New Chat button */
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
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1e1e2f 0%, #2d2d44 100%);
    }
    
    /* Chat input */
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
# ============================================
# CURRENT DATE & TIME
# ============================================

from datetime import datetime
import pytz  # You'll need to install: pip install pytz

def get_current_datetime():
    """Get current date and time"""
    tz = pytz.timezone('Asia/Seoul')  # Change to your timezone
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
# SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are MozeAI, an AI with REAL-TIME information access.

Created by Mukiibi Moses, a Computer Engineering student at Kyungdong University.
He is an AI builder focused on designing intelligent autonomous agents, language model applications, and practical AI systems that solve real-world problems such as education, automation, and decision support.
He is an active researcher on researchGate, Aademia and other research Platforms.


CAPABILITIES:
- Access to current date/time
- Real-time web search
- Latest news headlines
- Calculator for math problems
- Memory of past conversations

INSTRUCTIONS:
- Use the provided context which includes CURRENT information
- For time/date questions, use the current information provided
- For news/events, rely on the search results given
- Answer clearly and factually
- If information isn't in context, say "I don't have current information on that"
- Do not hallucinate or make up dates/events
- Do not show internal reasoning
"""



# ============================================
# SESSION MEMORY
# ============================================

if "memory_store" not in st.session_state:
    st.session_state.memory_store = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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
# WIKIPEDIA SEARCH
# ============================================

# ============================================
# REAL-TIME WEB SEARCH (using DuckDuckGo - free, no API key)
# ============================================

def internet_search(query):
    """Search the web for current information"""
    try:
        # Using DuckDuckGo HTML API (free, no key needed)
        url = "https://html.duckduckgo.com/html/"
        params = {"q": query}
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        }
        
        response = requests.post(url, data=params, headers=headers)
        
        if response.status_code == 200:
            # Extract search results
            import re
            results = re.findall(r'<a rel="nofollow" class="result__a" href="[^"]*">([^<]+)</a>', response.text)
            snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', response.text)
            
            if results and snippets:
                context = f"Recent search results for '{query}':\n\n"
                for i in range(min(3, len(results))):
                    context += f"• {results[i]}\n"
                    if i < len(snippets):
                        context += f"  {snippets[i]}\n\n"
                return context[:1500]
        
        # Fallback to Wikipedia
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


# ============================================
# NEWS SEARCH for current events
# ============================================

def get_current_news(topic="latest"):
    """Get current news (using NewsAPI - free tier)"""
    try:
        # You'll need a free API key from https://newsapi.org/
        api_key = st.secrets.get("NEWS_API_KEY", "")
        
        if not api_key:
            # Fallback to simple RSS feed
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
            # Use NewsAPI if you have key
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
# ROUTER
# ============================================

# ============================================
# ROUTER with current info triggers
# ============================================

def route(query):
    q = query.lower()
    
    # Check for calculator
    if any(x in q for x in ["+","-","*","/","×","calculate"]):
        return "calculator"
    
    # Check for current time/date requests
    if any(x in q for x in ["time", "date", "today", "current time", "what day", "what's the date"]):
        return "datetime"
    
    # Check for news requests
    if any(x in q for x in ["news", "headlines", "current events", "breaking", "latest"]):
        return "news"
    
    # Check for general web search
    if any(x in q for x in [
        "capital", "population", "leader", "history", "tell me about",
        "who is", "what is", "when did", "where is", "current", "recent",
        "latest", "today's", "this week", "2024", "2025"
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
# AGENT
# ============================================

# ============================================
# AGENT with current information
# ============================================

def run_agent(query):
    tool = route(query)
    context = ""
    
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
    
    # If no context found, add current date at least
    if not context:
        context = get_current_datetime()
    
    answer = reason(query, context)
    store_memory(answer)
    
    return answer

# ============================================
# UI
# ============================================

# Welcome message with styling
st.markdown('<h1>🧠 Mukiibi-Moses AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #667eea;">Your Intelligent Autonomous Agent</p>', unsafe_allow_html=True)
st.markdown("---")
# ============================================
# SIDEBAR WITH NEW CHAT BUTTON
# ============================================

with st.sidebar:
    st.markdown("### 🧠 MozeAI")
    st.markdown("---")
    
    if st.button("🔄 New Chat", use_container_width=True):
        # Reset all session states
        st.session_state.memory_store = []
        st.session_state.chat_history = []
        st.rerun()
    
    st.markdown("---")
    st.markdown("### About")
    st.markdown("Created by **Mukiibi Moses**")
    st.markdown("Computer Engineering @ Kyungdong University")
    st.markdown("---")
    st.markdown("### Features")
    st.markdown("✅ Access to current date/time")
    st.markdown("✅ Real-time web search")
    st.markdown("✅ Latest news headlines")
    st.markdown("✅ Calculator for math problems")
    st.markdown("✅ Memory of past Conversations")

for role, msg in st.session_state.chat_history:

    with st.chat_message(role):
        st.write(msg)

query = st.chat_input("Ask anything")

if query:

    st.session_state.chat_history.append(("user", query))

    with st.chat_message("user"):
        st.write(query)

    response = run_agent(query)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.chat_history.append(("assistant", response))
