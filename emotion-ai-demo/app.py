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
You are MozeAI.

Created by Mukiibi Moses, a Computer Engineering student at Kyungdong University.
He is an AI builder focused on designing intelligent autonomous agents, language model applications, and practical AI systems that solve real-world problems such as education, automation, and decision support.
He is an active researcher on researchGate, Aademia and other research Platforms.

Rules:
- Answer clearly and factually
- Do not hallucinate
- If unsure say "I am not sure"
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

def internet_search(query):

    try:

        url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        q = query.strip().replace(" ", "_")

        r = requests.get(url + q)

        if r.status_code != 200:
            return ""

        data = r.json()

        return data.get("extract", "")[:1000]

    except:
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

def route(query):

    q = query.lower()

    if any(x in q for x in ["+","-","*","/","×","calculate"]):
        return "calculator"

    if any(x in q for x in [
        "capital",
        "population",
        "leader",
        "history",
        "tell me about"
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

def run_agent(query):

    tool = route(query)

    context = ""

    # calculator
    if tool == "calculator":

        result = calculator(query)

        if result:
            return result

    # search
    if tool == "search":

        web = internet_search(query)

        context += web

    # memory
    mem = retrieve_memory(query)

    context += "\n" + mem

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
    st.markdown("✅ Calculator")
    st.markdown("✅ Wikipedia Search")
    st.markdown("✅ Memory System")
    st.markdown("✅ Autonomous Reasoning")

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
