import streamlit as st
import sqlite3
import numpy as np
import re
import requests
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions
# -------------------------
# PAGE SETTINGS
# -------------------------

st.set_page_config(
    page_title="Autonomous AI Agent",
    page_icon="🧠",
    layout="wide"
)

# -------------------------
# DATABASE MEMORY
# -------------------------

conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS memories(
id INTEGER PRIMARY KEY AUTOINCREMENT,
content TEXT
)
""")

conn.commit()

def save_memory(text):
    cursor.execute("INSERT INTO memories(content) VALUES (?)", (text,))
    conn.commit()

def load_memories():
    cursor.execute("SELECT content FROM memories ORDER BY id DESC LIMIT 30")
    rows = cursor.fetchall()
    return [r[0] for r in rows]

# -------------------------
# VECTOR MEMORY
# -------------------------

@st.cache_resource
def load_embed_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

embed_model = load_embed_model()

vector_memory = []

def store_vector(text):
    vec = embed_model.encode(text)
    vector_memory.append((text, vec))

def retrieve_vector(query, k=2):

    if not vector_memory:
        return []

    qvec = embed_model.encode(query)

    scores = []

    for text, vec in vector_memory:
        sim = np.dot(qvec, vec)
        scores.append((sim, text))

    scores.sort(reverse=True)

    return [s[1] for s in scores[:k]]

# -------------------------
# LOAD LLM (FLAN-T5)
# -------------------------

@st.cache_resource
def load_llm():

    model_name = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    return tokenizer, model

tokenizer, model = load_llm()

def llm_generate(prompt):

    inputs = tokenizer(prompt, return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_length=150,
        temperature=0.7
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -------------------------
# INTERNET SEARCH TOOL
# -------------------------

def internet_search(query):

    try:

        url = f"https://api.duckduckgo.com/?q={query}&format=json"

        r = requests.get(url)

        data = r.json()

        if data.get("AbstractText"):
            return data["AbstractText"]

    except:
        pass

    return None

# -------------------------
# CALCULATOR TOOL
# -------------------------

def calculator(query):

    try:

        expression = re.findall(r'[\d\.\+\-\*\/\(\)]+', query)

        if expression:
            return str(eval(expression[0]))

    except:
        pass

    return None

# -------------------------
# PLANNER (TOOL ROUTER)
# -------------------------

def planner(query):

    q = query.lower()

    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", q):
        return "calculator"

    if "who is" in q or "what is" in q or "search" in q:
        return "search"

    if "time" in q:
        return "time"

    if "date" in q:
        return "date"

    return "llm"

# -------------------------
# AGENT CONTROLLER
# -------------------------

def agent_controller(query):

    task = planner(query)

    if task == "calculator":

        result = calculator(query)

        if result:
            return f"🧮 Result: {result}"

    if task == "search":

        result = internet_search(query)

        if result:
            return f"🔎 {result}"

    if task == "time":
        return datetime.now().strftime("%H:%M:%S")

    if task == "date":
        return datetime.now().strftime("%A, %B %d, %Y")

    memories = retrieve_vector(query)

    memory_context = "\n".join(memories)

    prompt = f"""
You are a helpful AI assistant.

Use the context if relevant.

Context:
{memory_context}

Question:
{query}

Answer clearly and concisely.
"""

    answer = llm_generate(prompt)

    store_vector(query)
    store_vector(answer)

    save_memory(query)
    save_memory(answer)

    return answer

# -------------------------
# SESSION STATE
# -------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# SIDEBAR
# -------------------------

with st.sidebar:

    st.title("AI Agent")

    st.write("Autonomous AI Assistant")

    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.subheader("Recent Memory")

    mems = load_memories()

    for m in mems[:5]:
        st.write("-", m[:60])

# -------------------------
# CHAT UI
# -------------------------

st.title("🧠 Autonomous AI Agent")

for m in st.session_state.messages:

    with st.chat_message(m["role"]):
        st.write(m["content"])

query = st.chat_input("Ask anything...")

if query:

    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("user"):
        st.write(query)

    with st.spinner("Thinking..."):

        response = agent_controller(query)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
