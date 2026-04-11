import streamlit as st
import sqlite3
import numpy as np
import re
import requests
import os
from datetime import datetime

from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# PAGE CONFIG
# -----------------------------

st.set_page_config(
    page_title="MozeAI Autonomous Agent",
    page_icon="🧠",
    layout="wide"
)

# -----------------------------
# GROQ CLIENT
# -----------------------------

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


def llm_generate(prompt):

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        temperature=0.2,
        max_tokens=400,
        messages=[
            {
                "role": "system",
                "content": """
You are MozeAI, an intelligent AI assistant created by Mukiibi Moses.

Rules:
- Always answer factually and clearly.
- If you are unsure about something, say "I am not certain".
- Never invent facts about real people.
- Use reasoning before answering.
- If the question requires calculation, compute it.
- If the question requires general knowledge, answer based on known facts.
"""
            },
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content
# -----------------------------
# SQLITE MEMORY
# -----------------------------

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

    cursor.execute("SELECT content FROM memories ORDER BY id DESC LIMIT 20")
    rows = cursor.fetchall()

    return [r[0] for r in rows]


# -----------------------------
# VECTOR MEMORY
# -----------------------------

@st.cache_resource
def load_embed_model():

    return SentenceTransformer("all-MiniLM-L6-v2")


embed_model = load_embed_model()

vector_memory = []


def store_vector(text):

    vec = embed_model.encode(text)

    vector_memory.append((text, vec))


def retrieve_vector(query, k=3):

    if not vector_memory:
        return []

    qvec = embed_model.encode(query)

    scores = []

    for text, vec in vector_memory:

        sim = np.dot(qvec, vec)

        scores.append((sim, text))

    scores.sort(reverse=True)

    return [s[1] for s in scores[:k]]


# -----------------------------
# KNOWLEDGE BASE
# -----------------------------

knowledge_base = []


def load_knowledge():

    folder = "knowledge"

    if not os.path.exists(folder):
        return

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        try:

            with open(path, "r", encoding="utf-8") as f:

                text = f.read()

                knowledge_base.append(text)

        except:
            pass


load_knowledge()


def retrieve_knowledge(query):

    if not knowledge_base:
        return ""

    qvec = embed_model.encode(query)

    scores = []

    for text in knowledge_base:

        vec = embed_model.encode(text)

        sim = np.dot(qvec, vec)

        scores.append((sim, text))

    scores.sort(reverse=True)

    return scores[0][1]


# -----------------------------
# INTERNET SEARCH
# -----------------------------

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


# -----------------------------
# CALCULATOR
# -----------------------------

def calculator(query):

    try:

        expression = re.findall(r'[\d\.\+\-\*\/\(\)]+', query)

        if expression:

            return str(eval(expression[0]))

    except:
        pass

    return None


# -----------------------------
# AI TASK PLANNER
# -----------------------------

def create_plan(query):

    prompt = f"""
You are an AI planner.

Break the task into steps.

User request:
{query}

Steps:
"""

    return llm_generate(prompt)


# -----------------------------
# TOOL DECISION
# -----------------------------

def decide_tool(query):

    prompt = f"""
You are an autonomous AI agent.

Available tools:
search
calculator
memory
knowledge
llm

User query:
{query}

Return ONLY the tool name.
"""

    tool = llm_generate(prompt)

    return tool.lower().strip()


# -----------------------------
# AGENT CONTROLLER
# -----------------------------

def agent_controller(query):

    today = datetime.now().strftime("%Y-%m-%d")

    # Force calculator if math detected
    if re.search(r"\d+\s*[\+\-\*\/]\s*\d+", query):

        result = calculator(query)

        if result:
            return f"🧮 Result: {result}"

    plan = create_plan(query)

    tool = decide_tool(query)

    if tool == "search":

        result = internet_search(query)

        if result:
            return f"🔎 {result}"

    memories = retrieve_vector(query)

    memory_context = "\n".join(memories)

    knowledge_context = retrieve_knowledge(query)

    prompt = f"""
You are MozeAI created by Mukiibi Moses,
a Computer Engineering student at KyungDong University.

Today's date: {today}

Relevant knowledge:
{knowledge_context}

Conversation memory:
{memory_context}

User question:
{query}

Answer clearly and accurately.
"""

    answer = llm_generate(prompt)

    store_vector(query)
    store_vector(answer)

    save_memory(query)
    save_memory(answer)

    return f"🧠 Plan:\n{plan}\n\n💡 Answer:\n{answer}"


# -----------------------------
# SESSION STATE
# -----------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# -----------------------------
# SIDEBAR
# -----------------------------

with st.sidebar:

    st.title("MozeAI")

    st.write("Autonomous AI Agent")

    if st.button("Clear Chat"):

        st.session_state.messages = []

    st.subheader("Recent Memory")

    mems = load_memories()

    for m in mems[:5]:

        st.write("-", m[:60])


# -----------------------------
# CHAT UI
# -----------------------------

st.title("🧠 MozeAI Autonomous Agent")

for m in st.session_state.messages:

    with st.chat_message(m["role"]):

        st.write(m["content"])


query = st.chat_input("Ask anything...")

if query:

    st.session_state.messages.append(
        {"role": "user", "content": query}
    )

    with st.chat_message("user"):

        st.write(query)

    with st.spinner("Thinking..."):

        response = agent_controller(query)

    with st.chat_message("assistant"):

        st.write(response)

    st.session_state.messages.append(
        {"role": "assistant", "content": response}
    )
