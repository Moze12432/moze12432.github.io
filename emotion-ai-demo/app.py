import streamlit as st
import sqlite3
import numpy as np
import re
import requests
from datetime import datetime
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# CONFIG
# -----------------------------

st.set_page_config(page_title="MozeAI", page_icon="🧠", layout="wide")

client = Groq(api_key=st.secrets["GROQ_API_KEY"])


# -----------------------------
# LLM FUNCTION
# -----------------------------

def llm_generate(prompt):

    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        temperature=0.2,
        max_tokens=400,
        messages=[
            {
                "role": "system",
                "content": """
You are MozeAI.

Created by Mukiibi Moses,
a Computer Engineering student at KyungDong University.

Rules:
- Be factual.
- If unsure, say you are unsure.
- Use reasoning.
"""
            },
            {"role": "user", "content": prompt}
        ]
    )

    return completion.choices[0].message.content


# -----------------------------
# MEMORY DATABASE
# -----------------------------

conn = sqlite3.connect("memory.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS memory(
id INTEGER PRIMARY KEY AUTOINCREMENT,
content TEXT
)
""")

conn.commit()


def save_memory(text):

    cursor.execute("INSERT INTO memory(content) VALUES (?)", (text,))
    conn.commit()


def load_memory():

    cursor.execute("SELECT content FROM memory ORDER BY id DESC LIMIT 10")
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
# THINKING AGENT LOOP
# -----------------------------

def reasoning_loop(query):

    context = ""
    thoughts = []

    for step in range(3):  # multi-step reasoning

        prompt = f"""
User question: {query}

Current context:
{context}

Think about the next step.

Choose one action:
SEARCH
CALCULATE
ANSWER

Explain briefly.
"""

        thought = llm_generate(prompt)

        thoughts.append(thought)

        if "SEARCH" in thought:

            result = internet_search(query)

            if result:
                context += result + "\n"

        elif "CALCULATE" in thought:

            result = calculator(query)

            if result:
                context += "Calculation result: " + result

        elif "ANSWER" in thought:

            break

    final_prompt = f"""
Use the gathered information to answer.

Context:
{context}

Question:
{query}
"""

    answer = llm_generate(final_prompt)

    return thoughts, answer


# -----------------------------
# AGENT CONTROLLER
# -----------------------------

def agent_controller(query):

    today = datetime.now().strftime("%A, %B %d, %Y")

    memories = retrieve_vector(query)

    memory_context = "\n".join(memories)

    thoughts, answer = reasoning_loop(query)

    store_vector(query)
    store_vector(answer)

    save_memory(query)
    save_memory(answer)

    thought_text = "\n".join(thoughts)

    return f"""
🧠 Reasoning:
{thought_text}

💡 Answer:
{answer}
"""


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

    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.subheader("Recent Memory")

    mems = load_memory()

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
