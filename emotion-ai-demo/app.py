import streamlit as st
import torch
import random
import re
import requests
import sqlite3
import faiss
import numpy as np
from datetime import datetime
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer

# -------------------------
# PAGE CONFIG
# -------------------------

st.set_page_config(
    page_title="Autonomous AI Companion",
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

    cursor.execute("SELECT content FROM memories ORDER BY id DESC LIMIT 50")

    rows = cursor.fetchall()

    return [r[0] for r in rows]


# -------------------------
# VECTOR MEMORY
# -------------------------

embed_model = SentenceTransformer("all-MiniLM-L6-v2")

dimension = 384

index = faiss.IndexFlatL2(dimension)

vector_texts = []


def store_vector(text):

    vec = embed_model.encode([text])

    index.add(np.array(vec))

    vector_texts.append(text)


def retrieve_vector(query, k=3):

    if len(vector_texts) == 0:
        return []

    vec = embed_model.encode([query])

    D, I = index.search(np.array(vec), k)

    results = []

    for i in I[0]:

        if i < len(vector_texts):
            results.append(vector_texts[i])

    return results


# -------------------------
# LOAD LLM
# -------------------------

@st.cache_resource
def load_llm():

    model_name = "microsoft/DialoGPT-medium"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForCausalLM.from_pretrained(model_name)

    return tokenizer, model


tokenizer, model = load_llm()


def llm_generate(prompt):

    inputs = tokenizer.encode(prompt + tokenizer.eos_token, return_tensors="pt")

    output = model.generate(
        inputs,
        max_length=300,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9
    )

    response = tokenizer.decode(output[:, inputs.shape[-1]:][0], skip_special_tokens=True)

    return response


# -------------------------
# EMOTION MODEL
# -------------------------

@st.cache_resource
def load_emotion():

    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )


emotion_model = load_emotion()

# -------------------------
# INTERNET SEARCH
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


def calculator(expression):

    try:

        return str(eval(expression))

    except:

        return None


# -------------------------
# SELF REFLECTION
# -------------------------


def improve_response(answer):

    prompt = f"""
Improve this answer so it is clearer and more helpful.

Answer:
{answer}
"""

    improved = llm_generate(prompt)

    return improved


# -------------------------
# PLANNER
# -------------------------


def planner(query):

    q = query.lower()

    if "calculate" in q or re.search(r"\d+\s*[\+\-\*\/]\s*\d+", q):
        return "calculator"

    if "who is" in q or "what is" in q or "search" in q:
        return "search"

    if "time" in q:
        return "time"

    if "date" in q:
        return "date"

    return "llm"


# -------------------------
# AUTONOMOUS CONTROLLER
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

        return datetime.now().strftime("%A %B %d %Y")

    # MEMORY RETRIEVAL

    memories = retrieve_vector(query)

    history = "\n".join(memories)

    prompt = f"""
You are an intelligent AI assistant.

Relevant memory:
{history}

User: {query}

Answer clearly.
"""

    answer = llm_generate(prompt)

    answer = improve_response(answer)

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

    st.title("AI System")

    st.write("Autonomous AI Agent")

    if st.button("Clear Chat"):

        st.session_state.messages = []

    st.subheader("Stored Memory")

    mems = load_memories()

    for m in mems[:5]:
        st.write("-", m[:60])


# -------------------------
# UI
# -------------------------

st.title("🧠 Autonomous AI Companion")

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
