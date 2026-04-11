import streamlit as st
import sqlite3
import numpy as np
import re
import requests
import os
from datetime import datetime

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils import embedding_functions

# ----------------------------------
# PAGE CONFIG
# ----------------------------------

st.set_page_config(
    page_title="MozeAI Autonomous Agent",
    page_icon="🧠",
    layout="wide"
)

# ----------------------------------
# SQLITE MEMORY
# ----------------------------------

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


# ----------------------------------
# VECTOR MEMORY
# ----------------------------------

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


# ----------------------------------
# CHROMA KNOWLEDGE BASE
# ----------------------------------

client = chromadb.Client(
    settings=chromadb.Settings(
        persist_directory="chroma_db"
    )
)

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="moze_knowledge",
    embedding_function=embedding_function
)


def load_knowledge():

    folder = "knowledge"

    if not os.path.exists(folder):
        return

    for file in os.listdir(folder):

        path = os.path.join(folder, file)

        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        try:
            collection.add(
                documents=[text],
                ids=[file]
            )
        except:
            pass


load_knowledge()


def retrieve_knowledge(query):

    results = collection.query(
        query_texts=[query],
        n_results=2
    )

    docs = results["documents"][0]

    return "\n".join(docs)


def learn_new_fact(query, answer):

    text = f"{query} -> {answer}"

    try:
        collection.add(
            documents=[text],
            ids=[str(datetime.now().timestamp())]
        )
    except:
        pass


# ----------------------------------
# LOAD LLM
# ----------------------------------

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


# ----------------------------------
# INTERNET SEARCH TOOL
# ----------------------------------

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


# ----------------------------------
# CALCULATOR TOOL
# ----------------------------------

def calculator(query):

    try:

        expression = re.findall(r'[\d\.\+\-\*\/\(\)]+', query)

        if expression:
            return str(eval(expression[0]))

    except:
        pass

    return None


# ----------------------------------
# AI TASK PLANNER
# ----------------------------------

def create_plan(query):

    prompt = f"""
You are an AI planner.

Break the task into steps.

User request:
{query}

Steps:
"""

    return llm_generate(prompt)


# ----------------------------------
# AI TOOL DECISION
# ----------------------------------

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

    tool = llm_generate(prompt).lower()

    return tool.strip()


# ----------------------------------
# AGENT CONTROLLER
# ----------------------------------

def agent_controller(query):

    plan = create_plan(query)

    tool = decide_tool(query)

    if tool == "calculator":

        result = calculator(query)

        if result:
            return f"🧮 Result: {result}"

    if tool == "search":

        result = internet_search(query)

        if result:
            return f"🔎 {result}"

    if tool == "knowledge":

        knowledge = retrieve_knowledge(query)

        prompt = f"""
You are MozeAI.

Use the knowledge below.

Knowledge:
{knowledge}

Question:
{query}

Answer clearly.
"""

        answer = llm_generate(prompt)

    else:

        memories = retrieve_vector(query)

        memory_context = "\n".join(memories)

        prompt = f"""
You are MozeAI created by Mukiibi Moses,
a Computer Engineering student at KyungDong University.

Memory:
{memory_context}

Question:
{query}

Answer clearly.
"""

        answer = llm_generate(prompt)

    store_vector(query)
    store_vector(answer)

    save_memory(query)
    save_memory(answer)

    learn_new_fact(query, answer)

    return f"🧠 Plan:\n{plan}\n\n💡 Answer:\n{answer}"


# ----------------------------------
# SESSION STATE
# ----------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


# ----------------------------------
# SIDEBAR
# ----------------------------------

with st.sidebar:

    st.title("MozeAI")

    st.write("Autonomous AI Agent")

    if st.button("Clear Chat"):
        st.session_state.messages = []

    st.subheader("Recent Memory")

    mems = load_memories()

    for m in mems[:5]:
        st.write("-", m[:60])


# ----------------------------------
# CHAT UI
# ----------------------------------

st.title("🧠 MozeAI Autonomous Agent")

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
