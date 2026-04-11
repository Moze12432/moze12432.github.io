import streamlit as st
from groq import Groq
import requests
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================

MODEL_NAME = "llama3-70b-8192"
TEMPERATURE = 0
MAX_TOKENS = 600

# =========================================================
# LLM SETUP
# =========================================================

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def llm(messages):

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=messages
    )

    return completion.choices[0].message.content


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are MozeAI.

Created by Mukiibi Moses,
a Computer Engineering student at Kyungdong University.

Rules:
- Always give factual answers.
- Never invent facts.
- If information is unknown, say "I am not sure".
- Use provided context when available.
- Answer clearly and concisely.
"""


# =========================================================
# EMBEDDINGS + MEMORY
# =========================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

memory_store = []

def store_memory(text):

    if len(text) < 40:
        return

    if "I am not sure" in text:
        return

    vec = embedder.encode(text)

    memory_store.append((text, vec))


def retrieve_memory(query):

    if len(memory_store) == 0:
        return ""

    qvec = embedder.encode(query)

    scores = []

    for text, vec in memory_store:

        sim = np.dot(qvec, vec)

        scores.append((sim, text))

    scores.sort(reverse=True)

    top = scores[:3]

    return "\n".join([t[1] for t in top])


# =========================================================
# INTERNET SEARCH (Wikipedia)
# =========================================================

def internet_search(query):

    try:

        q = query.replace(" ", "_")

        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{q}"

        r = requests.get(url)

        data = r.json()

        return data.get("extract","")

    except:

        return ""


# =========================================================
# CALCULATOR TOOL
# =========================================================

def calculator(query):

    try:

        exp = re.findall(r'[\d\+\-\*\/\.\(\)]+', query)

        if exp:
            result = eval(exp[0])
            return str(result)

    except:
        pass

    return None


# =========================================================
# ROUTER
# =========================================================

def route(query):

    q = query.lower()

    if any(x in q for x in ["+","-","*","/","calculate"]):
        return "calculator"

    if any(x in q for x in [
        "who","when","where","president","leader",
        "capital","country","population",
        "tell me about","what is"
    ]):
        return "search"

    return "reason"


# =========================================================
# REASONING
# =========================================================

def reason(question, context):

    prompt = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"""
Context:
{context}

Question:
{question}

Answer using the context if possible.
If context is insufficient say "I am not sure".
"""}
    ]

    return llm(prompt)


# =========================================================
# SELF-VERIFICATION
# =========================================================

def verify(question, answer):

    prompt = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"""
Question:
{question}

Answer:
{answer}

Is the answer factually correct?

Reply only YES or NO.
"""}
    ]

    result = llm(prompt)

    return "yes" in result.lower()


# =========================================================
# AGENT CORE
# =========================================================

def run_agent(query):

    tool = route(query)

    context = ""

    # TOOL: calculator
    if tool == "calculator":

        result = calculator(query)

        if result:
            return result

    # TOOL: search
    if tool == "search":

        web = internet_search(query)

        context += web + "\n"

    # MEMORY
    mem = retrieve_memory(query)

    context += mem

    # REASONING
    answer = reason(query, context)

    # VERIFY
    if not verify(query, answer):

        web = internet_search(query)

        answer = reason(query, web)

    # STORE MEMORY
    store_memory(answer)

    return answer


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(page_title="MozeAI", page_icon="🧠")

st.title("MozeAI")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:

    with st.chat_message(role):
        st.write(msg)

query = st.chat_input("Ask anything")

if query:

    st.session_state.chat.append(("user", query))

    with st.chat_message("user"):
        st.write(query)

    response = run_agent(query)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.chat.append(("assistant", response))
