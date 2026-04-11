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
MAX_TOKENS = 500

# =========================================================
# STREAMLIT SETTINGS
# =========================================================

st.set_page_config(page_title="MozeAI", page_icon="🧠")

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

    return completion.choices[0].message.content.strip()


# =========================================================
# SYSTEM PROMPT
# =========================================================

SYSTEM_PROMPT = """
You are MozeAI.

Created by Mukiibi Moses,
a Computer Engineering student at Kyungdong University.

Rules:
- Provide factual answers.
- Never invent facts.
- If unsure say "I am not sure".
- Be concise and clear.
"""


# =========================================================
# SESSION MEMORY (PER USER)
# =========================================================

if "memory_store" not in st.session_state:
    st.session_state.memory_store = []

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# =========================================================
# EMBEDDINGS
# =========================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")


# =========================================================
# MEMORY FUNCTIONS
# =========================================================

def store_memory(text):

    if len(text) < 40:
        return

    if "I am not sure" in text:
        return

    vec = embedder.encode(text)

    st.session_state.memory_store.append((text, vec))


def retrieve_memory(query):

    memory_store = st.session_state.memory_store

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
# INTERNET SEARCH (WIKIPEDIA)
# =========================================================

def internet_search(query):

    try:

        url = "https://en.wikipedia.org/api/rest_v1/page/summary/"

        q = query.strip().replace(" ", "_")

        r = requests.get(url + q)

        data = r.json()

        return data.get("extract","")

    except:

        return ""


# =========================================================
# CALCULATOR TOOL
# =========================================================

def calculator(query):

    try:

        expression = query.lower()

        expression = expression.replace("×","*")
        expression = expression.replace("x","*")

        numbers = re.findall(r'[0-9\+\-\*\/\.\(\) ]+', expression)

        if numbers:
            result = eval(numbers[0])
            return str(result)

    except:
        pass

    return None


# =========================================================
# ROUTER
# =========================================================

def route(query):

    q = query.lower()

    if any(x in q for x in ["+","-","*","/","calculate","×"]):
        return "calculator"

    if any(x in q for x in [
        "who","when","where","president",
        "capital","leader","population",
        "tell me about","what is"
    ]):
        return "search"

    return "reason"


# =========================================================
# CLEAN RESPONSE
# =========================================================

def clean_answer(text):

    if "💡 Answer:" in text:
        text = text.split("💡 Answer:")[-1]

    if "🧠 Plan:" in text:
        text = text.split("🧠 Plan:")[0]

    return text.strip()


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

Answer clearly using the context if possible.
If context is insufficient say "I am not sure".
"""}
    ]

    return clean_answer(llm(prompt))


# =========================================================
# SELF VERIFICATION
# =========================================================

def verify(question, answer):

    prompt = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"""
Question:
{question}

Answer:
{answer}

Is this answer factually correct?

Reply only YES or NO.
"""}
    ]

    result = llm(prompt)

    return "yes" in result.lower()


# =========================================================
# AGENT PIPELINE
# =========================================================

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

        context += web + "\n"

    # memory
    mem = retrieve_memory(query)

    context += mem

    # reasoning
    answer = reason(query, context)

    # verification
    if not verify(query, answer):

        web = internet_search(query)

        answer = reason(query, web)

    # store useful memory
    store_memory(answer)

    return answer


# =========================================================
# STREAMLIT UI
# =========================================================

st.title("MozeAI")

for role, msg in st.session_state.chat_history:

    if role == "user":

        with st.chat_message("user"):
            st.write(msg)

    elif role == "assistant":

        with st.chat_message("assistant"):
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
