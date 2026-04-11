import streamlit as st
from groq import Groq
import requests
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime

# ======================================================
# CONFIG
# ======================================================

MODEL_NAME = "llama3-70b-8192"
TEMPERATURE = 0.1
MAX_TOKENS = 700

# ======================================================
# LLM SETUP
# ======================================================

client = Groq(api_key=st.secrets["GROQ_API_KEY"])

def llm(messages):

    completion = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        messages=messages
    )

    return completion.choices[0].message.content


# ======================================================
# SYSTEM PROMPT
# ======================================================

SYSTEM_PROMPT = """
You are MozeAI.

Created by Mukiibi Moses,
a Computer Engineering student at Kyungdong University.

Rules:
- Always prefer factual answers
- Never invent facts
- If unsure say "I am not sure"
- Use context provided
- Be concise but clear
"""


# ======================================================
# EMBEDDINGS + MEMORY
# ======================================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

memory_store = []

def store_memory(text):

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


# ======================================================
# INTERNET SEARCH
# ======================================================

def internet_search(query):

    try:

        url = f"https://api.duckduckgo.com/?q={query}&format=json"

        r = requests.get(url)

        data = r.json()

        text = data.get("AbstractText","")

        if text == "":
            text = data.get("Heading","")

        return text

    except:
        return ""


# ======================================================
# CALCULATOR
# ======================================================

def calculator(query):

    try:

        exp = re.findall(r'[\d\+\-\*\/\.\(\)]+',query)

        if exp:
            result = eval(exp[0])
            return str(result)

    except:
        pass

    return None


# ======================================================
# ROUTER
# ======================================================

def route(query):

    q = query.lower()

    if any(x in q for x in ["+","-","*","/","calculate"]):
        return "calculator"

    if any(x in q for x in [
        "who","when","where","president","leader",
        "capital","country","population"
    ]):
        return "search"

    return "reason"


# ======================================================
# PLANNING STEP
# ======================================================

def plan(question):

    prompt = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"""
Create a short reasoning plan to answer this question.

Question:
{question}

Plan steps only.
"""}
    ]

    return llm(prompt)


# ======================================================
# REASONING
# ======================================================

def reason(question, context):

    prompt = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"""
Context:
{context}

Question:
{question}

Answer using the context if possible.
If the context is insufficient say you are unsure.
"""}
    ]

    return llm(prompt)


# ======================================================
# FACT CHECK
# ======================================================

def verify(question, answer):

    prompt = [
        {"role":"system","content":SYSTEM_PROMPT},
        {"role":"user","content":f"""
Question:
{question}

Answer:
{answer}

Is the answer factually correct and grounded?

Reply only YES or NO.
"""}
    ]

    result = llm(prompt)

    return "yes" in result.lower()


# ======================================================
# AGENT CORE
# ======================================================

def run_agent(query):

    # Step 1: plan
    reasoning_plan = plan(query)

    # Step 2: route tool
    tool = route(query)

    context = ""

    # Step 3: tools
    if tool == "calculator":

        calc = calculator(query)

        if calc:
            return calc

    if tool == "search":

        web = internet_search(query)

        context += web + "\n"

    # Step 4: memory
    mem = retrieve_memory(query)

    context += mem

    # Step 5: reasoning
    answer = reason(query, context)

    # Step 6: verification
    if not verify(query, answer):

        fallback = internet_search(query)

        answer = reason(query, fallback)

    # Step 7: store memory
    store_memory(query)
    store_memory(answer)

    return answer


# ======================================================
# STREAMLIT UI
# ======================================================

st.set_page_config(page_title="MozeAI", page_icon="🧠")

st.title("MozeAI")

if "chat" not in st.session_state:
    st.session_state.chat = []

for role, msg in st.session_state.chat:

    with st.chat_message(role):
        st.write(msg)


query = st.chat_input("Ask anything")


if query:

    st.session_state.chat.append(("user",query))

    with st.chat_message("user"):
        st.write(query)

    response = run_agent(query)

    with st.chat_message("assistant"):
        st.write(response)

    st.session_state.chat.append(("assistant",response))
