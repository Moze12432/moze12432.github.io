import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import requests
import re
from datetime import datetime
import urllib.parse

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Emotion AI Companion",
    page_icon="🧠",
    layout="centered"
)

# -------------------------
# Load Models
# -------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

@st.cache_resource
def load_flan_model():
    model_name = "google/flan-t5-large"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    return tokenizer, model

emotion_model = load_emotion_model()
flan_tokenizer, flan_model = load_flan_model()

# -------------------------
# Session State
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# -------------------------
# VAD Mapping
# -------------------------
VAD_MAP = {
    "joy": (0.9, 0.7, 0.8),
    "love": (0.95, 0.6, 0.85),
    "surprise": (0.7, 0.9, 0.6),
    "anger": (0.1, 0.85, 0.7),
    "fear": (0.1, 0.9, 0.2),
    "sadness": (0.1, 0.3, 0.2),
    "neutral": (0.5, 0.3, 0.5)
}

# -------------------------
# AI Identity
# -------------------------
def get_ai_identity():
    return "I am an AI assistant created by Moses, a student at KyungDong University. I can answer questions, do math, translate, and understand emotions."

# -------------------------
# MATH (direct calculation)
# -------------------------
def calculate_math(question):
    math_patterns = [
        (r"square root of (\d+)", lambda m: f"The square root of {m.group(1)} is {float(m.group(1))**0.5}"),
        (r"(\d+) squared", lambda m: f"{m.group(1)} squared is {int(m.group(1))**2}"),
        (r"(\d+) \* (\d+)", lambda m: f"{int(m.group(1))} × {int(m.group(2))} = {int(m.group(1)) * int(m.group(2))}"),
        (r"(\d+) \+ (\d+)", lambda m: f"{int(m.group(1))} + {int(m.group(2))} = {int(m.group(1)) + int(m.group(2))}"),
    ]
    
    for pattern, func in math_patterns:
        match = re.search(pattern, question.lower())
        if match:
            return func(match)
    return None

# -------------------------
# KNOWLEDGE BASE (direct answers)
# -------------------------
def get_direct_answer(question):
    q = question.lower().strip()
    
    # Physics laws
    if "newton's third law" in q or "newton third law" in q:
        return "Newton's Third Law of Motion states: For every action, there is an equal and opposite reaction. This means that whenever one object exerts a force on another object, the second object exerts a force of equal magnitude but in the opposite direction on the first object."
    
    if "newton's first law" in q:
        return "Newton's First Law of Motion (Law of Inertia) states: An object at rest stays at rest, and an object in motion stays in motion with the same speed and in the same direction unless acted upon by an unbalanced force."
    
    if "newton's second law" in q:
        return "Newton's Second Law of Motion states: Force equals mass times acceleration (F = ma). The acceleration of an object depends on the net force acting on it and its mass."
    
    # Translations
    translations = {
        "how to say hello in korean": "In Korean, 'hello' is '안녕하세요' (annyeonghaseyo).",
        "hello in korean": "Hello in Korean is '안녕하세요' (annyeonghaseyo).",
        "thank you in korean": "Thank you in Korean is '감사합니다' (gamsahamnida).",
        "how to say thank you in korean": "In Korean, 'thank you' is '감사합니다' (gamsahamnida).",
        "how to say hello in japanese": "In Japanese, 'hello' is 'こんにちは' (konnichiwa).",
        "how to say hello in spanish": "In Spanish, 'hello' is 'hola'.",
        "how to say hello in french": "In French, 'hello' is 'bonjour'.",
    }
    
    for key, value in translations.items():
        if key in q:
            return value
    
    # Capitals
    capitals = {
        "capital of france": "The capital of France is Paris.",
        "capital of japan": "The capital of Japan is Tokyo.",
        "capital of south korea": "The capital of South Korea is Seoul.",
        "capital of china": "The capital of China is Beijing.",
    }
    
    for key, value in capitals.items():
        if key in q:
            return value
    
    return None

# -------------------------
# WEB SEARCH WITH CLEAN PARSING
# -------------------------
def search_and_extract_answer(query):
    """Search the web and extract just the answer using FLAN"""
    try:
        # Search DuckDuckGo
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=15, headers=headers)
        
        if response.status_code == 200:
            # Extract all result snippets
            snippet_pattern = r'class="result__snippet"[^>]*>(.*?)</a>'
            snippets = re.findall(snippet_pattern, response.text, re.DOTALL)
            
            if snippets:
                # Clean the first snippet
                raw_snippet = snippets[0]
                raw_snippet = re.sub(r'<[^>]+>', '', raw_snippet)
                raw_snippet = re.sub(r'&[a-z]+;', '', raw_snippet)
                raw_snippet = ' '.join(raw_snippet.split())
                
                # Use FLAN to extract just the answer
                prompt = f"""Question: {query}

Search result snippet: "{raw_snippet}"

Extract ONLY the direct answer to the question from the snippet above. Do not include extra text, just the answer.

Answer:"""
                
                inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
                with torch.no_grad():
                    outputs = flan_model.generate(
                        inputs.input_ids,
                        max_length=150,
                        num_beams=4,
                        temperature=0.3,
                        pad_token_id=flan_tokenizer.eos_token_id
                    )
                answer = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Clean up the answer
                answer = answer.replace("Answer:", "").strip()
                if len(answer) > 10 and len(answer) < 500:
                    return answer
                
                # Fallback to cleaned snippet
                if len(raw_snippet) > 20:
                    return raw_snippet[:400]
    
    except Exception as e:
        pass
    
    return None

# -------------------------
# JOKES
# -------------------------
def get_joke():
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "What do you call a fake noodle? An impasta!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What do you call a bear with no teeth? A gummy bear!"
    ]
    return random.choice(jokes)

# -------------------------
# EMOTIONAL RESPONSES
# -------------------------
def get_emotional_response(user_input):
    user_lower = user_input.lower()
    
    if "sick" in user_lower:
        return "I'm sorry you're not feeling well. Please rest and take care of yourself."
    if "tired" in user_lower:
        return "I hear that you're tired. Can you take a short break or get some rest?"
    if "sad" in user_lower:
        return "I'm sorry you're feeling sad. Would you like to talk about it?"
    if "happy" in user_lower or "good" in user_lower:
        return "That's wonderful to hear! What's making you happy today?"
    if "stressed" in user_lower:
        return "Stress can be challenging. Take a deep breath. What might help you feel better?"
    
    return "I'm here for you. How can I support you right now?"

# -------------------------
# MAIN RESPONSE GENERATOR
# -------------------------
def generate_response(user_input):
    user_lower = user_input.lower()
    
    # 1. Identity
    if any(q in user_lower for q in ["who are you", "what are you"]):
        return get_ai_identity()
    
    # 2. Date and time
    if any(q in user_lower for q in ["date today", "today's date", "what day is it"]):
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."
    
    if any(q in user_lower for q in ["what time is it", "current time"]):
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."
    
    # 3. Math
    math_result = calculate_math(user_input)
    if math_result:
        return math_result
    
    # 4. Jokes
    if any(q in user_lower for q in ["joke", "funny", "make me laugh"]):
        return get_joke()
    
    # 5. Greetings
    if user_lower.strip() in ["hi", "hello", "hey"]:
        return "Hello! How can I help you today?"
    
    if "thank" in user_lower:
        return "You're very welcome!"
    
    # 6. Emotional support
    if any(em in user_lower for em in ["feel", "feeling", "sad", "happy", "angry", "tired", "sick", "stressed"]):
        return get_emotional_response(user_input)
    
    # 7. Direct answers (knowledge base)
    direct_answer = get_direct_answer(user_input)
    if direct_answer:
        return direct_answer
    
    # 8. Search the web
    with st.spinner("Searching for the answer..."):
        search_result = search_and_extract_answer(user_input)
        
        if search_result:
            return search_result
        
        # 9. Fallback
        return "I couldn't find an answer. Try rephrasing your question."

# -------------------------
# UI
# -------------------------
st.markdown("<h1 style='text-align: center;'>Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University</p>", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.markdown("### Features")
    st.markdown("- Direct answers to questions")
    st.markdown("- Math calculations")
    st.markdown("- Translations")
    st.markdown("- Emotional support")
    st.markdown("- Jokes")
    
    st.divider()
    
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.emotion_history = []
        st.rerun()
    
    st.divider()
    st.info("Try asking:\n\n- State Newton's third law\n- What is the square root of 16?\n- How to say hello in Korean?\n- Tell me a joke\n- I'm feeling tired")

if st.button("Show / Hide Emotional Insights"):
    st.session_state.show_dashboard = not st.session_state.show_dashboard

if st.session_state.show_dashboard and st.session_state.emotion_history:
    st.subheader("Emotional Insights")
    emotion_counts = {}
    for item in st.session_state.emotion_history[-20:]:
        e = item["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    if emotion_counts:
        st.bar_chart(emotion_counts)

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Emotion detection
    try:
        raw = emotion_model(user_input)
        emotions = raw[0] if isinstance(raw[0], list) else raw
        top = max(emotions, key=lambda x: x.get("score", 0))
        emotion = top.get("label", "neutral")
    except:
        emotion = "neutral"
    
    v, a, d = VAD_MAP.get(emotion.lower(), (0.5, 0.5, 0.5))
    st.session_state.emotion_history.append({
        "emotion": emotion, "valence": v, "arousal": a, "dominance": d
    })
    
    reply = generate_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
