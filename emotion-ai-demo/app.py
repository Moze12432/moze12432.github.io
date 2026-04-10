import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import requests
import json
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

if "memory" not in st.session_state:
    st.session_state.memory = {
        "user_facts": {},
        "qa_pairs": []
    }

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
    return "I am an advanced AI assistant created by Moses, a student at KyungDong University. I search the internet in real-time to answer your questions accurately."

# -------------------------
# INTERNET SEARCH FUNCTIONS
# -------------------------

def search_duckduckgo_api(query):
    """Search DuckDuckGo Instant Answer API - Best for facts and translations"""
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            data = response.json()
            
            # Return the most relevant information
            if data.get("AbstractText"):
                return data["AbstractText"]
            if data.get("Definition"):
                return data["Definition"]
            if data.get("Answer"):
                return data["Answer"]
    except:
        pass
    return None

def search_wikipedia(query):
    """Search Wikipedia for detailed information"""
    try:
        # Clean query for better search
        search_terms = query.lower().strip()
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(search_terms)}"
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            data = response.json()
            if "extract" in data:
                return data["extract"]
    except:
        pass
    return None

def search_web(query):
    """Search the web using multiple sources"""
    # Try DuckDuckGo first (best for quick facts)
    result = search_duckduckgo_api(query)
    if result and len(result) > 20:
        return result
    
    # Try Wikipedia for detailed information
    result = search_wikipedia(query)
    if result and len(result) > 50:
        return result
    
    return None

# -------------------------
# RESPONSE ANALYSIS WITH FLAN
# -------------------------

def analyze_search_result(query, search_result):
    """Use FLAN to analyze search results and provide a clear answer"""
    try:
        prompt = f"""Question: {query}

Information found online: "{search_result}"

Please provide a clear, accurate, and helpful answer to the question based on this information. Keep it concise (2-3 sentences).

Answer:"""
        
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
        with torch.no_grad():
            outputs = flan_model.generate(
                inputs.input_ids,
                max_length=200,
                num_beams=4,
                temperature=0.5,
                do_sample=True,
                pad_token_id=flan_tokenizer.eos_token_id
            )
        response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if response and len(response) > 10 and "Answer:" not in response:
            return response
    except:
        pass
    return None

# -------------------------
# UTILITY FUNCTIONS
# -------------------------

def get_current_date_time():
    """Get current date and time"""
    now = datetime.now()
    return now.strftime("%A, %B %d, %Y at %I:%M %p")

def calculate(expression):
    """Safely calculate mathematical expressions"""
    try:
        expression = re.sub(r'[^0-9+\-*/().% ]', '', expression)
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return None

def get_joke():
    """Get a random joke from the internet"""
    try:
        url = "https://official-joke-api.appspot.com/random_joke"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"{data['setup']}\n\n{data['punchline']}"
    except:
        pass
    # Fallback jokes
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "What do you call a fake noodle? An impasta!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!"
    ]
    return random.choice(jokes)

# -------------------------
# EMOTIONAL RESPONSES
# -------------------------

def get_emotional_response(user_input):
    """Generate empathetic emotional response"""
    user_lower = user_input.lower()
    
    if "sick" in user_lower:
        return "I'm sorry you're not feeling well. Please rest and take care of yourself. Do you need anything specific?"
    
    if "tired" in user_lower:
        return "I hear that you're tired. Can you take a short break or get some rest? Your well-being is important."
    
    if "sad" in user_lower:
        return "I'm sorry you're feeling sad. Would you like to talk about what's bothering you? Sometimes sharing helps."
    
    if "happy" in user_lower or "good" in user_lower:
        return "That's wonderful to hear! What's making you happy today?"
    
    if "angry" in user_lower or "frustrated" in user_lower:
        return "I hear your frustration. Would you like to talk about what's bothering you?"
    
    if "stressed" in user_lower or "anxious" in user_lower:
        return "Stress can be challenging. Take a deep breath. What's one small thing that might help you feel better?"
    
    return "I'm here for you. How can I support you right now?"

# -------------------------
# MAIN RESPONSE GENERATOR - ALL QUESTIONS GO TO INTERNET
# -------------------------

def generate_smart_response(user_input):
    """Generate response by searching the internet for every question"""
    
    user_lower = user_input.lower()
    
    # 1. Identity (don't need search for this)
    if any(q in user_lower for q in ["who are you", "what are you", "your creator"]):
        return get_ai_identity()
    
    # 2. Date and time (real-time, no search needed)
    if any(q in user_lower for q in ["date today", "today's date", "what day is it", "current date"]):
        return f"Today is {get_current_date_time()}."
    
    if any(q in user_lower for q in ["what time is it", "current time"]):
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."
    
    # 3. Simple calculations
    calc_match = re.search(r'(\d+\s*[\+\-\*/%]\s*\d+)', user_input)
    if calc_match:
        try:
            result = calculate(user_input)
            if result:
                return result
        except:
            pass
    
    # 4. Greetings
    if user_lower.strip() in ["hi", "hello", "hey", "greetings"]:
        return "Hello! How are you feeling today? Ask me anything, and I'll search the internet for the answer!"
    
    # 5. Thanks
    if "thank" in user_lower:
        return "You're very welcome! Is there anything else I can help you with?"
    
    # 6. Emotional responses
    if any(em in user_lower for em in ["feel", "feeling", "sad", "happy", "angry", "scared", "tired", "sick", "stressed"]):
        return get_emotional_response(user_input)
    
    # 7. Jokes
    if any(q in user_lower for q in ["joke", "funny", "make me laugh"]):
        return get_joke()
    
    # 8. FOR EVERYTHING ELSE - SEARCH THE INTERNET
    with st.spinner("Searching the internet for accurate information..."):
        search_result = search_web(user_input)
        
        if search_result:
            # Analyze the search result to provide a clear answer
            analyzed = analyze_search_result(user_input, search_result)
            if analyzed:
                return analyzed
            else:
                return f"Based on my search:\n\n{search_result}"
        else:
            # If search fails, try a different query
            alt_result = search_web(user_input + " meaning")
            if alt_result:
                analyzed = analyze_search_result(user_input, alt_result)
                if analyzed:
                    return analyzed
                return f"Based on my search:\n\n{alt_result}"
            
            return "I searched the internet but couldn't find reliable information. Could you rephrase your question? I can help with translations, facts, how-to questions, and more!"

# -------------------------
# UI COMPONENTS
# -------------------------

def show_memory_dashboard():
    """Display what the AI has learned"""
    st.subheader("What I've Learned")
    
    if st.session_state.memory["user_facts"]:
        st.write("About you:")
        for key, value in st.session_state.memory["user_facts"].items():
            st.write(f"- {key.capitalize()}: {value}")

def reset_chat():
    """Reset the entire conversation"""
    st.session_state.messages = []
    st.session_state.emotion_history = []
    st.session_state.memory = {
        "user_facts": {},
        "qa_pairs": []
    }
    st.rerun()

# -------------------------
# MAIN UI
# -------------------------
st.markdown("<h1 style='text-align: center;'>Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University | Powered by Internet Search</p>", unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### How I Work")
    st.markdown("- I search the internet for EVERY question")
    st.markdown("- I analyze search results to give clear answers")
    st.markdown("- I can translate phrases, answer facts, explain concepts")
    st.markdown("- I understand emotions and provide support")
    
    st.divider()
    
    if st.button("Start New Chat", use_container_width=True):
        reset_chat()
    
    st.divider()
    show_memory_dashboard()
    
    st.divider()
    st.info("Try asking:\n\n- How to say hello in Korean?\n- What is the capital of France?\n- How to make pasta?\n- What is quantum physics?\n- Tell me a joke\n- I'm feeling tired")

# Dashboard toggle
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

# Chat display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything! I'll search the internet...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Emotion detection (for dashboard only)
    try:
        raw = emotion_model(user_input)
        emotions = raw[0] if isinstance(raw[0], list) else raw
        top = max(emotions, key=lambda x: x.get("score", 0))
        emotion = top.get("label", "neutral")
    except:
        emotion = "neutral"
    
    v, a, d = VAD_MAP.get(emotion.lower(), (0.5, 0.5, 0.5))
    st.session_state.emotion_history.append({
        "emotion": emotion,
        "valence": v,
        "arousal": a,
        "dominance": d
    })
    
    # Generate response - this will search the internet
    reply = generate_smart_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
