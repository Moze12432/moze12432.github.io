import streamlit as st
from transformers import pipeline
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
# Load Emotion Model
# -------------------------
@st.cache_resource
def load_emotion_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_model = load_emotion_model()

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
    return "I am an AI assistant created by Moses, a student at KyungDong University. I can search the internet, translate languages, answer questions, and understand emotions."

# -------------------------
# IMPROVED INTERNET SEARCH - MULTIPLE SOURCES
# -------------------------

def search_duckduckgo_lite(query):
    """Search DuckDuckGo HTML version for real results"""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, timeout=10, headers=headers)
        
        if response.status_code == 200:
            # Extract result snippets
            snippets = re.findall(r'class="result__snippet"[^>]*>(.*?)</a>', response.text, re.DOTALL)
            if snippets:
                # Clean HTML tags
                clean = re.sub(r'<[^>]+>', '', snippets[0])
                clean = re.sub(r'&[a-z]+;', '', clean)
                return clean[:500]
    except:
        pass
    return None

def search_duckduckgo_api(query):
    """Search DuckDuckGo Instant Answer API"""
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get("AbstractText"):
                return data["AbstractText"]
            if data.get("Definition"):
                return data["Definition"]
            if data.get("Answer"):
                return data["Answer"]
            if data.get("RelatedTopics") and len(data["RelatedTopics"]) > 0:
                first = data["RelatedTopics"][0]
                if isinstance(first, dict) and first.get("Text"):
                    return first["Text"]
    except:
        pass
    return None

def search_wikipedia(query):
    """Search Wikipedia"""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if "extract" in data:
                return data["extract"]
    except:
        pass
    return None

def search_wikihow(query):
    """Search WikiHow for how-to questions"""
    if "how to" in query.lower():
        try:
            search_term = query.lower().replace("how to", "").strip()
            url = f"https://www.wikihow.com/api.php?action=query&list=search&srsearch={urllib.parse.quote(search_term)}&format=json"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data.get("query", {}).get("search"):
                    title = data["query"]["search"][0]["title"]
                    return f"How to {title.lower()}: Visit wikihow.com for step-by-step instructions."
        except:
            pass
    return None

def search_translation(query):
    """Handle translation requests specifically"""
    # Pattern: "how to say X in Y" or "translate X to Y"
    patterns = [
        r"how to say (.+?) in (\w+)",
        r"translate (.+?) to (\w+)",
        r"what is (.+?) in (\w+)",
        r"say (.+?) in (\w+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, query.lower())
        if match:
            word = match.group(1).strip()
            language = match.group(2).strip()
            
            # Use MyMemory API for translations (free)
            try:
                url = f"https://api.mymemory.translated.net/get?q={urllib.parse.quote(word)}&langpair=en|{language}"
                response = requests.get(url, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    if data.get("responseData", {}).get("translatedText"):
                        translation = data["responseData"]["translatedText"]
                        return f'"{word}" in {language.capitalize()} is: "{translation}"'
            except:
                pass
            
            # Fallback for common languages
            common_translations = {
                "korean": {"hello": "안녕하세요 (annyeonghaseyo)"},
                "japanese": {"hello": "こんにちは (konnichiwa)"},
                "chinese": {"hello": "你好 (nǐ hǎo)"},
                "spanish": {"hello": "hola"},
                "french": {"hello": "bonjour"},
                "german": {"hello": "hallo"},
                "italian": {"hello": "ciao"},
                "russian": {"hello": "здравствуйте (zdravstvuyte)"},
            }
            
            if language in common_translations and word in common_translations[language]:
                return f'"{word}" in {language.capitalize()} is: "{common_translations[language][word]}"'
            
            return f"I searched but couldn't find the translation for '{word}' in {language}. Could you try a different word or language?"
    
    return None

def comprehensive_search(query):
    """Combine all search methods"""
    
    # First, check if it's a translation request
    translation = search_translation(query)
    if translation:
        return translation
    
    # Check for how-to questions
    wikihow = search_wikihow(query)
    if wikihow:
        return wikihow
    
    # Try DuckDuckGo API
    result = search_duckduckgo_api(query)
    if result and len(result) > 20:
        return result
    
    # Try Wikipedia
    result = search_wikipedia(query)
    if result and len(result) > 50:
        return result
    
    # Try DuckDuckGo HTML as last resort
    result = search_duckduckgo_lite(query)
    if result and len(result) > 20:
        return result
    
    return None

# -------------------------
# RESPONSE GENERATION
# -------------------------

def get_joke():
    """Get a joke from the internet"""
    try:
        url = "https://official-joke-api.appspot.com/random_joke"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            return f"{data['setup']}\n\n{data['punchline']}"
    except:
        pass
    
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "What do you call a fake noodle? An impasta!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!"
    ]
    return random.choice(jokes)

def get_emotional_response(user_input):
    """Generate empathetic response"""
    user_lower = user_input.lower()
    
    if "sick" in user_lower:
        return "I'm sorry you're not feeling well. Please rest and take care of yourself."
    if "tired" in user_lower:
        return "I hear that you're tired. Can you take a short break or get some rest?"
    if "sad" in user_lower:
        return "I'm sorry you're feeling sad. Would you like to talk about it?"
    if "happy" in user_lower:
        return "That's wonderful to hear! What's making you happy today?"
    if "angry" in user_lower:
        return "I hear your frustration. Would you like to talk about what's bothering you?"
    if "stressed" in user_lower:
        return "Stress can be challenging. Take a deep breath. What might help you feel better?"
    
    return "I'm here for you. How can I support you right now?"

def generate_response(user_input):
    """Main response generator"""
    user_lower = user_input.lower()
    
    # Identity
    if any(q in user_lower for q in ["who are you", "what are you"]):
        return get_ai_identity()
    
    # Date and time
    if any(q in user_lower for q in ["date today", "today's date", "what day is it"]):
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."
    
    if any(q in user_lower for q in ["what time is it", "current time"]):
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."
    
    # Simple math
    calc_match = re.search(r'(\d+\s*[\+\-\*/]\s*\d+)', user_input)
    if calc_match:
        try:
            result = eval(calc_match.group(1))
            return f"{calc_match.group(1)} = {result}"
        except:
            pass
    
    # Jokes
    if any(q in user_lower for q in ["joke", "funny", "make me laugh"]):
        return get_joke()
    
    # Greetings
    if user_lower.strip() in ["hi", "hello", "hey"]:
        return "Hello! How can I help you today? Ask me anything, and I'll search the internet for the answer!"
    
    if "thank" in user_lower:
        return "You're very welcome! Is there anything else I can help with?"
    
    # Emotional
    if any(em in user_lower for em in ["feel", "feeling", "sad", "happy", "angry", "tired", "sick", "stressed"]):
        return get_emotional_response(user_input)
    
    # SEARCH THE INTERNET FOR EVERYTHING ELSE
    with st.spinner("Searching the internet..."):
        search_result = comprehensive_search(user_input)
        
        if search_result:
            return search_result
        
        # If search fails, try a more general search
        general_search = search_duckduckgo_lite(user_input)
        if general_search:
            return general_search
        
        return "I searched the internet but couldn't find an answer. Could you rephrase your question? For example: 'How to say hello in Korean?' or 'What is the capital of France?'"

# -------------------------
# UI
# -------------------------
st.markdown("<h1 style='text-align: center;'>Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University | Internet Search Powered</p>", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.markdown("### Features")
    st.markdown("- Searches the internet for answers")
    st.markdown("- Translates phrases (uses MyMemory API)")
    st.markdown("- Answers factual questions")
    st.markdown("- Provides how-to instructions")
    st.markdown("- Understands emotions")
    
    st.divider()
    
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.emotion_history = []
        st.rerun()
    
    st.divider()
    st.info("Try asking:\n\n- How to say hello in Korean?\n- Translate good morning to Spanish\n- What is the capital of Japan?\n- How to make coffee?\n- Tell me a joke\n- I'm feeling tired")

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

user_input = st.chat_input("Ask me anything! I'll search the internet...")

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
