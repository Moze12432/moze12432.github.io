import streamlit as st
from transformers import pipeline
import torch
import random
import requests
import json
import re
from datetime import datetime, timedelta
import pytz
import calendar

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
    return """I am an advanced AI assistant created by Moses, a student at KyungDong University. 
    I can help you with real-time information (date, time, weather), calculations, translations, 
    factual questions, and emotional support. Think of me as your personal AI companion! 
    What would you like to know or talk about today?"""

# -------------------------
# REAL-TIME INFORMATION FUNCTIONS
# -------------------------

def get_current_date_time():
    """Get current date and time"""
    now = datetime.now()
    return {
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "full": now.strftime("%A, %B %d, %Y at %I:%M %p")
    }

def get_current_timezone():
    """Get timezone info"""
    return str(datetime.now().astimezone().tzinfo)

def calculate(expression):
    """Safely calculate mathematical expressions"""
    try:
        # Remove any dangerous characters
        expression = re.sub(r'[^0-9+\-*/().% ]', '', expression)
        result = eval(expression)
        return f"{expression} = {result}"
    except:
        return None

def get_weather(city="London"):
    """Get weather using free API"""
    try:
        # Using wttr.in - free weather service
        url = f"https://wttr.in/{city}?format=%C+%t+%w+%h"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return f"Weather in {city}: {response.text}"
    except:
        pass
    return None

def get_news():
    """Get top news headlines using free API"""
    try:
        # Using GNews API (free, no key required for basic)
        url = "https://gnews.io/api/v4/top-headlines?lang=en&country=us&max=5&token=demo"  # Demo token works for basic
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])[:3]
            if articles:
                news_list = []
                for i, article in enumerate(articles, 1):
                    title = article.get("title", "")
                    news_list.append(f"{i}. {title}")
                return "\n".join(news_list)
    except:
        pass
    return None

def get_joke():
    """Get a random joke"""
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "What do you call a fake noodle? An impasta!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What do you call a bear with no teeth? A gummy bear!",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "What do you call a fish wearing a bowtie? Sofishticated!",
        "Why did the math book look so sad? Because it had too many problems!"
    ]
    return random.choice(jokes)

# -------------------------
# COMPREHENSIVE KNOWLEDGE BASE
# -------------------------
KNOWLEDGE_BASE = {
    # Capitals
    "capital of south korea": "Seoul is the capital of South Korea.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "capital of china": "Beijing is the capital of China.",
    "capital of india": "New Delhi is the capital of India.",
    "capital of usa": "Washington, D.C. is the capital of the United States.",
    "capital of uk": "London is the capital of the United Kingdom.",
    "capital of france": "Paris is the capital of France.",
    "capital of germany": "Berlin is the capital of Germany.",
    "capital of italy": "Rome is the capital of Italy.",
    "capital of spain": "Madrid is the capital of Spain.",
    "capital of russia": "Moscow is the capital of Russia.",
    "capital of brazil": "Brasília is the capital of Brazil.",
    "capital of canada": "Ottawa is the capital of Canada.",
    "capital of australia": "Canberra is the capital of Australia.",
    "capital of egypt": "Cairo is the capital of Egypt.",
    "capital of turkey": "Ankara is the capital of Turkey.",
    
    # Populations (updated)
    "population of uganda": "Uganda has a population of approximately 48 million people (2024 estimate).",
    "population of japan": "Japan has a population of approximately 124 million people (2024 estimate).",
    "population of china": "China has a population of approximately 1.4 billion people.",
    "population of india": "India has a population of approximately 1.4 billion people.",
    "population of usa": "The United States has a population of approximately 335 million people.",
    
    # Science
    "what is ai": "Artificial Intelligence (AI) is the simulation of human intelligence in machines.",
    "what is machine learning": "Machine learning is a subset of AI that allows systems to learn from data.",
    "what is deep learning": "Deep learning is a subset of machine learning using neural networks.",
    "what is python": "Python is a high-level programming language known for its simplicity and readability.",
    
    # History
    "who is einstein": "Albert Einstein was a theoretical physicist who developed the theory of relativity.",
    "who is newton": "Isaac Newton was a physicist and mathematician who discovered gravity.",
    "who is tesla": "Nikola Tesla was an inventor known for his work with alternating current electricity.",
    
    # Math
    "what is pi": "Pi (π) is approximately 3.14159, the ratio of a circle's circumference to its diameter.",
    "what is euler": "Euler's number (e) is approximately 2.71828, the base of natural logarithms.",
}

def search_knowledge_base(query):
    """Search knowledge base for answers"""
    query_lower = query.lower().strip()
    
    # Direct match
    for key, value in KNOWLEDGE_BASE.items():
        if key in query_lower:
            return value
    
    return None

# -------------------------
# COUNTRY DATA API
# -------------------------
def get_country_info(query):
    """Get comprehensive country information"""
    try:
        # Extract country name
        country_match = re.search(r'(?:population|capital|about|information on|tell me about)\s+(\w+)', query.lower())
        if country_match:
            country = country_match.group(1)
        else:
            # Common country names
            countries = ['uganda', 'japan', 'china', 'india', 'usa', 'uk', 'france', 'germany', 
                        'italy', 'spain', 'brazil', 'canada', 'australia', 'russia', 'mexico', 
                        'south korea', 'north korea', 'vietnam', 'thailand', 'indonesia']
            for c in countries:
                if c in query.lower():
                    country = c
                    break
            else:
                return None
        
        # REST Countries API
        url = f"https://restcountries.com/v3.1/name/{country.replace(' ', '%20')}"
        response = requests.get(url, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            if data:
                country_data = data[0]
                name = country_data.get('name', {}).get('common', country.capitalize())
                capital = country_data.get('capital', ['Unknown'])[0]
                population = country_data.get('population', 0)
                region = country_data.get('region', 'Unknown')
                subregion = country_data.get('subregion', '')
                area = country_data.get('area', 0)
                currency = list(country_data.get('currencies', {}).keys())
                currency = currency[0] if currency else 'Unknown'
                language = list(country_data.get('languages', {}).values())
                language = language[0] if language else 'Unknown'
                
                # Format response based on what was asked
                if 'population' in query.lower():
                    return f"{name} has a population of {population:,} people."
                elif 'capital' in query.lower():
                    return f"The capital of {name} is {capital}."
                else:
                    return f"{name}: Capital: {capital}, Population: {population:,}, Region: {region}, Area: {area:,} km², Currency: {currency}, Language: {language}"
    except:
        pass
    return None

# -------------------------
# SMART RESPONSE GENERATION
# -------------------------
def generate_smart_response(user_input):
    """Generate intelligent response for any query"""
    
    user_lower = user_input.lower()
    
    # 1. Identity
    if any(q in user_lower for q in ["who are you", "what are you", "your creator"]):
        return get_ai_identity()
    
    # 2. Date/Time
    if any(q in user_lower for q in ["what is the date", "today's date", "what day is it", "current date"]):
        date_info = get_current_date_time()
        return f"Today is {date_info['date']}. The current time is {date_info['time']}."
    
    if any(q in user_lower for q in ["what time is it", "current time"]):
        date_info = get_current_date_time()
        return f"The current time is {date_info['time']}."
    
    # 3. Weather
    if "weather" in user_lower:
        city_match = re.search(r'weather in (\w+)', user_lower)
        city = city_match.group(1) if city_match else "London"
        weather = get_weather(city)
        if weather:
            return weather
        return f"I couldn't get weather for {city}. Please try again."
    
    # 4. News
    if "news" in user_lower or "headlines" in user_lower:
        news = get_news()
        if news:
            return f"Here are the top news headlines:\n{news}"
        return "I couldn't fetch news at the moment. Please try again."
    
    # 5. Jokes
    if any(q in user_lower for q in ["joke", "funny", "make me laugh"]):
        return get_joke()
    
    # 6. Calculations
    calc_match = re.search(r'(\d+\s*[\+\-\*/\%\^]\s*\d+)', user_lower)
    if calc_match or any(q in user_lower for q in ["calculate", "what is", "equals"]):
        # Extract calculation
        calc = re.sub(r'[^0-9+\-*/().% ]', '', user_lower)
        if calc and any(op in calc for op in ['+', '-', '*', '/', '%']):
            result = calculate(calc)
            if result:
                return result
    
    # 7. Knowledge Base
    kb_answer = search_knowledge_base(user_input)
    if kb_answer:
        return kb_answer
    
    # 8. Country Information (REST Countries API)
    country_info = get_country_info(user_input)
    if country_info:
        return country_info
    
    # 9. Greetings
    if any(g in user_lower for g in ["hi", "hello", "hey", "greetings"]):
        greetings = [
            "Hello! How are you feeling today?",
            "Hi there! 😊 What can I help you with?",
            "Hey! How's your day going?"
        ]
        return random.choice(greetings)
    
    # 10. Personal memory
    if "my name is" in user_lower or "call me" in user_lower:
        name_match = re.search(r'(?:my name is|call me)\s+(\w+)', user_lower)
        if name_match:
            name = name_match.group(1).capitalize()
            st.session_state.memory["user_facts"]["name"] = name
            return f"Nice to meet you, {name}! I'll remember that. How can I help you today?"
    
    if "what is my name" in user_lower:
        if "name" in st.session_state.memory["user_facts"]:
            return f"Your name is {st.session_state.memory['user_facts']['name']}!"
        return "You haven't told me your name yet. What should I call you?"
    
    # 11. Thank you
    if "thank" in user_lower:
        return "You're very welcome! 😊 Is there anything else I can help you with?"
    
    # 12. Emotional responses
    if any(em in user_lower for em in ["feel", "feeling", "sad", "happy", "angry", "scared", "tired", "sick"]):
        return get_emotional_response(user_input)
    
    # 13. General knowledge question
    if user_lower.endswith('?') or any(qw in user_lower.split()[:1] for qw in ["what", "who", "where", "when", "why", "how"]):
        # Try to answer using general knowledge
        try:
            # Use a simple response for common questions
            if "deepseek" in user_lower or "chatgpt" in user_lower or "gemini" in user_lower:
                return "I'm designed to be like those AI assistants! I can answer questions, provide information, and have natural conversations. What would you like to know?"
            
            if "date today" in user_lower:
                date_info = get_current_date_time()
                return f"Today is {date_info['date']}."
            
            return f"That's a good question! I'm still learning, but I can tell you that {user_input[:50]}... Could you rephrase or ask something more specific?"
        except:
            pass
    
    # 14. Default response
    return "I'm here to help! You can ask me about the date, time, weather, news, jokes, calculations, country information, or just talk to me. What would you like to know?"

def get_emotional_response(user_input):
    """Generate empathetic emotional response"""
    user_lower = user_input.lower()
    
    if "sick" in user_lower or "ill" in user_lower:
        return "I'm sorry you're not feeling well. Please take care of yourself, rest, and stay hydrated. Do you need anything?"
    
    if "tired" in user_lower:
        return "I hear that you're tired. Is there any way you can take a short break or rest? Your well-being is important."
    
    if "sad" in user_lower:
        return "I'm sorry you're feeling sad. It's okay to have these feelings. Would you like to talk about what's bothering you?"
    
    if "happy" in user_lower or "good" in user_lower:
        return "That's wonderful to hear! 😊 What's making you happy today?"
    
    if "angry" in user_lower or "mad" in user_lower:
        return "I hear your frustration. It's okay to feel angry. Would you like to talk about what's bothering you?"
    
    if "scared" in user_lower or "fear" in user_lower:
        return "I understand being scared. You're not alone. What's worrying you? Sometimes talking about it helps."
    
    if "stressed" in user_lower:
        return "Stress can be overwhelming. Take a deep breath. What's one small thing that might help right now?"
    
    # Default emotional response
    return "I'm here for you. How can I support you right now?"

# -------------------------
# UI Components
# -------------------------
def reset_chat():
    st.session_state.messages = []
    st.session_state.emotion_history = []
    st.session_state.memory = {"user_facts": {}, "qa_pairs": []}
    st.rerun()

# -------------------------
# Main UI
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University • Smart AI Assistant</p>", unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### ✨ Features")
    st.markdown("✅ **Real-time date & time**")
    st.markdown("✅ **Weather information**")
    st.markdown("✅ **News headlines**")
    st.markdown("✅ **Math calculations**")
    st.markdown("✅ **Country data & populations**")
    st.markdown("✅ **Jokes & entertainment**")
    st.markdown("✅ **Emotion understanding**")
    st.markdown("✅ **Remembers your name**")
    
    st.divider()
    
    if st.button("🔄 Start New Chat", use_container_width=True):
        reset_chat()
    
    st.divider()
    st.info("💡 **Try asking:**\n• What is the date today?\n• What's the weather in Seoul?\n• Tell me a joke\n• What is 25 * 4?\n• Population of Japan\n• I feel tired")

# Dashboard toggle
if st.button("📊 Show / Hide Emotional Insights"):
    st.session_state.show_dashboard = not st.session_state.show_dashboard

if st.session_state.show_dashboard and st.session_state.emotion_history:
    st.subheader("📊 Emotional Insights")
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
user_input = st.chat_input("Ask me anything! I know the date, time, weather, and more...")

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
        "emotion": emotion, "valence": v, "arousal": a, "dominance": d
    })
    
    # Generate smart response
    with st.spinner("Thinking..."):
        reply = generate_smart_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
