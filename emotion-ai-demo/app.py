import streamlit as st
from transformers import pipeline
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
        "conversation_context": []
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
# REAL INTERNET SEARCH - MULTIPLE SOURCES
# -------------------------

def search_wikipedia(query):
    """Search Wikipedia for comprehensive information"""
    try:
        # Clean query for Wikipedia
        search_query = query.replace("what would happen if", "").replace("what is", "").replace("how to", "").strip()
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(search_query)}"
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            data = response.json()
            if "extract" in data:
                return data["extract"]
    except:
        pass
    return None

def search_duckduckgo(query):
    """Search DuckDuckGo for answers"""
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for abstract
            if data.get("AbstractText"):
                return data["AbstractText"]
            
            # Check for definition
            if data.get("Definition"):
                return data["Definition"]
            
            # Check for answer
            if data.get("Answer"):
                return data["Answer"]
            
            # Check related topics
            if data.get("RelatedTopics") and len(data["RelatedTopics"]) > 0:
                first = data["RelatedTopics"][0]
                if isinstance(first, dict) and first.get("Text"):
                    return first["Text"]
    except:
        pass
    return None

def search_google_custom(query):
    """Use a free search API (simulated with DuckDuckGo HTML for comprehensive results)"""
    try:
        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=10, headers=headers)
        
        if response.status_code == 200:
            # Extract snippets from HTML
            snippets = re.findall(r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>', response.text, re.DOTALL)
            if snippets:
                # Clean HTML tags
                clean = re.sub(r'<[^>]+>', '', snippets[0])
                return clean[:500]
    except:
        pass
    return None

def search_news_api(query):
    """Search for news about the query"""
    try:
        url = f"https://gnews.io/api/v4/search?q={urllib.parse.quote(query)}&lang=en&max=3&token=demo"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            if articles:
                result = []
                for article in articles[:2]:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    if description:
                        result.append(f"• {title}: {description[:200]}")
                    else:
                        result.append(f"• {title}")
                return "\n".join(result)
    except:
        pass
    return None

def search_scientific(query):
    """Search for scientific/hypothetical questions"""
    try:
        # For hypothetical questions like "what would happen if"
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for abstract
            if data.get("AbstractText"):
                return data["AbstractText"]
            
            # Check infobox
            if data.get("Infobox"):
                infobox = data["Infobox"]
                if "content" in infobox:
                    for content in infobox["content"]:
                        if "value" in content:
                            return content["value"]
    except:
        pass
    return None

def comprehensive_search(query):
    """Combine multiple search methods for comprehensive results"""
    
    # First try Wikipedia (best for factual/scientific questions)
    result = search_wikipedia(query)
    if result and len(result) > 50:
        return result
    
    # Try DuckDuckGo API
    result = search_duckduckgo(query)
    if result and len(result) > 30:
        return result
    
    # Try scientific/hypothetical search
    result = search_scientific(query)
    if result and len(result) > 30:
        return result
    
    # Try news search for current events
    result = search_news_api(query)
    if result and len(result) > 30:
        return result
    
    # Final fallback with HTML search
    result = search_google_custom(query)
    if result and len(result) > 30:
        return result
    
    return None

# -------------------------
# CODE GENERATION
# -------------------------
def generate_code(instruction):
    """Generate code based on user instruction"""
    code_templates = {
        "python": {
            "hello world": "print('Hello, World!')",
            "fibonacci": """def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=' ')
        a, b = b, a + b
    print()
    
fibonacci(10)""",
            "sort": """def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

numbers = [64, 34, 25, 12, 22, 11, 90]
print(f"Sorted: {bubble_sort(numbers)}")"""
        }
    }
    
    instruction_lower = instruction.lower()
    
    if "python" in instruction_lower:
        if "hello" in instruction_lower:
            return "```python\n" + code_templates["python"]["hello world"] + "\n```"
        elif "fibonacci" in instruction_lower:
            return "```python\n" + code_templates["python"]["fibonacci"] + "\n```"
        elif "sort" in instruction_lower:
            return "```python\n" + code_templates["python"]["sort"] + "\n```"
        else:
            return "```python\n# Here's a basic Python template\nprint('Your code here')\n```"
    
    return None

# -------------------------
# SMART RESPONSE WITH INTERNET SEARCH
# -------------------------
def generate_smart_response(user_input):
    """Generate response with real internet search"""
    
    user_lower = user_input.lower()
    
    # 1. Identity
    if any(q in user_lower for q in ["who are you", "what are you", "your creator"]):
        return """I am an advanced AI assistant created by Moses, a student at KyungDong University. 
        I can search the internet in real-time, generate code, answer complex questions, 
        understand emotions, and help you with almost any task. What would you like to know?"""
    
    # 2. Date/Time
    if any(q in user_lower for q in ["date today", "today's date", "what day is it", "current date"]):
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}. The time is {now.strftime('%I:%M %p')}."
    
    # 3. Code generation
    if any(q in user_lower for q in ["write code", "generate code", "python code", "function for"]):
        code = generate_code(user_input)
        if code:
            return f"Here's the code you requested:\n\n{code}\n\nYou can copy and run this code. Let me know if you need modifications!"
        return "I can help you write code! What programming language and what would you like the code to do?"
    
    # 4. Calculations
    calc_match = re.search(r'(\d+\s*[\+\-\*/%]\s*\d+)', user_input)
    if calc_match:
        try:
            calc = re.sub(r'[^0-9+\-*/().% ]', '', user_input)
            result = eval(calc)
            return f"{calc} = {result}"
        except:
            pass
    
    # 5. SEARCH THE INTERNET FOR EVERYTHING ELSE
    with st.spinner("🌐 Searching the internet for accurate information..."):
        search_result = comprehensive_search(user_input)
        
        if search_result:
            # Format the response nicely
            response = f"🔍 **I searched the internet and found:**\n\n{search_result}\n\n"
            
            # Add context for hypothetical questions
            if "what would happen if" in user_lower:
                response += "\n💡 This is a hypothetical scenario. The actual outcome might vary based on many factors."
            
            return response
        
        # 6. Emotional response if search fails and it's emotional
        if any(em in user_lower for em in ["feel", "feeling", "sad", "happy", "angry", "tired", "sick"]):
            return get_emotional_response(user_input)
        
        # 7. General help
        if "help" in user_lower or "what can you do" in user_lower:
            return """I can help you with:

🌐 **Internet Search** - Ask me anything! I'll search the web for accurate answers
💻 **Code Generation** - "Write Python code to sort a list"
📅 **Date & Time** - "What's the date today?"
🧮 **Calculations** - "What is 25 * 4?"
💭 **Emotional Support** - "I'm feeling sad" or "I'm stressed"
🔍 **Research** - "What would happen if the moon disappeared?"
📰 **Current Events** - "What's the latest news about AI?"

What would you like to know? Just ask naturally!"""
        
        # 8. Default response
        return f"I searched the internet but couldn't find specific information about '{user_input[:50]}...'. Could you rephrase your question or ask something else? I can search for current events, scientific concepts, historical facts, and more!"

def get_emotional_response(user_input):
    """Generate empathetic response"""
    user_lower = user_input.lower()
    
    if "sick" in user_lower:
        return "I'm sorry you're not feeling well. Please rest and take care of yourself. Do you need anything?"
    
    if "tired" in user_lower:
        return "I hear that you're tired. Can you take a short break or get some rest?"
    
    if "sad" in user_lower:
        return "I'm sorry you're feeling sad. Would you like to talk about what's bothering you?"
    
    if "happy" in user_lower or "good" in user_lower:
        return "That's wonderful to hear! 😊 What's making you happy today?"
    
    if "angry" in user_lower:
        return "I hear your frustration. Would you like to talk about what's bothering you?"
    
    if "stressed" in user_lower:
        return "Stress can be overwhelming. Take a deep breath. What's one small thing that might help?"
    
    return "I'm here for you. How can I support you right now?"

# -------------------------
# UI Components
# -------------------------
def reset_chat():
    st.session_state.messages = []
    st.session_state.emotion_history = []
    st.session_state.memory = {"user_facts": {}, "conversation_context": []}
    st.rerun()

# -------------------------
# Main UI
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University • Real Internet Search</p>", unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### ✨ Capabilities")
    st.markdown("🌐 **Real Internet Search** - Searches Wikipedia, DuckDuckGo, and news sources")
    st.markdown("💻 **Code Generation** - Write Python code on demand")
    st.markdown("📅 **Real-time Date & Time**")
    st.markdown("🧮 **Math Calculations**")
    st.markdown("💭 **Emotion Understanding**")
    st.markdown("🔍 **Answers complex questions**")
    
    st.divider()
    
    if st.button("🔄 Start New Chat", use_container_width=True):
        reset_chat()
    
    st.divider()
    st.info("💡 **Try asking:**\n• What would happen if the Moon disappeared?\n• Write Python code to calculate factorial\n• What is the latest news about AI?\n• I'm feeling stressed\n• What is 156 * 23?")

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
user_input = st.chat_input("Ask me anything! I can search the internet for any question...")

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
    
    if len(st.session_state.emotion_history) > 50:
        st.session_state.emotion_history.pop(0)
    
    # Generate response
    reply = generate_smart_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
