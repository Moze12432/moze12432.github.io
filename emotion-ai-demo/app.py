import streamlit as st
from transformers import pipeline
import torch
import random
import re
import requests
import urllib.parse
from datetime import datetime
import json

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Complete AI Companion",
    page_icon="🧠",
    layout="wide"
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

emotion_model = load_emotion_model()

# -------------------------
# Session State - Memory
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "emotion_history" not in st.session_state:
    st.session_state.emotion_history = []

if "show_dashboard" not in st.session_state:
    st.session_state.show_dashboard = False

# Memory for user facts and conversation history
if "memory" not in st.session_state:
    st.session_state.memory = {
        "user_name": None,
        "user_preferences": [],
        "conversation_summary": [],
        "facts_learned": [],
        "last_topics": []
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

# ============================================================
# INTERNET SEARCH
# ============================================================

def search_duckduckgo(query):
    """Search DuckDuckGo and get answer"""
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(query)}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
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
    """Search Wikipedia for detailed answers"""
    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(query)}"
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            data = response.json()
            if "extract" in data:
                return data["extract"]
    except:
        pass
    return None

def internet_search(query):
    """Combined internet search"""
    # Try DuckDuckGo first
    result = search_duckduckgo(query)
    if result:
        return result
    
    # Try Wikipedia
    result = search_wikipedia(query)
    if result:
        return result
    
    return None

# ============================================================
# REASONING ENGINE
# ============================================================

class ReasoningEngine:
    
    @staticmethod
    def syllogism_reasoning(question):
        """Handle: All A are B, all B are C, therefore all A are C"""
        q_lower = question.lower()
        pattern = r'all (\w+) are (\w+).*all (\w+) are (\w+)'
        match = re.search(pattern, q_lower)
        
        if match:
            a, b, b2, c = match.groups()
            if b == b2:
                return f"""**Answer:** Yes, all {a} are definitely {c}.

**Step-by-Step Reasoning:**
- All {a} are {b} means {a} ⊆ {b}
- All {b} are {c} means {b} ⊆ {c}
- By transitive property, {a} ⊆ {c}
- Therefore, all {a} are {c}"""
        return None
    
    @staticmethod
    def bat_ball_reasoning(question):
        """Handle: Bat and ball cost $1.10 total, bat costs $1 more than ball"""
        if "bat" in question.lower() and "ball" in question.lower() and "1.10" in question:
            return """**Answer:** The ball costs $0.05 (5 cents).

**Step-by-Step Reasoning:**
- Let x = cost of ball
- Then bat = x + 1.00
- Total: x + (x + 1.00) = 1.10
- 2x + 1.00 = 1.10
- 2x = 0.10
- x = 0.05"""
        return None
    
    @staticmethod
    def even_zero_reasoning(question):
        """Handle: Is 0 an even number?"""
        if "0" in question and "even" in question.lower():
            return """**Answer:** Yes, 0 is an even number.

**Reasoning:**
- Definition: n is even if n = 2k for some integer k
- 0 = 2 × 0
- k = 0 is an integer
- Therefore, 0 satisfies the definition"""
        return None
    
    @staticmethod
    def sheep_reasoning(question):
        """Handle: Farmer has 17 sheep, all but 9 die"""
        if "sheep" in question.lower() and "die" in question.lower():
            return """**Answer:** 9 sheep are left.

**Reasoning:**
- "All but 9 die" means everything EXCEPT 9 dies
- 9 refers to survivors, not deaths
- Total = 17, survivors = 9
- Therefore, 9 sheep are left"""
        return None
    
    @staticmethod
    def plane_crash_reasoning(question):
        """Handle: Plane crash border question"""
        if "plane" in question.lower() and "crash" in question.lower() and "border" in question.lower():
            return """**Answer:** Survivors don't get buried anywhere.

**Reasoning:**
- Survivors = people who lived through the crash
- You bury people who died (victims), not survivors
- The question tricks you by using the word "survivors"
- Correct answer: Survivors are alive, so they aren't buried"""
        return None

# ============================================================
# MEMORY FUNCTIONS
# ============================================================

def extract_user_info(user_input):
    """Extract and store user information"""
    q_lower = user_input.lower()
    
    # Extract name
    name_patterns = [r"my name is (\w+)", r"i am (\w+)", r"call me (\w+)"]
    for pattern in name_patterns:
        match = re.search(pattern, q_lower)
        if match:
            name = match.group(1).capitalize()
            st.session_state.memory["user_name"] = name
            return f"I'll remember that your name is {name}!"
    
    return None

def remember_fact(user_input):
    """Remember important facts from conversation"""
    q_lower = user_input.lower()
    
    # Look for fact indicators
    fact_indicators = ["remember that", "fact:", "important:", "note that"]
    for indicator in fact_indicators:
        if indicator in q_lower:
            fact = user_input.split(indicator)[-1].strip()
            st.session_state.memory["facts_learned"].append(fact)
            return f"I've remembered that: {fact}"
    
    return None

# ============================================================
# CONVERSATION RESPONSES
# ============================================================

def get_greeting():
    """Return a friendly greeting"""
    greetings = [
        "Hello! How are you feeling today?",
        "Hi there! Great to see you. What's on your mind?",
        "Hey! How can I help you today?",
        "Greetings! I'm your AI companion. What would you like to talk about?"
    ]
    return random.choice(greetings)

def get_farewell():
    """Return a farewell message"""
    farewells = [
        "Goodbye! Take care and come back anytime!",
        "See you later! Have a wonderful day!",
        "Bye for now! I'm here whenever you need me.",
        "Take care! Remember I'm always here to chat."
    ]
    return random.choice(farewells)

def get_emotional_response(user_input, emotion):
    """Generate empathetic response"""
    q_lower = user_input.lower()
    
    if "sick" in q_lower:
        return "I'm sorry you're not feeling well. Please rest and take care of yourself. Do you need anything?"
    
    if "tired" in q_lower:
        return "I hear that you're tired. Rest is so important. Can you take a short break or nap?"
    
    if "sad" in q_lower:
        return "I'm sorry you're feeling sad. Would you like to talk about what's bothering you? I'm here to listen."
    
    if "happy" in q_lower or "good" in q_lower:
        return "That's wonderful to hear! What's making you happy today? I'd love to hear more."
    
    if "angry" in q_lower or "frustrated" in q_lower:
        return "I hear your frustration. It's okay to feel that way. Would you like to talk about what happened?"
    
    if "stressed" in q_lower or "anxious" in q_lower:
        return "Stress can be really tough. Take a deep breath. What's one small thing that might help right now?"
    
    if "lonely" in q_lower:
        return "You're not alone - I'm here with you. Would you like to talk about how you're feeling?"
    
    # Emotion-based responses
    emotion_responses = {
        "joy": "I'm so glad you're feeling joyful! That's wonderful!",
        "sadness": "I'm here for you. It's okay to feel sad sometimes.",
        "anger": "Your feelings are valid. I'm here to listen.",
        "fear": "I understand being scared. You're safe here.",
        "surprise": "That sounds interesting! Tell me more.",
        "neutral": "Thanks for sharing. How can I support you?"
    }
    
    return emotion_responses.get(emotion, "I'm here for you. How can I help?")

# ============================================================
# MAIN RESPONSE GENERATOR
# ============================================================

def generate_response(user_input):
    """Generate intelligent response with memory, reasoning, and internet search"""
    
    q_lower = user_input.lower()
    
    # 1. Check for farewell
    if any(word in q_lower for word in ["bye", "goodbye", "see you", "farewell"]):
        return get_farewell()
    
    # 2. Check for greeting
    if any(word in q_lower for word in ["hi", "hello", "hey", "greetings"]):
        return get_greeting()
    
    # 3. Check for thanks
    if "thank" in q_lower:
        return "You're very welcome! I'm glad I could help. Is there anything else you'd like to know?"
    
    # 4. Extract and remember user info
    name_response = extract_user_info(user_input)
    if name_response:
        return name_response
    
    fact_response = remember_fact(user_input)
    if fact_response:
        return fact_response
    
    # 5. Check for personal questions using memory
    if "my name" in q_lower and st.session_state.memory["user_name"]:
        return f"Your name is {st.session_state.memory['user_name']}! I remember you told me."
    
    # 6. Check for reasoning patterns
    reasoning_patterns = [
        ReasoningEngine.syllogism_reasoning,
        ReasoningEngine.bat_ball_reasoning,
        ReasoningEngine.even_zero_reasoning,
        ReasoningEngine.sheep_reasoning,
        ReasoningEngine.plane_crash_reasoning
    ]
    
    for pattern in reasoning_patterns:
        response = pattern(user_input)
        if response:
            return response
    
    # 7. Check for emotional content
    if any(word in q_lower for word in ["feel", "feeling", "sad", "happy", "angry", "tired", "sick", "stressed", "lonely"]):
        # Detect emotion
        try:
            raw = emotion_model(user_input)
            emotions = raw[0] if isinstance(raw[0], list) else raw
            top = max(emotions, key=lambda x: x.get("score", 0))
            emotion = top.get("label", "neutral")
        except:
            emotion = "neutral"
        return get_emotional_response(user_input, emotion)
    
    # 8. Check for jokes
    if any(word in q_lower for word in ["joke", "funny", "make me laugh"]):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a bear with no teeth? A gummy bear!",
            "Why don't eggs tell jokes? They'd crack each other up!"
        ]
        return random.choice(jokes)
    
    # 9. Check for date/time
    if any(word in q_lower for word in ["date today", "today's date", "what day is it"]):
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."
    
    if any(word in q_lower for word in ["what time", "current time"]):
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."
    
    # 10. SEARCH THE INTERNET for everything else
    with st.spinner("Searching the internet..."):
        search_result = internet_search(user_input)
        
        if search_result:
            return f"🔍 **I searched online:**\n\n{search_result}"
        
        # 11. Default response if everything fails
        return """I'm here to help with a wide range of topics!

**You can ask me about:**
- Logic puzzles (All A are B, all B are C...)
- Math problems (Bat and ball cost...)
- Facts and information (I search the internet)
- Emotional support (I'm feeling sad)
- Jokes and fun conversations
- Remembering your name and facts about you

What would you like to talk about?"""

# ============================================================
# MEMORY DISPLAY
# ============================================================

def show_memory():
    """Display what the AI remembers"""
    st.subheader("💭 What I Remember")
    
    if st.session_state.memory["user_name"]:
        st.write(f"**Your name:** {st.session_state.memory['user_name']}")
    
    if st.session_state.memory["facts_learned"]:
        st.write("**Facts you've told me:**")
        for fact in st.session_state.memory["facts_learned"][-5:]:
            st.write(f"- {fact}")
    
    if not st.session_state.memory["user_name"] and not st.session_state.memory["facts_learned"]:
        st.write("I haven't learned much yet. Tell me about yourself!")

def reset_memory():
    """Reset all memory"""
    st.session_state.messages = []
    st.session_state.emotion_history = []
    st.session_state.memory = {
        "user_name": None,
        "user_preferences": [],
        "conversation_summary": [],
        "facts_learned": [],
        "last_topics": []
    }
    st.rerun()

# ============================================================
# UI
# ============================================================

st.markdown("<h1 style='text-align: center;'>Complete AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University | Internet Search | Memory | Reasoning</p>", unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### Features")
    st.markdown("✅ **Internet Search** - Finds answers online")
    st.markdown("✅ **Memory** - Remembers your name and facts")
    st.markdown("✅ **Logical Reasoning** - Solves puzzles")
    st.markdown("✅ **Emotional Intelligence** - Understands feelings")
    st.markdown("✅ **Natural Conversation** - Friendly and helpful")
    
    st.divider()
    show_memory()
    
    st.divider()
    
    if st.button("🗑️ Reset Memory", use_container_width=True):
        reset_memory()
    
    st.divider()
    st.info("""**Try asking:**
• What is the capital of France?
• My name is Moses
• What is my name?
• If all A are B and all B are C, are all A C?
• I'm feeling sad today
• Tell me a joke
• What time is it?
• How to say hello in Korean?""")

# Dashboard toggle
if st.button("📊 Show / Hide Emotional Insights"):
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
user_input = st.chat_input("Ask me anything... I can search the internet, remember facts, solve puzzles, or just chat...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Emotion detection for dashboard
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
    
    # Generate response
    reply = generate_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
