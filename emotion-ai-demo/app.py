import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import json
import requests
from datetime import datetime
import hashlib

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
    model_name = "google/flan-t5-small"
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

# Memory storage (learning)
if "memory" not in st.session_state:
    st.session_state.memory = {
        "user_facts": {},  # Stores facts about the user
        "conversation_summaries": [],  # Stores summaries of past conversations
        "preferences": {},  # Stores user preferences
        "qa_pairs": []  # Stores question-answer pairs for learning
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
# Internet Search Function (using DuckDuckGo HTML API - free)
# -------------------------
def search_web(query):
    """Search the web for information (free, no API key needed)"""
    try:
        # Using DuckDuckGo's lite version (no API key required)
        url = f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            # Simple extraction - get first result
            lines = response.text.split('\n')
            for i, line in enumerate(lines):
                if 'Result' in line and i+1 < len(lines):
                    result = lines[i+1].strip()
                    if result and len(result) > 20:
                        return result[:500]  # Return first 500 chars
        return None
    except Exception as e:
        return None

# -------------------------
# Memory/Learning Functions
# -------------------------
def extract_user_facts(user_input):
    """Extract and store facts about the user"""
    user_lower = user_input.lower()
    
    # Patterns to extract user information
    patterns = {
        "name": ["my name is", "i'm ", "i am ", "call me "],
        "age": ["i am ", "years old", "age "],
        "job": ["i work as", "my job is", "i'm a "],
        "hobby": ["i like", "i enjoy", "my hobby is", "i love to"],
        "feeling": ["i feel", "i'm feeling", "i am feeling"]
    }
    
    facts_extracted = {}
    
    for fact_type, triggers in patterns.items():
        for trigger in triggers:
            if trigger in user_lower:
                # Extract the fact
                parts = user_input.split(trigger, 1)
                if len(parts) > 1:
                    fact = parts[1].strip().split('.')[0][:100]
                    facts_extracted[fact_type] = fact
                    break
    
    # Store in memory
    for key, value in facts_extracted.items():
        st.session_state.memory["user_facts"][key] = value
    
    return facts_extracted

def store_qa_pair(question, answer):
    """Store question-answer pairs for learning"""
    # Check if similar question already exists
    q_hash = hashlib.md5(question.lower().encode()).hexdigest()
    
    for qa in st.session_state.memory["qa_pairs"]:
        if qa["hash"] == q_hash:
            qa["answer"] = answer
            qa["asked_count"] += 1
            qa["last_asked"] = datetime.now().isoformat()
            return
    
    # Add new QA pair
    st.session_state.memory["qa_pairs"].append({
        "hash": q_hash,
        "question": question,
        "answer": answer,
        "asked_count": 1,
        "first_asked": datetime.now().isoformat(),
        "last_asked": datetime.now().isoformat()
    })
    
    # Keep only last 100 pairs
    if len(st.session_state.memory["qa_pairs"]) > 100:
        st.session_state.memory["qa_pairs"] = st.session_state.memory["qa_pairs"][-100:]

def recall_from_memory(user_input):
    """Try to recall if this question was asked before"""
    user_lower = user_input.lower()
    
    for qa in st.session_state.memory["qa_pairs"]:
        if qa["question"].lower() in user_lower or user_lower in qa["question"].lower():
            return qa["answer"]
    return None

def needs_search(user_input):
    """Determine if the query needs internet search"""
    search_triggers = [
        "what is", "who is", "when did", "where is", "how to",
        "latest", "news", "weather", "today", "current",
        "tell me about", "explain", "define", "meaning of"
    ]
    
    user_lower = user_input.lower()
    return any(trigger in user_lower for trigger in search_triggers)

def is_joke_request(user_input):
    """Check if user wants a joke"""
    joke_triggers = ["tell me a joke", "make me laugh", "joke", "funny", "humor"]
    return any(trigger in user_input.lower() for trigger in joke_triggers)

def get_joke():
    """Get a joke (pre-loaded or from API)"""
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "What do you call a fake noodle? An impasta!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What do you call a bear with no teeth? A gummy bear!",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "What do you call a fish wearing a bowtie? Sofishticated!",
        "Why did the math book look so sad? Because it had too many problems!",
        "What's the best thing about Switzerland? I don't know, but the flag is a big plus!"
    ]
    return random.choice(jokes)

# -------------------------
# Enhanced Response Generation
# -------------------------
def generate_enhanced_response(user_input, emotion, v, a, d):
    """Generate response with memory and search capabilities"""
    
    user_lower = user_input.lower()
    
    # 1. Check for joke request
    if is_joke_request(user_input):
        return get_joke()
    
    # 2. Try to recall from memory
    recalled = recall_from_memory(user_input)
    if recalled:
        return f"I remember you asked something similar before: {recalled}"
    
    # 3. Check if internet search is needed
    if needs_search(user_input):
        with st.spinner("Searching the web..."):
            search_result = search_web(user_input)
            if search_result:
                # Use FLAN to summarize the search result
                try:
                    prompt = f"""Based on this information: "{search_result}"
                    Answer this question concisely: {user_input}
                    Keep answer to 1-2 sentences."""
                    
                    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=300)
                    with torch.no_grad():
                        outputs = flan_model.generate(
                            inputs.input_ids,
                            max_length=100,
                            num_beams=4,
                            temperature=0.7,
                            pad_token_id=flan_tokenizer.eos_token_id
                        )
                    response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    if response and len(response) > 5:
                        # Store this QA pair for future
                        store_qa_pair(user_input, response)
                        return response
                except:
                    return f"I found this: {search_result[:300]}..."
    
    # 4. Extract and store user facts
    facts = extract_user_facts(user_input)
    if facts:
        # Acknowledge that we learned something
        learned = f"Thanks for telling me! I'll remember that. "
        # Add the response
        response = get_empathetic_response(user_input, emotion, facts)
        store_qa_pair(user_input, response)
        return learned + response
    
    # 5. Regular empathetic response
    response = get_empathetic_response(user_input, emotion)
    store_qa_pair(user_input, response)
    return response

def get_empathetic_response(user_input, emotion, facts=None):
    """Generate empathetic response based on emotion and context"""
    
    user_lower = user_input.lower()
    emotion = emotion.lower()
    
    # Use remembered facts if available
    if facts:
        if "name" in facts:
            return f"Nice to meet you, {facts['name']}! How can I help you today?"
        if "feeling" in facts:
            return f"I hear that you're {facts['feeling']}. Thank you for sharing that with me."
    
    # Check for specific keywords
    if any(word in user_lower for word in ["sick", "ill", "pain", "hurt"]):
        return "I'm sorry you're not feeling well. Being sick is really tough. Make sure you rest and stay hydrated. Do you need anything?"
    
    if any(word in user_lower for word in ["tired", "exhausted", "sleep", "fatigue"]):
        return "I hear that you're tired. Lack of rest really affects how we feel. Is there any way you can take a short break or rest right now?"
    
    if any(word in user_lower for word in ["stressed", "overwhelmed", "anxious"]):
        return "That sounds really stressful. Remember to breathe deeply and take things one step at a time. What's one small thing that might help right now?"
    
    if any(word in user_lower for word in ["lonely", "alone", "isolated"]):
        return "I'm here with you. Feeling lonely is hard, but you're not alone in this moment. Would you like to talk about what's on your mind?"
    
    if any(word in user_lower for word in ["angry", "mad", "frustrated", "annoyed"]):
        return "I can hear your frustration. It's okay to feel angry. Would you like to talk about what's bothering you?"
    
    if any(word in user_lower for word in ["happy", "good", "great", "wonderful", "amazing"]):
        return "That's wonderful to hear! 😊 I'm glad you're feeling good. What's making you happy today?"
    
    if "who are you" in user_lower:
        return "I'm your Emotion-Aware AI Companion! I'm designed to understand your feelings and have meaningful conversations with you. I can learn from our chats and even search the internet for information. How can I help you today?"
    
    if "thank" in user_lower:
        return "You're very welcome! 😊 I'm glad I could help. Is there anything else you'd like to talk about?"
    
    # Emotion-based responses
    emotion_responses = {
        "sadness": [
            f"I hear that you're feeling down. I'm sorry you're going through this. Would you like to talk more about what's making you feel this way?",
            f"That sounds really difficult. It's okay to feel sad sometimes. I'm here to listen if you want to share more."
        ],
        "joy": [
            f"That's great to hear! 😊 What's been going well for you?",
            f"I love that you're feeling joyful! Tell me more about what's bringing you happiness."
        ],
        "anger": [
            f"That sounds frustrating. It's valid to feel angry. Would you like to tell me what happened?",
            f"I can hear that you're upset. Sometimes venting helps - I'm here to listen."
        ],
        "fear": [
            f"That sounds worrying. I'm here with you. What's the main thing that's concerning you?",
            f"Fear can be really overwhelming. You're not alone in this."
        ],
        "surprise": [
            f"That's interesting! How are you processing this?",
            f"Life definitely has its surprises! How do you feel about this?"
        ],
        "love": [
            f"That's beautiful to hear. Love makes everything feel brighter. Tell me more!",
            f"I'm so glad you're experiencing love. It's such a special feeling."
        ],
        "neutral": [
            f"I appreciate you sharing that with me. How can I support you today?",
            f"I hear you. Is there anything specific you'd like to talk about?"
        ]
    }
    
    responses = emotion_responses.get(emotion, emotion_responses["neutral"])
    return random.choice(responses)

# -------------------------
# Memory Dashboard
# -------------------------
def show_memory_dashboard():
    """Display what the AI has learned"""
    st.subheader("🧠 What I've Learned About You")
    
    if st.session_state.memory["user_facts"]:
        st.write("**Things you've told me:**")
        for key, value in st.session_state.memory["user_facts"].items():
            st.write(f"- Your {key}: {value}")
    else:
        st.write("I haven't learned much about you yet. Tell me about yourself!")
    
    if st.session_state.memory["qa_pairs"]:
        st.write("**Questions I remember:**")
        for qa in st.session_state.memory["qa_pairs"][-5:]:  # Show last 5
            st.write(f"- Asked {qa['asked_count']} time(s): {qa['question'][:50]}...")

# -------------------------
# UI Header
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>With Memory & Internet Search</p>", unsafe_allow_html=True)

st.divider()

# -------------------------
# Sidebar with Memory Info
# -------------------------
with st.sidebar:
    st.markdown("### 📚 AI Capabilities")
    st.markdown("✅ **Remembers** past conversations")
    st.markdown("✅ **Learns** from our chats")
    st.markdown("✅ **Searches** the internet (when needed)")
    st.markdown("✅ **Understands** your emotions")
    
    st.divider()
    show_memory_dashboard()
    
    if st.button("🗑️ Clear Memory"):
        st.session_state.memory = {
            "user_facts": {},
            "conversation_summaries": [],
            "preferences": {},
            "qa_pairs": []
        }
        st.rerun()

# -------------------------
# Dashboard Toggle
# -------------------------
if st.button("📊 Show / Hide Emotional Insights"):
    st.session_state.show_dashboard = not st.session_state.show_dashboard

# -------------------------
# Dashboard
# -------------------------
if st.session_state.show_dashboard and st.session_state.emotion_history:
    st.subheader("📊 Emotional Insights")
    history_data = st.session_state.emotion_history

    emotion_counts = {}
    for item in history_data:
        e = item["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    st.bar_chart(emotion_counts)

    vad_data = {
        "Valence": [h["valence"] for h in history_data],
        "Arousal": [h["arousal"] for h in history_data],
        "Dominance": [h["dominance"] for h in history_data]
    }
    st.line_chart(vad_data)

    avg_valence = sum(h["valence"] for h in history_data) / len(history_data)
    avg_arousal = sum(h["arousal"] for h in history_data) / len(history_data)

    if avg_valence > 0.6:
        mood = "generally positive"
    elif avg_valence < 0.4:
        mood = "often experiencing negative emotions"
    else:
        mood = "emotionally balanced"

    if avg_arousal > 0.6:
        energy = "high emotional intensity"
    else:
        energy = "calm emotional state"

    st.info(f"Over time: {mood} • {energy}")
    
    if emotion_counts:
        most_common = max(emotion_counts, key=emotion_counts.get)
        st.metric("Most frequent emotion", most_common.capitalize())

st.divider()

# -------------------------
# Display Chat
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# Chat Input
# -------------------------
user_input = st.chat_input("How are you feeling today? Or ask me anything!")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Emotion Detection
    try:
        raw = emotion_model(user_input)
        emotions = raw[0] if isinstance(raw[0], list) else raw
        top = max(emotions, key=lambda x: x.get("score", 0))
        emotion = top.get("label", "neutral")
        confidence = round(top.get("score", 0) * 100, 2)
    except:
        emotion = "neutral"
        confidence = 0.0
    
    # VAD Calculation
    v, a, d = VAD_MAP.get(emotion.lower(), (0.5, 0.5, 0.5))
    
    # Save Emotion History
    st.session_state.emotion_history.append({
        "emotion": emotion,
        "confidence": confidence,
        "valence": v,
        "arousal": a,
        "dominance": d
    })
    
    if len(st.session_state.emotion_history) > 20:
        st.session_state.emotion_history.pop(0)
    
    # Generate Enhanced Response
    with st.spinner("Thinking..."):
        reply = generate_enhanced_response(user_input, emotion, v, a, d)
    
    # Show Response
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    # Save response
    st.session_state.messages.append({"role": "assistant", "content": reply})
