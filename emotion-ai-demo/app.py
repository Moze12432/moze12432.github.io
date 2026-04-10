import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import json
import requests
from datetime import datetime
import hashlib
import re

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
        "qa_pairs": {}  # Question-answer pairs for learning (dictionary for faster lookup)
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
# Common Knowledge Base (built-in to avoid search for basic facts)
# -------------------------
KNOWLEDGE_BASE = {
    "capital of france": "Paris is the capital of France.",
    "france capital": "Paris is the capital of France.",
    "capital of germany": "Berlin is the capital of Germany.",
    "capital of italy": "Rome is the capital of Italy.",
    "capital of spain": "Madrid is the capital of Spain.",
    "capital of uk": "London is the capital of the United Kingdom.",
    "capital of england": "London is the capital of England.",
    "capital of japan": "Tokyo is the capital of Japan.",
    "capital of china": "Beijing is the capital of China.",
    "capital of india": "New Delhi is the capital of India.",
    "capital of usa": "Washington, D.C. is the capital of the United States.",
    "capital of canada": "Ottawa is the capital of Canada.",
    "capital of australia": "Canberra is the capital of Australia.",
    "what is ai": "AI (Artificial Intelligence) is the simulation of human intelligence in machines.",
    "who is einstein": "Albert Einstein was a famous physicist who developed the theory of relativity.",
    "what is python": "Python is a popular programming language known for its simplicity and readability.",
    "what is machine learning": "Machine learning is a subset of AI that allows systems to learn from data.",
}

# -------------------------
# Internet Search Function (free)
# -------------------------
def search_web(query):
    """Search the web for information (free, no API key needed)"""
    try:
        # Using DuckDuckGo's lite version
        url = f"https://lite.duckduckgo.com/lite/?q={query.replace(' ', '+')}"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        
        response = requests.get(url, headers=headers, timeout=5)
        
        if response.status_code == 200:
            # Extract results
            lines = response.text.split('\n')
            results = []
            for i, line in enumerate(lines):
                if 'Result' in line and i+1 < len(lines):
                    result = lines[i+1].strip()
                    if result and len(result) > 20 and not result.startswith('http'):
                        results.append(result)
                        if len(results) >= 2:  # Get top 2 results
                            break
            
            if results:
                return ' | '.join(results[:2])[:500]
        return None
    except Exception as e:
        return None

# -------------------------
# Memory/Learning Functions
# -------------------------
def extract_user_facts(user_input):
    """Extract and store facts about the user"""
    user_lower = user_input.lower()
    facts_extracted = {}
    
    # Name extraction
    name_patterns = [
        r"my name is (\w+)",
        r"i['']?m (\w+)",
        r"call me (\w+)",
        r"i am (\w+)"
    ]
    
    for pattern in name_patterns:
        match = re.search(pattern, user_lower)
        if match:
            name = match.group(1).capitalize()
            facts_extracted["name"] = name
            break
    
    # Age extraction
    age_match = re.search(r"i am (\d+) years? old", user_lower)
    if age_match:
        facts_extracted["age"] = age_match.group(1)
    
    # Store in memory
    for key, value in facts_extracted.items():
        st.session_state.memory["user_facts"][key] = value
    
    return facts_extracted

def store_qa_pair(question, answer):
    """Store question-answer pairs for learning"""
    # Create a normalized key for the question
    normalized_q = question.lower().strip()
    q_hash = hashlib.md5(normalized_q.encode()).hexdigest()
    
    st.session_state.memory["qa_pairs"][q_hash] = {
        "question": question,
        "answer": answer,
        "asked_count": st.session_state.memory["qa_pairs"].get(q_hash, {}).get("asked_count", 0) + 1,
        "last_asked": datetime.now().isoformat()
    }
    
    # Keep only last 200 pairs
    if len(st.session_state.memory["qa_pairs"]) > 200:
        # Remove oldest based on last_asked
        oldest = min(st.session_state.memory["qa_pairs"].items(), 
                    key=lambda x: x[1].get("last_asked", ""))
        del st.session_state.memory["qa_pairs"][oldest[0]]

def recall_from_memory(user_input):
    """Try to recall if this question was asked before"""
    user_lower = user_input.lower().strip()
    
    # Direct match
    q_hash = hashlib.md5(user_lower.encode()).hexdigest()
    if q_hash in st.session_state.memory["qa_pairs"]:
        return st.session_state.memory["qa_pairs"][q_hash]["answer"]
    
    # Partial match (check if any stored question is similar)
    for q_hash, qa in st.session_state.memory["qa_pairs"].items():
        if qa["question"].lower() in user_lower or user_lower in qa["question"].lower():
            if len(qa["question"]) > 5:  # Avoid matching very short questions
                return qa["answer"]
    
    return None

def is_factual_question(user_input):
    """Determine if the query is a factual question (not emotional)"""
    # Question indicators
    question_words = ["what", "who", "where", "when", "why", "how", "which", "is", "are", "do", "does"]
    
    # Emotional words that should be treated as emotional even if they're questions
    emotional_words = ["feel", "feeling", "sad", "happy", "angry", "scared", "lonely", "tired", "sick"]
    
    user_lower = user_input.lower()
    
    # Check if it's a question
    is_question = any(user_lower.startswith(word) for word in question_words) or user_lower.endswith('?')
    
    # Check if it's emotional
    is_emotional = any(word in user_lower for word in emotional_words)
    
    # It's factual if it's a question and not primarily emotional
    return is_question and not is_emotional

def is_personal_question(user_input):
    """Check if user is asking about themselves"""
    personal_patterns = [
        r"what is my name",
        r"who am i",
        r"what do i like",
        r"my name",
        r"remember my",
        r"do you remember"
    ]
    
    user_lower = user_input.lower()
    return any(re.search(pattern, user_lower) for pattern in personal_patterns)

def get_from_knowledge_base(user_input):
    """Check knowledge base for common questions"""
    user_lower = user_input.lower().strip()
    
    # Direct match
    for key, value in KNOWLEDGE_BASE.items():
        if key in user_lower:
            return value
    
    return None

def is_greeting(user_input):
    """Check if user is just greeting"""
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
    return user_input.lower().strip() in greetings or user_input.lower().startswith(tuple(greetings))

def get_greeting_response():
    """Return a friendly greeting"""
    greetings = [
        "Hello! How are you feeling today?",
        "Hi there! 😊 What's on your mind?",
        "Hey! How can I help you today?",
        "Greetings! How are you doing?"
    ]
    return random.choice(greetings)

# -------------------------
# Enhanced Response Generation with Proper Priority
# -------------------------
def generate_enhanced_response(user_input, emotion):
    """Generate response with proper priority: Memory > Knowledge > Search > Emotional"""
    
    # PRIORITY 1: Check if it's just a greeting
    if is_greeting(user_input):
        return get_greeting_response()
    
    # PRIORITY 2: Check personal questions (about the user)
    if is_personal_question(user_input):
        if "name" in user_input.lower() and "my name" in st.session_state.memory["user_facts"]:
            return f"Your name is {st.session_state.memory['user_facts']['name']}! I remember you told me."
        elif "name" in user_input.lower():
            return "I don't think you've told me your name yet. What should I call you?"
        else:
            return "I remember our conversations! What would you like to know?"
    
    # PRIORITY 3: Check knowledge base first (fast, no API)
    kb_answer = get_from_knowledge_base(user_input)
    if kb_answer:
        store_qa_pair(user_input, kb_answer)
        return kb_answer
    
    # PRIORITY 4: Check memory for previously asked questions
    recalled = recall_from_memory(user_input)
    if recalled:
        return f"{recalled}"
    
    # PRIORITY 5: Factual questions get search or FLAN
    if is_factual_question(user_input):
        # Try internet search
        with st.spinner("Searching..."):
            search_result = search_web(user_input)
            if search_result:
                # Use FLAN to formulate a clean answer
                try:
                    prompt = f"""Based on this information: "{search_result}"
                    Answer this question in 1 clear sentence: {user_input}"""
                    
                    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
                    with torch.no_grad():
                        outputs = flan_model.generate(
                            inputs.input_ids,
                            max_length=100,
                            num_beams=4,
                            temperature=0.5,
                            pad_token_id=flan_tokenizer.eos_token_id
                        )
                    response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    
                    if response and len(response) > 5:
                        store_qa_pair(user_input, response)
                        return response
                except:
                    pass
                store_qa_pair(user_input, search_result)
                return search_result
        
        # Try FLAN directly for factual questions
        try:
            prompt = f"""Answer this question concisely in 1 sentence: {user_input}"""
            inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
            with torch.no_grad():
                outputs = flan_model.generate(
                    inputs.input_ids,
                    max_length=80,
                    num_beams=4,
                    temperature=0.5,
                    pad_token_id=flan_tokenizer.eos_token_id
                )
            response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
            if response and len(response) > 5 and len(response) < 150:
                store_qa_pair(user_input, response)
                return response
        except:
            pass
        
        # Fallback for factual questions
        return "I'm not sure about that. Could you rephrase your question?"
    
    # PRIORITY 6: Extract and store user facts
    facts = extract_user_facts(user_input)
    if facts:
        if "name" in facts:
            return f"Nice to meet you, {facts['name']}! How can I help you today?"
    
    # PRIORITY 7: Emotional/Conversational responses
    return get_empathetic_response(user_input, emotion)

def get_empathetic_response(user_input, emotion):
    """Generate empathetic response for emotional content"""
    
    user_lower = user_input.lower()
    emotion = emotion.lower()
    
    # Health-related
    if any(word in user_lower for word in ["sick", "ill", "pain", "hurt", "unwell"]):
        return "I'm sorry you're not feeling well. Being sick is really tough. Make sure you rest and stay hydrated. Do you need anything?"
    
    # Tiredness
    if any(word in user_lower for word in ["tired", "exhausted", "sleep", "fatigue", "worn out"]):
        return "I hear that you're tired. Lack of rest really affects how we feel. Is there any way you can take a short break or rest right now?"
    
    # Stress/Anxiety
    if any(word in user_lower for word in ["stressed", "overwhelmed", "anxious", "worry"]):
        return "That sounds really stressful. Remember to breathe deeply and take things one step at a time. What's one small thing that might help right now?"
    
    # Loneliness
    if any(word in user_lower for word in ["lonely", "alone", "isolated", "nobody"]):
        return "I'm here with you. Feeling lonely is hard, but you're not alone in this moment. Would you like to talk about what's on your mind?"
    
    # Anger/Frustration
    if any(word in user_lower for word in ["angry", "mad", "frustrated", "annoyed", "upset"]):
        return "I can hear your frustration. It's okay to feel angry. Would you like to talk about what's bothering you?"
    
    # Happiness
    if any(word in user_lower for word in ["happy", "good", "great", "wonderful", "amazing", "excited"]):
        return "That's wonderful to hear! 😊 I'm glad you're feeling good. What's making you happy today?"
    
    # Jokes/humor
    if any(word in user_lower for word in ["joke", "funny", "make me laugh", "humor"]):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a bear with no teeth? A gummy bear!"
        ]
        return random.choice(jokes)
    
    # Thanks
    if "thank" in user_lower:
        return "You're very welcome! 😊 Is there anything else I can help you with?"
    
    # Emotion-based responses
    emotion_responses = {
        "sadness": [
            f"I hear that you're feeling down. I'm sorry you're going through this. Would you like to talk more?",
            f"That sounds really difficult. It's okay to feel sad sometimes. I'm here to listen."
        ],
        "joy": [
            f"That's great to hear! 😊 Tell me more about what's bringing you happiness.",
            f"I love that you're feeling joyful! What's the best part of your day?"
        ],
        "anger": [
            f"That sounds frustrating. Would you like to tell me what happened?",
            f"I can hear that you're upset. Sometimes venting helps - I'm here to listen."
        ],
        "fear": [
            f"That sounds worrying. I'm here with you. What's concerning you most?",
            f"Fear can be really overwhelming. You're not alone in this."
        ],
        "surprise": [
            f"That's interesting! How are you processing this?",
            f"Life definitely has its surprises! How do you feel about this?"
        ],
        "love": [
            f"That's beautiful to hear. Tell me more!",
            f"I'm so glad you're experiencing that feeling."
        ],
        "neutral": [
            f"I appreciate you sharing that. How can I support you today?",
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
    st.subheader("🧠 What I've Learned")
    
    if st.session_state.memory["user_facts"]:
        st.write("**About you:**")
        for key, value in st.session_state.memory["user_facts"].items():
            st.write(f"- {key.capitalize()}: {value}")
    else:
        st.write("I haven't learned much about you yet. Tell me about yourself!")
    
    if st.session_state.memory["qa_pairs"]:
        st.write(f"**I remember {len(st.session_state.memory['qa_pairs'])} past questions**")
        # Show last 3
        recent = list(st.session_state.memory["qa_pairs"].values())[-3:]
        for qa in recent:
            st.write(f"- {qa['question'][:50]}...")

# -------------------------
# UI Header
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>With Memory & Knowledge</p>", unsafe_allow_html=True)

st.divider()

# -------------------------
# Sidebar with Memory Info
# -------------------------
with st.sidebar:
    st.markdown("### 📚 AI Capabilities")
    st.markdown("✅ **Answers factual questions**")
    st.markdown("✅ **Remembers your name & facts**")
    st.markdown("✅ **Learns from conversations**")
    st.markdown("✅ **Understands emotions**")
    
    st.divider()
    show_memory_dashboard()
    
    if st.button("🗑️ Clear Memory"):
        st.session_state.memory = {
            "user_facts": {},
            "conversation_summaries": [],
            "preferences": {},
            "qa_pairs": {}
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
    if emotion_counts:
        st.bar_chart(emotion_counts)

    if len(history_data) > 1:
        vad_data = {
            "Valence": [h["valence"] for h in history_data],
            "Arousal": [h["arousal"] for h in history_data],
            "Dominance": [h["dominance"] for h in history_data]
        }
        st.line_chart(vad_data)

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
user_input = st.chat_input("How are you feeling? Or ask me anything!")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user"):
        st.markdown(user_input)
    
    # Emotion Detection (for emotional context only)
    try:
        raw = emotion_model(user_input)
        emotions = raw[0] if isinstance(raw[0], list) else raw
        top = max(emotions, key=lambda x: x.get("score", 0))
        emotion = top.get("label", "neutral")
    except:
        emotion = "neutral"
    
    # Save Emotion History
    v, a, d = VAD_MAP.get(emotion.lower(), (0.5, 0.5, 0.5))
    st.session_state.emotion_history.append({
        "emotion": emotion,
        "valence": v,
        "arousal": a,
        "dominance": d
    })
    
    if len(st.session_state.emotion_history) > 20:
        st.session_state.emotion_history.pop(0)
    
    # Generate Enhanced Response
    with st.spinner("Thinking..."):
        reply = generate_enhanced_response(user_input, emotion)
    
    # Show Response
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    # Save response
    st.session_state.messages.append({"role": "assistant", "content": reply})
