import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import requests
from datetime import datetime
import re
import json
from urllib.parse import quote

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
    model_name = "google/flan-t5-large"  # Upgraded to large for better responses
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

# Memory storage
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
    return """I am an advanced AI language model created by Moses, a student at KyungDong University. 
    I can search the internet in real-time, remember our conversations, understand emotions, 
    and provide accurate, up-to-date information. How can I help you today?"""

# -------------------------
# POWERFUL INTERNET SEARCH - MULTIPLE RELIABLE SOURCES
# -------------------------

def search_restcountries(query):
    """Search for country information using REST Countries API"""
    try:
        # Extract country name from query
        country_match = re.search(r'population of (\w+)', query.lower())
        if country_match:
            country = country_match.group(1)
        else:
            # Try to find any country name
            countries = ['uganda', 'japan', 'china', 'india', 'usa', 'uk', 'france', 'germany', 'italy', 'spain', 
                        'brazil', 'canada', 'australia', 'russia', 'mexico', 'indonesia', 'nigeria', 'bangladesh',
                        'pakistan', 'egypt', 'turkey', 'thailand', 'vietnam', 'philippines', 'south korea', 'kenya']
            for country_name in countries:
                if country_name in query.lower():
                    country = country_name
                    break
            else:
                return None
        
        # Clean country name
        country = country.replace(' ', '%20')
        
        # REST Countries API
        url = f"https://restcountries.com/v3.1/name/{country}"
        response = requests.get(url, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            if data and len(data) > 0:
                country_data = data[0]
                population = country_data.get('population', 0)
                capital = country_data.get('capital', ['Unknown'])[0]
                region = country_data.get('region', 'Unknown')
                
                # Format nicely
                if population > 0:
                    formatted_pop = f"{population:,}"
                    return f"{country.capitalize()} has a population of {formatted_pop}. Capital: {capital}. Region: {region}."
    except:
        pass
    return None

def search_wikipedia_api(query):
    """Enhanced Wikipedia search for accurate information"""
    try:
        # Clean query for better results
        search_terms = query.lower()
        
        # Remove question words
        for word in ['what is', 'who is', 'where is', 'when is', 'why is', 'how is', 'tell me about']:
            search_terms = search_terms.replace(word, '')
        
        search_terms = search_terms.strip()
        
        # Wikipedia API
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(search_terms)}"
        response = requests.get(url, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            if "extract" in data:
                extract = data["extract"]
                # Clean up the extract
                extract = re.sub(r'\([^)]*\)', '', extract)  # Remove parentheticals
                extract = ' '.join(extract.split()[:50])  # First 50 words
                return extract
    except:
        pass
    return None

def search_duckduckgo_instant(query):
    """DuckDuckGo Instant Answer API - Best for facts"""
    try:
        url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1&t=hf"
        response = requests.get(url, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for abstract (most detailed)
            if data.get("AbstractText"):
                return data["AbstractText"][:500]
            
            # Check for definition
            if data.get("Definition"):
                return data["Definition"]
            
            # Check for answer
            if data.get("Answer"):
                return data["Answer"]
            
            # Check infobox
            if data.get("Infobox"):
                infobox = data["Infobox"]
                if "content" in infobox:
                    for content in infobox["content"]:
                        if "label" in content and "value" in content:
                            if "population" in content["label"].lower():
                                return f"Population: {content['value']}"
    except:
        pass
    return None

def search_numerical_fact(query):
    """Handle numerical facts like population, area, etc."""
    query_lower = query.lower()
    
    # Extract what's being asked
    if "population" in query_lower:
        # Extract country name
        words = query_lower.split()
        for i, word in enumerate(words):
            if word == "population" and i+1 < len(words):
                country = words[i+1]
                # Special cases
                country_map = {
                    'uganda': 'Uganda',
                    'japan': 'Japan',
                    'china': 'China',
                    'india': 'India',
                    'usa': 'USA',
                    'us': 'USA',
                    'uk': 'United Kingdom',
                    'france': 'France',
                    'germany': 'Germany'
                }
                country_name = country_map.get(country, country.capitalize())
                
                # Use REST Countries API
                return search_restcountries(f"population of {country_name}")
    
    return None

def search_combined(query):
    """Combine multiple search methods for best results"""
    
    # Method 1: Numerical facts (population, etc.)
    result = search_numerical_fact(query)
    if result:
        return result
    
    # Method 2: DuckDuckGo (good for general facts)
    result = search_duckduckgo_instant(query)
    if result and len(result) > 20:
        return result
    
    # Method 3: REST Countries (excellent for country data)
    result = search_restcountries(query)
    if result:
        return result
    
    # Method 4: Wikipedia (fallback for detailed info)
    result = search_wikipedia_api(query)
    if result and len(result) > 30:
        return result
    
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
    
    for key, value in facts_extracted.items():
        st.session_state.memory["user_facts"][key] = value
    
    return facts_extracted

def store_qa_pair(question, answer):
    """Store question-answer pairs for learning"""
    for qa in st.session_state.memory["qa_pairs"]:
        if qa["question"].lower() == question.lower():
            qa["answer"] = answer
            qa["asked_count"] += 1
            return
    
    st.session_state.memory["qa_pairs"].append({
        "question": question,
        "answer": answer,
        "asked_count": 1
    })
    
    if len(st.session_state.memory["qa_pairs"]) > 100:
        st.session_state.memory["qa_pairs"] = st.session_state.memory["qa_pairs"][-100:]

def recall_from_memory(user_input):
    """Try to recall if this question was asked before"""
    user_lower = user_input.lower().strip()
    
    for qa in st.session_state.memory["qa_pairs"]:
        if qa["question"].lower() == user_lower:
            return qa["answer"]
    
    return None

def is_factual_question(user_input):
    """Determine if the query is a factual question"""
    question_words = ["what", "who", "where", "when", "why", "how", "which", "is", "are", "population", "capital"]
    emotional_words = ["feel", "feeling", "sad", "happy", "angry", "scared", "lonely", "tired", "sick", "joke"]
    
    user_lower = user_input.lower()
    
    is_question = any(user_lower.startswith(word) for word in question_words) or user_lower.endswith('?')
    is_emotional = any(word in user_lower for word in emotional_words)
    
    # Keywords that indicate factual question
    factual_keywords = ["population", "capital", "area", "currency", "language", "president", "prime minister"]
    is_factual_keyword = any(keyword in user_lower for keyword in factual_keywords)
    
    return (is_question and not is_emotional) or is_factual_keyword

def is_personal_question(user_input):
    """Check if user is asking about themselves"""
    personal_patterns = [r"what is my name", r"who am i", r"my name", r"do you remember"]
    user_lower = user_input.lower()
    return any(re.search(pattern, user_lower) for pattern in personal_patterns)

def is_greeting(user_input):
    """Check if user is just greeting"""
    greetings = ["hi", "hello", "hey", "good morning", "good afternoon", "good evening", "greetings"]
    return user_input.lower().strip() in greetings

def get_greeting_response():
    greetings = [
        "Hello! How are you feeling today?",
        "Hi there! 😊 What's on your mind?",
        "Hey! How can I help you today?",
        "Greetings! How are you doing?"
    ]
    return random.choice(greetings)

# -------------------------
# Response Generation
# -------------------------
def generate_enhanced_response(user_input, emotion):
    """Generate smart response with search"""
    
    # Identity question
    if "who are you" in user_input.lower():
        return get_ai_identity()
    
    # Greeting
    if is_greeting(user_input):
        return get_greeting_response()
    
    # Personal questions
    if is_personal_question(user_input):
        if "name" in user_input.lower() and "name" in st.session_state.memory["user_facts"]:
            return f"Your name is {st.session_state.memory['user_facts']['name']}! I remember you told me."
        elif "name" in user_input.lower():
            return "I don't think you've told me your name yet. What should I call you?"
    
    # Memory recall
    recalled = recall_from_memory(user_input)
    if recalled:
        return recalled
    
    # Extract user facts
    facts = extract_user_facts(user_input)
    if facts and "name" in facts:
        return f"Nice to meet you, {facts['name']}! How can I help you today?"
    
    # Factual questions - SEARCH
    if is_factual_question(user_input):
        with st.spinner("🌐 Searching for accurate information..."):
            search_result = search_combined(user_input)
            
            if search_result:
                # Use FLAN to clean up the answer
                try:
                    prompt = f"Based on this information: '{search_result}'\n\nAnswer this question clearly and concisely in 1 sentence: {user_input}"
                    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=400)
                    with torch.no_grad():
                        outputs = flan_model.generate(
                            inputs.input_ids,
                            max_length=150,
                            num_beams=4,
                            temperature=0.5,
                            pad_token_id=flan_tokenizer.eos_token_id
                        )
                    cleaned = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if cleaned and len(cleaned) > 5:
                        store_qa_pair(user_input, cleaned)
                        return cleaned
                except:
                    pass
                
                store_qa_pair(user_input, search_result)
                return search_result
            else:
                return "I couldn't find reliable information. Could you rephrase your question or ask about something else?"
    
    # Emotional responses
    return get_empathetic_response(user_input, emotion)

def get_empathetic_response(user_input, emotion):
    """Generate empathetic response"""
    
    user_lower = user_input.lower()
    emotion = emotion.lower()
    
    # Health
    if any(word in user_lower for word in ["sick", "ill", "pain", "hurt"]):
        return "I'm sorry you're not feeling well. Please rest and stay hydrated. Do you need anything?"
    
    # Tired
    if any(word in user_lower for word in ["tired", "exhausted"]):
        return "I hear you're tired. Can you take a short break or rest?"
    
    # Stressed
    if any(word in user_lower for word in ["stressed", "overwhelmed", "anxious"]):
        return "That sounds stressful. Take a deep breath. What's one small thing that might help?"
    
    # Lonely
    if "lonely" in user_lower or "alone" in user_lower:
        return "I'm here with you. You're not alone. Would you like to talk?"
    
    # Angry
    if any(word in user_lower for word in ["angry", "mad", "frustrated"]):
        return "I hear your frustration. Would you like to talk about what's bothering you?"
    
    # Happy
    if any(word in user_lower for word in ["happy", "good", "great", "wonderful"]):
        return "That's wonderful! 😊 What's making you happy today?"
    
    # Joke
    if any(word in user_lower for word in ["joke", "funny", "make me laugh"]):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ]
        return random.choice(jokes)
    
    # Thank you
    if "thank" in user_lower:
        return "You're very welcome! 😊 Is there anything else I can help with?"
    
    # Emotion-based
    responses = {
        "sadness": ["I hear you're feeling down. I'm here to listen if you want to talk."],
        "joy": ["That's great to hear! 😊 Tell me more."],
        "anger": ["That sounds frustrating. I'm here to listen."],
        "fear": ["That sounds worrying. I'm here with you."],
        "neutral": ["How can I support you today?"]
    }
    
    return random.choice(responses.get(emotion, ["I appreciate you sharing that. How can I help?"]))

# -------------------------
# UI Components
# -------------------------
def show_memory_dashboard():
    st.subheader("🧠 What I've Learned")
    if st.session_state.memory["user_facts"]:
        st.write("**About you:**")
        for key, value in st.session_state.memory["user_facts"].items():
            st.write(f"- {key.capitalize()}: {value}")
    
    if st.session_state.memory["qa_pairs"]:
        st.write(f"**I remember {len(st.session_state.memory['qa_pairs'])} past questions**")

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
# Main UI
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University • Smart Internet Search</p>", unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### 📚 Features")
    st.markdown("✅ **Real-time internet search**")
    st.markdown("✅ **Population & country data**")
    st.markdown("✅ **Remembers your name**")
    st.markdown("✅ **Understands emotions**")
    st.markdown("✅ **Created by Moses at KyungDong University**")
    
    st.divider()
    show_memory_dashboard()
    
    st.divider()
    if st.button("🔄 Start New Chat", use_container_width=True):
        reset_chat()
    
    if st.button("🗑️ Clear Memory Only", use_container_width=True):
        st.session_state.memory = {
            "user_facts": {},
            "qa_pairs": []
        }
        st.rerun()

# Dashboard toggle
if st.button("📊 Show / Hide Emotional Insights"):
    st.session_state.show_dashboard = not st.session_state.show_dashboard

if st.session_state.show_dashboard and st.session_state.emotion_history:
    st.subheader("📊 Emotional Insights")
    emotion_counts = {}
    for item in st.session_state.emotion_history:
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
user_input = st.chat_input("Ask me anything! I can search the internet for facts.")

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
    
    if len(st.session_state.emotion_history) > 20:
        st.session_state.emotion_history.pop(0)
    
    reply = generate_enhanced_response(user_input, emotion)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
