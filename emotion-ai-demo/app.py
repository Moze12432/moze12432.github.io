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
    """Return the AI's identity"""
    return """I am an AI language model created by Moses, a student at KyungDong University. 
    I'm designed to be an Emotion-Aware AI Companion that can understand feelings, 
    answer questions, search the internet for information, and learn from our conversations. 
    How can I help you today?"""

# -------------------------
# RELIABLE INTERNET SEARCH FUNCTIONS
# -------------------------

def search_wikipedia(query):
    """Search Wikipedia for information - Most reliable for facts"""
    try:
        # Clean the query for Wikipedia
        clean_query = query.replace("what is ", "").replace("who is ", "").replace("capital of ", "")
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(clean_query)}"
        response = requests.get(url, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            if "extract" in data:
                extract = data["extract"]
                # Limit length
                if len(extract) > 400:
                    extract = extract[:400] + "..."
                return extract
    except Exception as e:
        pass
    return None

def search_duckduckgo_api(query):
    """Search using DuckDuckGo Instant Answer API (Free, no key needed)"""
    try:
        url = f"https://api.duckduckgo.com/?q={quote(query)}&format=json&no_html=1&skip_disambig=1"
        response = requests.get(url, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            
            # Check for abstract (summary)
            if data.get("AbstractText"):
                abstract = data["AbstractText"]
                if len(abstract) > 400:
                    abstract = abstract[:400] + "..."
                return abstract
            
            # Check for definition
            if data.get("Definition"):
                definition = data["Definition"]
                if len(definition) > 400:
                    definition = definition[:400] + "..."
                return definition
            
            # Check for answer
            if data.get("Answer"):
                return data["Answer"]
            
            # Check related topics
            if data.get("RelatedTopics") and len(data["RelatedTopics"]) > 0:
                first_topic = data["RelatedTopics"][0]
                if isinstance(first_topic, dict) and first_topic.get("Text"):
                    text = first_topic["Text"]
                    if len(text) > 400:
                        text = text[:400] + "..."
                    return text
    except Exception as e:
        pass
    return None

def search_wikidata(query):
    """Search Wikidata for factual information"""
    try:
        # Search for entity
        search_url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={quote(query)}&language=en&format=json&limit=1"
        response = requests.get(search_url, timeout=8)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('search') and len(data['search']) > 0:
                entity_id = data['search'][0]['id']
                # Get entity data
                entity_url = f"https://www.wikidata.org/wiki/Special:EntityData/{entity_id}.json"
                entity_response = requests.get(entity_url, timeout=8)
                
                if entity_response.status_code == 200:
                    entity_data = entity_response.json()
                    entities = entity_data.get('entities', {})
                    
                    if entity_id in entities:
                        entity_info = entities[entity_id]
                        
                        # Get description
                        description = ""
                        if 'descriptions' in entity_info and 'en' in entity_info['descriptions']:
                            description = entity_info['descriptions']['en']['value']
                        
                        # Get label
                        label = ""
                        if 'labels' in entity_info and 'en' in entity_info['labels']:
                            label = entity_info['labels']['en']['value']
                        
                        if description:
                            return f"{label}: {description}" if label else description
    except Exception as e:
        pass
    return None

def search_google_custom(query):
    """Fallback: Use a free search API (JSONPlaceholder doesn't work, so we'll use a different approach)"""
    # This is a list of common facts that might be asked
    common_facts = {
        "capital of south korea": "Seoul is the capital city of South Korea.",
        "south korea capital": "Seoul is the capital city of South Korea.",
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
        "capital of australia": "Canberra is the capital of Australia.",
        "capital of canada": "Ottawa is the capital of Canada.",
        "capital of mexico": "Mexico City is the capital of Mexico.",
        "capital of egypt": "Cairo is the capital of Egypt.",
        "capital of turkey": "Ankara is the capital of Turkey.",
        "capital of thailand": "Bangkok is the capital of Thailand.",
        "capital of vietnam": "Hanoi is the capital of Vietnam.",
        "capital of indonesia": "Jakarta is the capital of Indonesia.",
        "capital of malaysia": "Kuala Lumpur is the capital of Malaysia.",
        "capital of singapore": "Singapore is the capital of Singapore.",
        "capital of philippines": "Manila is the capital of the Philippines."
    }
    
    # Check if query matches any common fact
    query_lower = query.lower()
    for key, value in common_facts.items():
        if key in query_lower:
            return value
    
    return None

def search_internet(query):
    """Search the internet using multiple sources - RELIABLE VERSION"""
    
    # First, check common facts (fastest)
    result = search_google_custom(query)
    if result:
        return result
    
    # Try DuckDuckGo API (best for general queries)
    result = search_duckduckgo_api(query)
    if result:
        return result
    
    # Try Wikipedia (best for factual/encyclopedic info)
    result = search_wikipedia(query)
    if result:
        return result
    
    # Try Wikidata as last resort
    result = search_wikidata(query)
    if result:
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
    
    # Store in memory
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
    
    # Keep only last 100 pairs
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
    question_words = ["what", "who", "where", "when", "why", "how", "which", "is", "are", "do", "does", "capital"]
    emotional_words = ["feel", "feeling", "sad", "happy", "angry", "scared", "lonely", "tired", "sick", "joke"]
    
    user_lower = user_input.lower()
    
    is_question = any(user_lower.startswith(word) for word in question_words) or user_lower.endswith('?')
    is_emotional = any(word in user_lower for word in emotional_words)
    
    # Special case: questions about capital cities are factual
    if "capital" in user_lower:
        return True
    
    return is_question and not is_emotional

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
    """Return a friendly greeting"""
    greetings = [
        "Hello! How are you feeling today?",
        "Hi there! 😊 What's on your mind?",
        "Hey! How can I help you today?",
        "Greetings! How are you doing?"
    ]
    return random.choice(greetings)

# -------------------------
# Enhanced Response Generation
# -------------------------
def generate_enhanced_response(user_input, emotion):
    """Generate response with internet search for factual questions"""
    
    # PRIORITY 1: Identity question
    if "who are you" in user_input.lower() or "what are you" in user_input.lower():
        return get_ai_identity()
    
    # PRIORITY 2: Greeting
    if is_greeting(user_input):
        return get_greeting_response()
    
    # PRIORITY 3: Personal questions
    if is_personal_question(user_input):
        if "name" in user_input.lower() and "name" in st.session_state.memory["user_facts"]:
            return f"Your name is {st.session_state.memory['user_facts']['name']}! I remember you told me."
        elif "name" in user_input.lower():
            return "I don't think you've told me your name yet. What should I call you?"
        else:
            return "I remember our conversations! What would you like to know?"
    
    # PRIORITY 4: Memory recall
    recalled = recall_from_memory(user_input)
    if recalled:
        return recalled
    
    # PRIORITY 5: Extract user facts
    facts = extract_user_facts(user_input)
    if facts and "name" in facts:
        return f"Nice to meet you, {facts['name']}! How can I help you today?"
    
    # PRIORITY 6: Factual questions - SEARCH THE INTERNET
    if is_factual_question(user_input):
        with st.spinner("🌐 Searching the internet..."):
            search_result = search_internet(user_input)
            
            if search_result:
                # Store in memory
                store_qa_pair(user_input, search_result)
                return search_result
            else:
                # Try FLAN as fallback
                try:
                    prompt = f"Answer this question concisely: {user_input}"
                    inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
                    with torch.no_grad():
                        outputs = flan_model.generate(
                            inputs.input_ids,
                            max_length=100,
                            num_beams=4,
                            temperature=0.5,
                            pad_token_id=flan_tokenizer.eos_token_id
                        )
                    response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
                    if response and len(response) > 5 and "can't answer" not in response.lower():
                        store_qa_pair(user_input, response)
                        return response
                except:
                    pass
                
                return "I'm having trouble finding that information. Could you rephrase your question?"
    
    # PRIORITY 7: Emotional responses
    return get_empathetic_response(user_input, emotion)

def get_empathetic_response(user_input, emotion):
    """Generate empathetic response for emotional content"""
    
    user_lower = user_input.lower()
    emotion = emotion.lower()
    
    # Specific triggers
    if any(word in user_lower for word in ["sick", "ill", "pain", "hurt", "unwell"]):
        return "I'm sorry you're not feeling well. Make sure you rest and stay hydrated. Do you need anything?"
    
    if any(word in user_lower for word in ["tired", "exhausted", "sleep", "fatigue"]):
        return "I hear that you're tired. Is there any way you can take a short break or rest right now?"
    
    if any(word in user_lower for word in ["stressed", "overwhelmed", "anxious", "worry"]):
        return "That sounds stressful. Remember to breathe deeply and take things one step at a time."
    
    if any(word in user_lower for word in ["lonely", "alone", "isolated"]):
        return "I'm here with you. Feeling lonely is hard, but you're not alone. Would you like to talk?"
    
    if any(word in user_lower for word in ["angry", "mad", "frustrated", "annoyed", "upset"]):
        return "I hear your frustration. It's okay to feel angry. Would you like to talk about what's bothering you?"
    
    if any(word in user_lower for word in ["happy", "good", "great", "wonderful", "amazing", "excited"]):
        return "That's wonderful to hear! 😊 What's making you happy today?"
    
    if any(word in user_lower for word in ["joke", "funny", "make me laugh"]):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!",
            "What do you call a bear with no teeth? A gummy bear!"
        ]
        return random.choice(jokes)
    
    if "thank" in user_lower:
        return "You're very welcome! 😊 Is there anything else I can help you with?"
    
    # Emotion-based responses
    emotion_responses = {
        "sadness": ["I hear that you're feeling down. I'm sorry you're going through this. Would you like to talk more?"],
        "joy": ["That's great to hear! 😊 Tell me more about what's bringing you happiness."],
        "anger": ["That sounds frustrating. Would you like to tell me what happened?"],
        "fear": ["That sounds worrying. I'm here with you. What's concerning you most?"],
        "surprise": ["That's interesting! How are you processing this?"],
        "love": ["That's beautiful to hear. Tell me more!"],
        "neutral": ["I appreciate you sharing that. How can I support you today?"]
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
    
    if st.session_state.memory["qa_pairs"]:
        st.write(f"**I remember {len(st.session_state.memory['qa_pairs'])} past questions**")

# -------------------------
# UI Header
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University • With Internet Search</p>", unsafe_allow_html=True)

st.divider()

# -------------------------
# Sidebar
# -------------------------
with st.sidebar:
    st.markdown("### 📚 AI Capabilities")
    st.markdown("✅ **Searches Wikipedia, DuckDuckGo & Wikidata**")
    st.markdown("✅ **Remembers your name & facts**")
    st.markdown("✅ **Learns from conversations**")
    st.markdown("✅ **Understands emotions**")
    st.markdown("✅ **Created by Moses at KyungDong University**")
    
    st.divider()
    show_memory_dashboard()
    
    if st.button("🗑️ Clear Memory"):
        st.session_state.memory = {
            "user_facts": {},
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
    if emotion_counts:
        st.bar_chart(emotion_counts)

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
user_input = st.chat_input("Ask me anything! I can search the internet for facts.")

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
    
    # Generate Response
    reply = generate_enhanced_response(user_input, emotion)
    
    # Show Response
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    # Save response
    st.session_state.messages.append({"role": "assistant", "content": reply})
