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
        "conversation_context": [],
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
    return "I am an advanced AI assistant created by Moses, a student at KyungDong University. I have capabilities including real-time internet search with intelligent analysis, code generation in Python, emotional intelligence, mathematical calculations, memory, hypothetical reasoning, and current date/time. How can I help you today?"

# -------------------------
# REAL INTERNET SEARCH FUNCTIONS
# -------------------------

def search_wikipedia(query):
    """Search Wikipedia for comprehensive information"""
    try:
        search_terms = re.sub(r'what would happen if|what is|who is|where is|when is|how to|tell me about', '', query.lower())
        search_terms = search_terms.strip()
        
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{urllib.parse.quote(search_terms)}"
        response = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
        
        if response.status_code == 200:
            data = response.json()
            if "extract" in data:
                return data["extract"]
    except:
        pass
    return None

def search_duckduckgo(query):
    """Search DuckDuckGo Instant Answer API"""
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
    except:
        pass
    return None

def search_restcountries(query):
    """Search for country information"""
    try:
        country_match = re.search(r'\b(uganda|japan|china|india|usa|uk|france|germany|italy|spain|brazil|canada|australia|russia|south korea|vietnam|thailand)\b', query.lower())
        if country_match:
            country = country_match.group(1)
            url = f"https://restcountries.com/v3.1/name/{country.replace(' ', '%20')}"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    country_data = data[0]
                    name = country_data.get('name', {}).get('common', country.title())
                    population = country_data.get('population', 0)
                    capital = country_data.get('capital', ['Unknown'])[0]
                    
                    if 'population' in query.lower():
                        return f"{name} has a population of {population:,} people."
                    elif 'capital' in query.lower():
                        return f"The capital of {name} is {capital}."
                    else:
                        return f"{name}: Capital: {capital}, Population: {population:,}"
    except:
        pass
    return None

def comprehensive_search(query):
    """Combine multiple search methods"""
    result = search_restcountries(query)
    if result:
        return result
    
    result = search_wikipedia(query)
    if result and len(result) > 50:
        return result
    
    result = search_duckduckgo(query)
    if result and len(result) > 30:
        return result
    
    return None

# -------------------------
# INTELLIGENT RESPONSE ANALYSIS
# -------------------------

def analyze_with_flan(query, search_result):
    """Use FLAN to analyze search results and generate response"""
    try:
        query_lower = query.lower()
        
        if "what would happen if" in query_lower or "hypothetical" in query_lower or "what if" in query_lower:
            prompt = f"""Based on this scientific information: "{search_result}"

Question: {query}

Provide a thoughtful, well-reasoned answer that explains the key consequences. Be 3-4 sentences long.

Answer:"""
        elif "how to" in query_lower:
            prompt = f"""Based on this information: "{search_result}"

Question: {query}

Provide a helpful, step-by-step answer that is practical.

Answer:"""
        else:
            prompt = f"""Based on this information: "{search_result}"

Question: {query}

Provide a clear, accurate answer in 2-3 sentences.

Answer:"""
        
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=800)
        with torch.no_grad():
            outputs = flan_model.generate(
                inputs.input_ids,
                max_length=250,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                pad_token_id=flan_tokenizer.eos_token_id
            )
        response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        if response and len(response) > 20:
            return response
    except:
        pass
    return None

# -------------------------
# CODE GENERATION
# -------------------------

def generate_code(instruction):
    """Generate code based on user instruction"""
    instruction_lower = instruction.lower()
    
    if "factorial" in instruction_lower:
        return "```python\ndef factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    else:\n        return n * factorial(n - 1)\n\nprint(factorial(5))  # Output: 120\n```"
    
    elif "prime" in instruction_lower:
        return "```python\ndef is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n ** 0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprimes = [n for n in range(2, 101) if is_prime(n)]\nprint(f'Prime numbers up to 100: {primes}')\n```"
    
    elif "fibonacci" in instruction_lower:
        return "```python\ndef fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a, end=' ')\n        a, b = b, a + b\n    print()\n\nfibonacci(10)\n```"
    
    elif "sort" in instruction_lower:
        return "```python\ndef quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)\n\nnumbers = [3, 6, 8, 10, 1, 2, 1]\nprint(f'Sorted: {quicksort(numbers)}')\n```"
    
    return None

# -------------------------
# MEMORY FUNCTIONS
# -------------------------

def extract_user_facts(user_input):
    """Extract and store facts about the user"""
    user_lower = user_input.lower()
    facts_extracted = {}
    
    name_match = re.search(r'my name is (\w+)|i am (\w+)|call me (\w+)', user_lower)
    if name_match:
        name = next((g for g in name_match.groups() if g), None)
        if name:
            facts_extracted["name"] = name.capitalize()
    
    age_match = re.search(r'i am (\d+) years? old', user_lower)
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
            return
    
    st.session_state.memory["qa_pairs"].append({
        "question": question,
        "answer": answer,
        "asked_count": 1
    })

def recall_from_memory(user_input):
    """Try to recall if this question was asked before"""
    user_lower = user_input.lower().strip()
    
    for qa in st.session_state.memory["qa_pairs"]:
        if qa["question"].lower() == user_lower:
            return qa["answer"]
    
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
    """Get a random joke"""
    jokes = [
        "Why don't scientists trust atoms? Because they make up everything!",
        "What do you call a fake noodle? An impasta!",
        "Why did the scarecrow win an award? Because he was outstanding in his field!",
        "What do you call a bear with no teeth? A gummy bear!"
    ]
    return random.choice(jokes)

def get_random_fact():
    """Get an interesting random fact"""
    facts = [
        "Honey never spoils. Archaeologists have found 3000-year-old honey in Egyptian tombs that's still edible!",
        "Octopuses have three hearts and blue blood.",
        "A day on Venus is longer than a year on Venus.",
        "Bananas are berries, but strawberries aren't!"
    ]
    return random.choice(facts)

# -------------------------
# EMOTIONAL RESPONSES
# -------------------------

def get_emotional_response(user_input):
    """Generate empathetic emotional response"""
    user_lower = user_input.lower()
    
    if "sick" in user_lower:
        return "I'm sorry you're not feeling well. Please rest and take care of yourself."
    
    if "tired" in user_lower:
        return "I hear that you're tired. Can you take a short break or get some rest?"
    
    if "sad" in user_lower:
        return "I'm sorry you're feeling sad. Would you like to talk about what's bothering you?"
    
    if "happy" in user_lower or "good" in user_lower:
        return "That's wonderful to hear! What's making you happy today?"
    
    if "angry" in user_lower or "frustrated" in user_lower:
        return "I hear your frustration. Would you like to talk about what's bothering you?"
    
    if "stressed" in user_lower or "anxious" in user_lower:
        return "Stress can be challenging. Take a deep breath. What's one thing that might help?"
    
    return "I'm here for you. How can I support you right now?"

# -------------------------
# MAIN RESPONSE GENERATOR
# -------------------------

def generate_smart_response(user_input):
    """Generate intelligent response using all capabilities"""
    
    user_lower = user_input.lower()
    
    # 1. Identity question
    if any(q in user_lower for q in ["who are you", "what are you", "your creator"]):
        return get_ai_identity()
    
    # 2. Date and time
    if any(q in user_lower for q in ["date today", "today's date", "what day is it", "current date"]):
        return f"Today is {get_current_date_time()}."
    
    if any(q in user_lower for q in ["what time is it", "current time"]):
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."
    
    # 3. Jokes
    if any(q in user_lower for q in ["joke", "funny", "make me laugh", "tell me a joke"]):
        return get_joke()
    
    # 4. Random facts
    if any(q in user_lower for q in ["fact", "interesting fact", "did you know"]):
        return get_random_fact()
    
    # 5. Code generation
    if any(q in user_lower for q in ["write code", "python code", "code for"]):
        code_response = generate_code(user_input)
        if code_response:
            return code_response
    
    # 6. Mathematical calculations
    calc_match = re.search(r'(\d+\s*[\+\-\*/%]\s*\d+)', user_input)
    if calc_match:
        try:
            result = calculate(user_input)
            if result:
                return result
        except:
            pass
    
    # 7. Personal memory
    facts = extract_user_facts(user_input)
    if facts and "name" in facts:
        return f"Nice to meet you, {facts['name']}! I'll remember that. How can I help you today?"
    
    if "what is my name" in user_lower:
        if "name" in st.session_state.memory["user_facts"]:
            return f"Your name is {st.session_state.memory['user_facts']['name']}!"
        return "You haven't told me your name yet. What should I call you?"
    
    # 8. Memory recall
    recalled = recall_from_memory(user_input)
    if recalled:
        return recalled
    
    # 9. Greetings
    if any(g in user_lower for g in ["hi", "hello", "hey", "greetings"]):
        return "Hello! How are you feeling today?"
    
    # 10. Thanks
    if "thank" in user_lower:
        return "You're very welcome! Is there anything else I can help with?"
    
    # 11. Help
    if "help" in user_lower or "what can you do" in user_lower:
        return """I can help you with:
- Complex questions (What would happen if the Moon disappeared?)
- Code generation (Write Python code)
- Date & time
- Calculations
- Jokes and fun facts
- Emotional support
- Research and information

Just ask me anything!"""
    
    # 12. Emotional responses
    if any(em in user_lower for em in ["feel", "feeling", "sad", "happy", "angry", "scared", "tired", "sick", "stressed"]):
        return get_emotional_response(user_input)
    
    # 13. SEARCH AND ANALYZE
    with st.spinner("Searching the internet and analyzing information..."):
        search_result = comprehensive_search(user_input)
        
        if search_result:
            analyzed_response = analyze_with_flan(user_input, search_result)
            if analyzed_response:
                store_qa_pair(user_input, analyzed_response)
                return analyzed_response
            else:
                store_qa_pair(user_input, search_result)
                return search_result
        
        # 14. Default response
        return "I searched but couldn't find specific information about that. Could you rephrase your question?"

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
        "conversation_context": [],
        "qa_pairs": []
    }
    st.rerun()

# -------------------------
# MAIN UI
# -------------------------
st.markdown("<h1 style='text-align: center;'>Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University</p>", unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### Capabilities")
    st.markdown("- Internet search with analysis")
    st.markdown("- Code generation (Python)")
    st.markdown("- Hypothetical reasoning")
    st.markdown("- Emotional intelligence")
    st.markdown("- Memory & learning")
    st.markdown("- Date & time")
    st.markdown("- Calculations")
    
    st.divider()
    
    if st.button("Start New Chat", use_container_width=True):
        reset_chat()
    
    st.divider()
    show_memory_dashboard()

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
user_input = st.chat_input("Ask me anything! I'll search and analyze...")

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
        "emotion": emotion,
        "valence": v,
        "arousal": a,
        "dominance": d
    })
    
    # Generate response
    with st.spinner("Thinking..."):
        reply = generate_smart_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
