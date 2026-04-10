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
    return """I am an advanced AI assistant created by Moses, a student at KyungDong University.

I have the following capabilities:
- Real-time internet search with intelligent analysis
- Code generation in Python and other languages
- Emotional intelligence - I understand and respond to feelings
- Mathematical calculations
- Memory - I remember facts you tell me
- Hypothetical reasoning - I can answer "what if" questions
- Current date and time

How can I help you today? Just ask me anything!"""

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

def search_news(query):
    """Search for news related to the query"""
    try:
        url = f"https://gnews.io/api/v4/search?q={urllib.parse.quote(query)}&lang=en&max=3&token=demo"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            articles = data.get("articles", [])
            if articles:
                results = []
                for article in articles[:2]:
                    title = article.get("title", "")
                    description = article.get("description", "")
                    if description:
                        results.append(f"{title}: {description[:200]}")
                    else:
                        results.append(title)
                return " | ".join(results)
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
                    region = country_data.get('region', 'Unknown')
                    
                    if 'population' in query.lower():
                        return f"{name} has a population of {population:,} people."
                    elif 'capital' in query.lower():
                        return f"The capital of {name} is {capital}."
                    else:
                        return f"{name}: Capital: {capital}, Population: {population:,}, Region: {region}"
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
    
    result = search_news(query)
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

Please provide a thoughtful, well-reasoned answer that:
1. Explains the key consequences and effects
2. Uses logical reasoning and scientific principles
3. Is engaging and easy to understand
4. Is 3-4 sentences long

Answer:"""
        elif "how to" in query_lower or "step by step" in query_lower or "guide" in query_lower:
            prompt = f"""Based on this information: "{search_result}"

Question: {query}

Provide a helpful, step-by-step answer that is practical and actionable. Keep it concise but informative.

Answer:"""
        elif "what is" in query_lower or "who is" in query_lower or "where is" in query_lower or "when" in query_lower:
            prompt = f"""Based on this factual information: "{search_result}"

Question: {query}

Provide a clear, accurate, and informative answer. Include key facts and context. Keep it to 2-3 sentences.

Answer:"""
        else:
            prompt = f"""Based on this information: "{search_result}"

Question: {query}

Provide a thoughtful, accurate, and helpful response that directly answers the question. Be conversational but informative.

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
        
        if response and len(response) > 20 and not response.startswith("Based on"):
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
    
    if "python" in instruction_lower or "code" in instruction_lower:
        
        if "factorial" in instruction_lower:
            return """Here's a Python function to calculate factorial:

```python
def factorial(n):
    if n == 0 or n == 1:
        return 1
    else:
        return n * factorial(n - 1)

print(factorial(5))  # Output: 120
print(factorial(10)) # Output: 3628800
```"""
        
        elif "prime" in instruction_lower:
            return """Here's Python code to find prime numbers:

```python
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True

primes = [n for n in range(2, 101) if is_prime(n)]
print(f"Prime numbers up to 100: {primes}")
```"""
        
        elif "fibonacci" in instruction_lower:
            return """Here's Python code for Fibonacci sequence:

```python
def fibonacci(n):
    a, b = 0, 1
    for _ in range(n):
        print(a, end=' ')
        a, b = b, a + b
    print()

fibonacci(10)
```"""
        
        elif "sort" in instruction_lower:
            return """Here's a sorting implementation in Python:

```python
def quicksort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)

numbers = [3, 6, 8, 10, 1, 2, 1]
print(f"Sorted: {quicksort(numbers)}")
```"""
        
        else:
            return """Here's a versatile Python template:

```python
def main():
    print("Welcome to your Python program!")
    # Add your code here
    
if __name__ == "__main__":
    main()
