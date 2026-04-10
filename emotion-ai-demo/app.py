// Interactive web page example
function greetUser() {
    const name = prompt('What is your name?');
    if (name) {
        alert(`Hello, ${name}! Welcome to my website.`);
        document.getElementById('greeting').innerHTML = `Hello ${name}!`;
    }
}

// Array manipulation
const numbers = [1, 2, 3, 4, 5];
const doubled = numbers.map(n => n * 2);
console.log(doubled); // [2, 4, 6, 8, 10]

// Fetch API example
async function fetchData() {
    try {
        const response = await fetch('https://api.example.com/data');
        const data = await response.json();
        console.log(data);
    } catch (error) {
        console.error('Error:', error);
    }
}
```"""
    
    return None

# -------------------------
# MEMORY FUNCTIONS
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
    
    # Location extraction
    location_match = re.search(r"i live in (\w+)", user_lower)
    if location_match:
        facts_extracted["location"] = location_match.group(1).capitalize()
    
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

# -------------------------
# UTILITY FUNCTIONS
# -------------------------

def get_current_date_time():
    """Get current date and time"""
    now = datetime.now()
    return {
        "date": now.strftime("%A, %B %d, %Y"),
        "time": now.strftime("%I:%M %p"),
        "full": now.strftime("%A, %B %d, %Y at %I:%M %p")
    }

def calculate(expression):
    """Safely calculate mathematical expressions"""
    try:
        # Remove any dangerous characters
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
        "What do you call a bear with no teeth? A gummy bear!",
        "Why don't eggs tell jokes? They'd crack each other up!",
        "What do you call a fish wearing a bowtie? Sofishticated!",
        "Why did the math book look so sad? Because it had too many problems!",
        "What do you call a sleeping bull? A bulldozer!",
        "Why did the bicycle fall over? Because it was two-tired!",
        "What do you call a pig that does karate? A pork chop!"
    ]
    return random.choice(jokes)

def get_random_fact():
    """Get an interesting random fact"""
    facts = [
        "Honey never spoils. Archaeologists have found 3000-year-old honey in Egyptian tombs that's still edible!",
        "Octopuses have three hearts and blue blood.",
        "A day on Venus is longer than a year on Venus.",
        "Bananas are berries, but strawberries aren't!",
        "The Eiffel Tower can grow up to 15 cm taller in summer due to thermal expansion.",
        "A group of flamingos is called a 'flamboyance'.",
        "The shortest war in history lasted only 38 minutes (between Britain and Zanzibar in 1896).",
        "Your brain uses about 20% of your body's total oxygen and energy.",
        "The average person walks the equivalent of three times around the Earth in their lifetime.",
        "A shrimp's heart is in its head."
    ]
    return random.choice(facts)

# -------------------------
# EMOTIONAL RESPONSES
# -------------------------

def get_emotional_response(user_input):
    """Generate empathetic emotional response"""
    user_lower = user_input.lower()
    
    if "sick" in user_lower or "ill" in user_lower:
        return "I'm sorry you're not feeling well. Please rest and take care of yourself. Do you need anything specific? Drink plenty of water and get some sleep."
    
    if "tired" in user_lower or "exhausted" in user_lower:
        return "I hear that you're tired. Your well-being is important. Can you take a short break or get some rest? Even a 15-minute nap can help."
    
    if "sad" in user_lower or "depressed" in user_lower:
        return "I'm sorry you're feeling sad. It's okay to have these feelings. Would you like to talk about what's bothering you? Sometimes sharing helps lighten the load."
    
    if "happy" in user_lower or "good" in user_lower or "great" in user_lower:
        return "That's wonderful to hear! 😊 What's making you happy today? I'd love to hear more about the good things in your life."
    
    if "angry" in user_lower or "mad" in user_lower or "frustrated" in user_lower:
        return "I hear your frustration. It's okay to feel angry. Would you like to talk about what's bothering you? Sometimes venting helps."
    
    if "scared" in user_lower or "fear" in user_lower or "afraid" in user_lower:
        return "I understand being scared. You're not alone. What's worrying you? Sometimes talking about our fears makes them feel more manageable."
    
    if "stressed" in user_lower or "overwhelmed" in user_lower or "anxious" in user_lower:
        return "Stress can be really challenging. Take a deep breath. What's one small thing that might help you feel better right now? Remember to be kind to yourself."
    
    if "lonely" in user_lower or "alone" in user_lower:
        return "I'm here with you. You're not alone. Would you like to talk about how you're feeling? I'm always here to listen."
    
    if "cry" in user_lower or "crying" in user_lower:
        return "It's okay to cry. Sometimes letting out emotions is healthy. I'm here for you. Do you want to talk about what's making you feel this way?"
    
    return "I'm here for you. How can I support you right now?"

# -------------------------
# MAIN RESPONSE GENERATOR
# -------------------------

def generate_smart_response(user_input):
    """Generate intelligent response using all capabilities"""
    
    user_lower = user_input.lower()
    
    # 1. Identity question
    if any(q in user_lower for q in ["who are you", "what are you", "your creator", "tell me about yourself"]):
        return get_ai_identity()
    
    # 2. Date and time
    if any(q in user_lower for q in ["date today", "today's date", "what day is it", "current date", "what's the date"]):
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}. The current time is {now.strftime('%I:%M %p')}."
    
    if any(q in user_lower for q in ["what time is it", "current time", "what's the time"]):
        now = datetime.now()
        return f"The current time is {now.strftime('%I:%M %p')}."
    
    # 3. Jokes
    if any(q in user_lower for q in ["joke", "funny", "make me laugh", "tell me a joke"]):
        return get_joke()
    
    # 4. Random facts
    if any(q in user_lower for q in ["fact", "tell me something interesting", "did you know", "interesting fact"]):
        return get_random_fact()
    
    # 5. Code generation
    if any(q in user_lower for q in ["write code", "generate code", "python code", "code for", "function for", "javascript"]):
        code_response = generate_code(user_input)
        if code_response:
            return code_response
        return "I can help you write code! What programming language and what would you like the code to do? Be specific with your request."
    
    # 6. Mathematical calculations
    calc_match = re.search(r'(\d+\s*[\+\-\*/%]\s*\d+)', user_input)
    if calc_match or "calculate" in user_lower:
        try:
            calc = re.sub(r'[^0-9+\-*/().% ]', '', user_input)
            result = calculate(calc)
            if result:
                return result
        except:
            pass
    
    # 7. Personal memory
    facts = extract_user_facts(user_input)
    if facts and "name" in facts:
        return f"Nice to meet you, {facts['name']}! I'll remember that. How can I help you today?"
    
    if "what is my name" in user_lower or "do you know my name" in user_lower:
        if "name" in st.session_state.memory["user_facts"]:
            return f"Your name is {st.session_state.memory['user_facts']['name']}! I remember you told me."
        return "You haven't told me your name yet. What should I call you?"
    
    # 8. Memory recall
    recalled = recall_from_memory(user_input)
    if recalled:
        return recalled
    
    # 9. Greetings
    if any(g in user_lower for g in ["hi", "hello", "hey", "greetings", "sup", "yo", "good morning", "good afternoon"]):
        greetings = [
            "Hello! How are you feeling today?",
            "Hi there! 😊 What can I help you with?",
            "Hey! How's your day going?",
            "Greetings! What's on your mind today?",
            "Hello! I'm here to help with anything you need."
        ]
        return random.choice(greetings)
    
    # 10. Thanks
    if "thank" in user_lower:
        thanks = [
            "You're very welcome! 😊 Is there anything else I can help with?",
            "Happy to help! Let me know if you need anything else.",
            "My pleasure! Feel free to ask me anything."
        ]
        return random.choice(thanks)
    
    # 11. Help
    if "help" in user_lower or "what can you do" in user_lower or "capabilities" in user_lower:
        return """I can help you with many things:

🌐 **Complex Questions** - "What would happen if the Moon disappeared?"
💻 **Code Generation** - "Write Python code to find prime numbers"
📅 **Date & Time** - "What's the date today?"
🧮 **Calculations** - "What is 156 * 23?"
😂 **Jokes & Fun Facts** - "Tell me a joke" or "Give me an interesting fact"
💭 **Emotional Support** - "I'm feeling stressed" or "I'm sad"
🔍 **Research** - I search the internet and analyze information
📚 **Memory** - I remember facts you tell me
🎯 **Hypothetical Scenarios** - "What if humans never discovered electricity?"

Just ask me anything naturally!"""
    
    # 12. Emotional responses
    if any(em in user_lower for em in ["feel", "feeling", "sad", "happy", "angry", "scared", "tired", "sick", "stressed", "lonely", "cry"]):
        return get_emotional_response(user_input)
    
    # 13. SEARCH AND ANALYZE (for everything else)
    with st.spinner("🌐 Searching the internet and analyzing information..."):
        search_result = comprehensive_search(user_input)
        
        if search_result:
            # Analyze and craft a thoughtful response
            analyzed_response = analyze_with_flan(user_input, search_result)
            
            if analyzed_response:
                store_qa_pair(user_input, analyzed_response)
                return analyzed_response
            else:
                # Fallback: provide a summarized version
                summary = search_result[:400] + "..." if len(search_result) > 400 else search_result
                response = f"Based on my research:\n\n{summary}"
                store_qa_pair(user_input, response)
                return response
        
        # 14. Default response for unrecognized queries
        return "I searched but couldn't find specific information about that. Could you rephrase your question? I can help with research, coding, calculations, emotional support, or just general conversation. What would you like to know?"

# -------------------------
# UI COMPONENTS
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
        recent = st.session_state.memory["qa_pairs"][-3:] if len(st.session_state.memory["qa_pairs"]) > 0 else []
        for qa in recent:
            question_preview = qa["question"][:50] + "..." if len(qa["question"]) > 50 else qa["question"]
            st.write(f"- {question_preview}")

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
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University • Intelligent Search & Analysis • Code Generation</p>", unsafe_allow_html=True)

st.divider()

# Sidebar
with st.sidebar:
    st.markdown("### ✨ Capabilities")
    st.markdown("🧠 **Intelligent Analysis** - Searches AND understands information")
    st.markdown("💻 **Code Generation** - Python, JavaScript, and more")
    st.markdown("🔍 **Complex Questions** - Hypothetical scenarios, science, history")
    st.markdown("📚 **Research** - Finds and synthesizes information from the web")
    st.markdown("💭 **Emotional Intelligence** - Understands and responds to feelings")
    st.markdown("📅 **Real-time Date & Time**")
    st.markdown("🧮 **Mathematical Calculations**")
    st.markdown("😂 **Jokes & Fun Facts**")
    st.markdown("💾 **Memory** - Remembers what you tell me")
    
    st.divider()
    
    if st.button("🔄 Start New Chat", use_container_width=True):
        reset_chat()
    
    st.divider()
    show_memory_dashboard()
    
    st.divider()
    st.info("💡 **Try these examples:**\n\n• What would happen if humans never discovered electricity?\n• Write Python code to check if a number is prime\n• I'm feeling really stressed about exams\n• What would happen if the Moon disappeared?\n• Calculate 247 * 89\n• Tell me a joke\n• My name is Moses")

# Dashboard toggle
col1, col2 = st.columns([3, 1])
with col1:
    if st.button("📊 Show / Hide Emotional Insights"):
        st.session_state.show_dashboard = not st.session_state.show_dashboard

if st.session_state.show_dashboard and st.session_state.emotion_history:
    st.subheader("📊 Emotional Insights")
    
    # Emotion frequency chart
    emotion_counts = {}
    for item in st.session_state.emotion_history[-20:]:
        e = item["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    if emotion_counts:
        st.bar_chart(emotion_counts)
    
    # Recent emotion
    if st.session_state.emotion_history:
        recent = st.session_state.emotion_history[-1]
        st.metric("Recent Emotion", recent["emotion"].capitalize())

st.divider()

# Chat display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything! I'll search, analyze, and give thoughtful answers...")

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
        "emotion": emotion,
        "valence": v,
        "arousal": a,
        "dominance": d,
        "timestamp": datetime.now().isoformat()
    })
    
    # Keep only last 50 entries
    if len(st.session_state.emotion_history) > 50:
        st.session_state.emotion_history.pop(0)
    
    # Generate response
    with st.spinner("🧠 Thinking and analyzing..."):
        reply = generate_smart_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
