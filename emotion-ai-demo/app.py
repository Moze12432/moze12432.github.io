import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch

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
    # Load model and tokenizer directly
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
# Emotion-based response templates (fallback)
# -------------------------
EMOTION_RESPONSES = {
    "sadness": [
        "I hear that you're feeling down. Would you like to talk about what's on your mind?",
        "I'm sorry you're feeling this way. Remember that tough feelings don't last forever.",
        "It's okay to feel sad sometimes. I'm here to listen if you want to share.",
        "That sounds difficult. What would help you feel a little better right now?",
        "I understand. Sometimes just acknowledging our feelings is the first step."
    ],
    "joy": [
        "That's wonderful to hear! What's making you feel so happy?",
        "I love hearing that! 😊 Tell me more about what brought you this joy.",
        "That's great! It's so nice when things are going well.",
        "I'm genuinely happy for you! What's the best part of what you're experiencing?",
        "That's awesome! Moments like these are precious."
    ],
    "anger": [
        "That sounds really frustrating. Would you like to tell me more about what happened?",
        "I can hear that you're upset. It's okay to feel angry sometimes.",
        "That must be really annoying. How are you planning to handle the situation?",
        "I understand why you'd feel that way. Sometimes things just don't go as planned.",
        "Your feelings are valid. Would talking about it help?"
    ],
    "fear": [
        "That sounds really worrying. I'm here with you.",
        "Fear can be overwhelming. What's the main thing that's concerning you?",
        "I understand feeling anxious. Let's break this down together.",
        "You're not alone in this. What would help you feel safer right now?",
        "It's okay to feel scared sometimes. Can we explore what's causing this feeling?"
    ],
    "surprise": [
        "Wow, that's unexpected! How are you processing this?",
        "That is surprising! What do you think about it?",
        "Life is full of surprises! How does this make you feel?",
        "Interesting! What's your take on this unexpected development?",
        "That caught me off guard too! How are you handling it?"
    ],
    "love": [
        "That's beautiful to hear. Love makes life so much richer.",
        "I'm so glad you're experiencing love. It's such a special feeling.",
        "That warms my heart! Tell me more about this.",
        "Love is wonderful. How has this affected your day?",
        "That's really special. It's great that you have that in your life."
    ],
    "neutral": [
        "I appreciate you sharing that. What else is on your mind?",
        "Interesting. Tell me more about that.",
        "I hear you. How can I support you today?",
        "Thanks for telling me. Is there anything specific you'd like to discuss?",
        "I'm listening. Feel free to share whatever you're thinking."
    ]
}

import random

# -------------------------
# Helper: Generate Response with FLAN
# -------------------------
def generate_flan_response(prompt):
    """Generate response using FLAN-T5 model directly"""
    try:
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
        
        with torch.no_grad():
            outputs = flan_model.generate(
                inputs.input_ids,
                max_length=100,
                num_beams=4,
                temperature=0.7,
                do_sample=True,
                pad_token_id=flan_tokenizer.eos_token_id
            )
        
        response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        st.write(f"FLAN generation error: {e}")
        return None

# -------------------------
# Helper: Generate Response (Free, no OpenAI)
# -------------------------
def generate_response_free(user_input, emotion, v, a, d):
    """
    Generate response using FLAN-T5 with emotion context
    Falls back to template responses if FLAN fails
    """
    
    # Create a prompt that guides FLAN to respond appropriately
    prompt = f"""You are a friendly, empathetic AI companion. The user is feeling {emotion} (Valence: {round(v,2)}, Arousal: {round(a,2)}, Dominance: {round(d,2)}).

User: {user_input}

Respond naturally, concisely (1-2 sentences), and with appropriate emotional tone. Be supportive and human-like."""
    
    # Try FLAN first
    try:
        response = generate_flan_response(prompt)
        
        # Check if response is valid and not just repeating the prompt
        if response and len(response) > 5 and len(response) < 200:
            if not response.startswith("User is feeling") and not response.startswith("You are"):
                return response
    except:
        pass
    
    # Try a simpler prompt if the first one fails
    try:
        simple_prompt = f"The user is {emotion}. Respond kindly: {user_input}"
        response = generate_flan_response(simple_prompt)
        if response and len(response) > 5 and len(response) < 200:
            return response
    except:
        pass
    
    # Fallback to template responses
    responses = EMOTION_RESPONSES.get(emotion, EMOTION_RESPONSES["neutral"])
    return random.choice(responses)

# -------------------------
# UI Header
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Free & Private • Powered by FLAN-T5</p>", unsafe_allow_html=True)

st.divider()

# -------------------------
# Toggle Dashboard
# -------------------------
if st.button("📊 Show / Hide Emotional Insights"):
    st.session_state.show_dashboard = not st.session_state.show_dashboard

# -------------------------
# Dashboard (ONLY if clicked)
# -------------------------
if st.session_state.show_dashboard and st.session_state.emotion_history:

    st.subheader("📊 Emotional Insights")

    history_data = st.session_state.emotion_history

    # Emotion Frequency
    st.write("### Emotion Frequency")
    emotion_counts = {}
    for item in history_data:
        e = item["emotion"]
        emotion_counts[e] = emotion_counts.get(e, 0) + 1
    st.bar_chart(emotion_counts)

    # VAD Trends
    st.write("### VAD Trends")
    vad_data = {
        "Valence": [h["valence"] for h in history_data],
        "Arousal": [h["arousal"] for h in history_data],
        "Dominance": [h["dominance"] for h in history_data]
    }
    st.line_chart(vad_data)

    # Personality Insight
    st.write("### 🧠 Personality Insight")

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

    st.info(f"""
    Over time, you appear to be:
    - {mood}
    - Showing {energy}
    """)
    
    # Show most common emotion
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
user_input = st.chat_input("How are you feeling today?")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------
    # Emotion Detection
    # -------------------------
    try:
        raw = emotion_model(user_input)
        emotions = raw[0] if isinstance(raw[0], list) else raw

        top = max(emotions, key=lambda x: x.get("score", 0))
        emotion = top.get("label", "neutral")
        confidence = round(top.get("score", 0) * 100, 2)

    except Exception as e:
        st.write(f"Emotion detection error: {e}")
        emotion = "neutral"
        confidence = 0.0
        emotions = []

    # -------------------------
    # VAD Calculation
    # -------------------------
    v, a, d = VAD_MAP.get(emotion.lower(), (0.5, 0.5, 0.5))

    # -------------------------
    # Save Emotion History
    # -------------------------
    st.session_state.emotion_history.append({
        "emotion": emotion,
        "confidence": confidence,
        "valence": v,
        "arousal": a,
        "dominance": d
    })

    # Keep history manageable
    if len(st.session_state.emotion_history) > 20:
        st.session_state.emotion_history.pop(0)

    # -------------------------
    # Generate Response (Free)
    # -------------------------
    with st.spinner("Thinking..."):
        reply = generate_response_free(user_input, emotion, v, a, d)

    # -------------------------
    # Show Response
    # -------------------------
    with st.chat_message("assistant"):
        st.markdown(reply)

    # Save response
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })
