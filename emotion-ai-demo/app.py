import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random

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
# Better emotion-based responses (direct, no FLAN issues)
# -------------------------
def get_empathetic_response(user_input, emotion):
    """Generate a natural response based on emotion and user input"""
    
    # Convert to lowercase for matching
    user_lower = user_input.lower()
    emotion = emotion.lower()
    
    # Check for specific keywords first
    if "sick" in user_lower or "ill" in user_lower or "pain" in user_lower:
        return "I'm sorry you're not feeling well. Being sick is tough. Make sure you rest and stay hydrated. Do you have someone to take care of you?"
    
    if "tired" in user_lower or "exhausted" in user_lower or "sleep" in user_lower:
        return "I hear that you're tired. Lack of rest can really affect how we feel. Is there any way you can take a short break or rest right now?"
    
    if "stressed" in user_lower or "overwhelmed" in user_lower:
        return "That sounds really stressful. Remember to breathe deeply and take things one step at a time. What's one small thing that might help right now?"
    
    if "lonely" in user_lower or "alone" in user_lower:
        return "I'm here with you. Feeling lonely is hard, but you're not alone in this moment. Would you like to talk about what's on your mind?"
    
    if "angry" in user_lower or "mad" in user_lower or "frustrated" in user_lower:
        return "I can hear your frustration. It's okay to feel angry. Would you like to talk about what's bothering you?"
    
    if "happy" in user_lower or "good" in user_lower or "great" in user_lower:
        return "That's wonderful to hear! 😊 I'm glad you're feeling good. What's making you happy today?"
    
    # Emotion-based responses
    if emotion == "sadness":
        responses = [
            f"I hear that you're feeling down. I'm sorry you're going through this. Would you like to talk more about what's making you feel this way?",
            f"That sounds really difficult. It's okay to feel sad sometimes. I'm here to listen if you want to share more.",
            f"I understand that feeling. Remember that tough emotions don't last forever. How can I support you right now?"
        ]
        return random.choice(responses)
    
    elif emotion == "joy":
        responses = [
            f"That's great to hear! 😊 What's been going well for you?",
            f"I love that you're feeling joyful! Tell me more about what's bringing you happiness.",
            f"That's wonderful! Moments of joy are so precious. What's the best part of your day?"
        ]
        return random.choice(responses)
    
    elif emotion == "anger":
        responses = [
            f"That sounds frustrating. It's valid to feel angry. Would you like to tell me what happened?",
            f"I can hear that you're upset. Sometimes venting helps - I'm here to listen.",
            f"Your feelings are completely valid. What would help you feel better right now?"
        ]
        return random.choice(responses)
    
    elif emotion == "fear":
        responses = [
            f"That sounds worrying. I'm here with you. What's the main thing that's concerning you?",
            f"Fear can be really overwhelming. You're not alone in this. Can we talk about what's scaring you?",
            f"I understand feeling afraid. Sometimes sharing our fears makes them feel smaller. Want to talk about it?"
        ]
        return random.choice(responses)
    
    elif emotion == "surprise":
        responses = [
            f"Wow, that's unexpected! How are you processing this?",
            f"Life definitely has its surprises! How do you feel about this?",
            f"That caught me off guard too! What's your take on this?"
        ]
        return random.choice(responses)
    
    elif emotion == "love":
        responses = [
            f"That's beautiful to hear. Love makes everything feel brighter. Tell me more!",
            f"I'm so glad you're experiencing love. It's such a special feeling.",
            f"That warms my heart! How has this affected your day?"
        ]
        return random.choice(responses)
    
    else:  # neutral or unknown
        responses = [
            f"I appreciate you sharing that with me. How can I support you today?",
            f"I hear you. Is there anything specific you'd like to talk about?",
            f"Thanks for telling me. I'm here to listen whenever you need."
        ]
        return random.choice(responses)

# -------------------------
# Try FLAN with better prompting (optional, will fall back if fails)
# -------------------------
def try_flan_response(user_input, emotion):
    """Attempt to use FLAN, return None if it fails or gives bad output"""
    try:
        # Better prompt that forces a natural response
        prompt = f"""Answer as a caring friend. The user is feeling {emotion}. 
User says: "{user_input}"
Your empathetic reply (keep it short, 1-2 sentences):"""
        
        inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=200)
        
        with torch.no_grad():
            outputs = flan_model.generate(
                inputs.input_ids,
                max_length=80,
                min_length=10,
                num_beams=4,
                temperature=0.8,
                do_sample=True,
                repetition_penalty=1.2,
                pad_token_id=flan_tokenizer.eos_token_id
            )
        
        response = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Check if response is bad (echoing input or too short)
        if response and len(response) > 8 and len(response) < 150:
            # Don't return if it's just echoing the user
            if user_input.lower() not in response.lower():
                return response
    except:
        pass
    return None

# -------------------------
# UI Header
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Your empathetic AI friend</p>", unsafe_allow_html=True)

st.divider()

# -------------------------
# Toggle Dashboard
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
user_input = st.chat_input("How are you feeling today?")

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
    
    # Generate Response - try FLAN first, fall back to template
    with st.spinner("Thinking..."):
        reply = try_flan_response(user_input, emotion)
        if not reply:
            reply = get_empathetic_response(user_input, emotion)
    
    # Show Response
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    # Save response
    st.session_state.messages.append({"role": "assistant", "content": reply})
