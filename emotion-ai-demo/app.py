import streamlit as st
from transformers import pipeline

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Emotion AI Companion",
    page_icon="🧠",
    layout="centered"
)

# -------------------------
# Load Model (cached)
# -------------------------
@st.cache_resource
def load_model():
    return pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        return_all_scores=True
    )

emotion_model = load_model()

# -------------------------
# Session State (Chat Memory)
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------
# Title
# -------------------------
st.markdown(
    "<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Talk to an AI that understands your emotions</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# Display Chat History
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# User Input
# -------------------------
user_input = st.chat_input("How are you feeling today?")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # -------------------------
    # Emotion Detection (FIXED)
    # -------------------------
    try:
        raw_results = emotion_model(user_input)

        # Normalize output across versions
        if isinstance(raw_results, list) and len(raw_results) > 0:
            if isinstance(raw_results[0], list):
                emotions = raw_results[0]
            elif isinstance(raw_results[0], dict):
                emotions = raw_results
            else:
                emotions = []
        else:
            emotions = []

        # Safe fallback
        if not emotions:
            emotion = "neutral"
            confidence = 0.0
        else:
            top_emotion = max(emotions, key=lambda x: x.get('score', 0))
            emotion = top_emotion.get('label', 'unknown')
            confidence = round(top_emotion.get('score', 0) * 100, 2)

    except Exception as e:
        emotion = "error"
        confidence = 0.0
        emotions = []
        reply = "⚠️ I had trouble understanding that. Please try again."

    # -------------------------
    # Empathetic Responses
    # -------------------------
    responses = {
        "sadness": "I'm really sorry you're feeling this way. Want to talk about it?",
        "joy": "That’s amazing to hear 😊 What made you feel this way?",
        "anger": "That sounds really frustrating. Do you want to share what happened?",
        "fear": "That must feel overwhelming. I'm here with you.",
        "love": "That’s beautiful. It’s great to feel that kind of connection.",
        "surprise": "Oh wow, that sounds unexpected! Tell me more.",
        "neutral": "I see. Tell me more about how you're feeling."
    }

    reply = responses.get(emotion.lower(), "I understand. Tell me more about that.")

    # -------------------------
    # Show AI Response
    # -------------------------
    with st.chat_message("assistant"):
        st.markdown(reply)
        st.markdown(f"**Detected Emotion:** {emotion} ({confidence}%)")

        # Emotion Breakdown Chart
        if emotions:
            st.write("**Emotion Breakdown:**")
            chart_data = {
                item.get('label', 'unknown'): item.get('score', 0)
                for item in emotions
            }
            st.bar_chart(chart_data)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": f"{reply}\n\nEmotion: {emotion} ({confidence}%)"
    })
