import streamlit as st
from transformers import pipeline
from openai import OpenAI

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Emotion AI Companion",
    page_icon="🧠",
    layout="centered"
)

# -------------------------
# OpenAI Client
# -------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------
# Load Emotion Model
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
# Session Memory
# -------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

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
# Title
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion-MUKIIBI MOSES</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Talk to an AI that understands your emotions</p>", unsafe_allow_html=True)

st.divider()

# -------------------------
# Display Chat
# -------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# -------------------------
# User Input
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

        if isinstance(raw[0], list):
            emotions = raw[0]
        else:
            emotions = raw

        if emotions:
            top = max(emotions, key=lambda x: x.get("score", 0))
            emotion = top.get("label", "neutral")
            confidence = round(top.get("score", 0) * 100, 2)
        else:
            emotion = "neutral"
            confidence = 0.0

    except:
        emotion = "neutral"
        confidence = 0.0
        emotions = []

    # -------------------------
    # VAD Calculation
    # -------------------------
    v, a, d = VAD_MAP.get(emotion.lower(), (0.5, 0.5, 0.5))

    # -------------------------
    # Build Conversation Context
    # -------------------------
    history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]]
    )

    # -------------------------
    # System Prompt
    # -------------------------
    system_prompt = f"""
You are a highly emotionally intelligent AI companion.

Your personality:
- Warm, natural, human-like
- Not repetitive
- Emotionally aware and adaptive

User emotional state:
- Emotion: {emotion}
- Valence: {round(v,2)}
- Arousal: {round(a,2)}
- Dominance: {round(d,2)}

Guidelines:
- If user is sad → be comforting
- If happy → be enthusiastic
- If angry → be calm and grounding
- If anxious → reassure gently
- Refer naturally to past conversation
- Keep responses concise but meaningful
"""

    # -------------------------
    # GPT Response
    # -------------------------
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": history}
            ],
            temperature=0.85,
            max_tokens=150
        )

        reply = response.choices[0].message.content.strip()

    except Exception as e:
        reply = "I'm here for you. Something went wrong, but please keep talking to me."

    # -------------------------
    # Display Response
    # -------------------------
    with st.chat_message("assistant"):
        st.markdown(reply)
        st.markdown(f"**Emotion:** {emotion} ({confidence}%)")

        # Emotion Chart
        if emotions:
            st.write("**Emotion Breakdown:**")
            chart_data = {
                e.get("label", "unknown"): e.get("score", 0)
                for e in emotions
            }
            st.bar_chart(chart_data)

        # VAD Visualization
        st.write("**VAD (Emotional Dimensions):**")
        st.progress(v, text=f"Valence: {round(v,2)}")
        st.progress(a, text=f"Arousal: {round(a,2)}")
        st.progress(d, text=f"Dominance: {round(d,2)}")

    # Save assistant response
    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })
