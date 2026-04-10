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
    return pipeline(
        "text-generation",   # ✅ change this
        model="google/flan-t5-small"  # lighter + safer
    )

emotion_model = load_emotion_model()
flan_model = load_flan_model()

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
# UI Header
# -------------------------
st.markdown("<h1 style='text-align: center;'>🧠 Emotion-Aware AI Companion</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Emotion AI + Memory + Hybrid Intelligence</p>", unsafe_allow_html=True)

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

    except:
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

    # -------------------------
    # Build Context
    # -------------------------
    history = "\n".join(
        [f"{m['role']}: {m['content']}" for m in st.session_state.messages[-6:]]
    )

    # Long-term summary
    if st.session_state.emotion_history:
        avg_valence = sum(h["valence"] for h in st.session_state.emotion_history) / len(st.session_state.emotion_history)

        if avg_valence > 0.6:
            long_term = "User is generally positive."
        elif avg_valence < 0.4:
            long_term = "User often experiences negative emotions."
        else:
            long_term = "User has balanced emotions."
    else:
        long_term = ""

    # -------------------------
    # System Prompt (GPT)
    # -------------------------
    system_prompt = f"""
You are a highly emotionally intelligent AI companion.

Personality:
- Natural, human-like, non-repetitive
- Emotionally adaptive

User emotion:
- {emotion}
- VAD: ({round(v,2)}, {round(a,2)}, {round(d,2)})

Long-term pattern:
{long_term}

Instructions:
- Respond like a real human friend
- Be concise but meaningful
- Adapt tone to emotion
- Use context naturally
"""

    # -------------------------
    # Flan Prompt (Fallback)
    # -------------------------
    flan_prompt = f"""
You are a supportive and emotionally intelligent AI.

User emotion: {emotion}
Valence: {round(v,2)}, Arousal: {round(a,2)}, Dominance: {round(d,2)}

Conversation:
{history}

Respond naturally and empathetically like a human.
Keep it short and meaningful.
"""

    # -------------------------
    # Hybrid Response System
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

    except Exception:
        try:
            flan_output = flan_model(flan_prompt)[0]["generated_text"]
            reply = flan_output.strip()

            if len(reply) < 10:
                raise ValueError("Weak output")

        except:
            if emotion == "sadness":
                reply = "I'm really sorry you're feeling this way. Do you want to talk about it?"
            elif emotion == "joy":
                reply = "That’s really nice to hear 😊 What made you feel this way?"
            elif emotion == "anger":
                reply = "That sounds frustrating. What happened?"
            elif emotion == "fear":
                reply = "That must feel overwhelming. I'm here with you."
            else:
                reply = "I understand. Tell me more about how you're feeling."

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
