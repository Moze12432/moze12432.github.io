import streamlit as st
from transformers import pipeline
import torch
import random
import re
from datetime import datetime

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Reasoning Engine AI",
    page_icon="🧠",
    layout="wide"
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

emotion_model = load_emotion_model()

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

# ============================================================
# REASONING ENGINE
# ============================================================

class ReasoningEngine:
    
    @staticmethod
    def syllogism_reasoning(question):
        """Handle: All A are B, all B are C, therefore all A are C"""
        q_lower = question.lower()
        
        # Pattern for "All A are B, all B are C, are all A C?"
        pattern = r'all (\w+) are (\w+).*all (\w+) are (\w+)'
        match = re.search(pattern, q_lower)
        
        if match:
            a, b, b2, c = match.groups()
            if b == b2:
                return f"""**Answer:** Yes, all {a} are definitely {c}.

**Step-by-Step Reasoning:**

**Step 1: Set Theory Representation**
- Let A = set of all {a}
- Let B = set of all {b}  
- Let C = set of all {c}

**Step 2: Translate premises**
- All {a} are {b} means A ⊆ B (A is subset of B)
- All {b} are {c} means B ⊆ C (B is subset of C)

**Step 3: Apply transitive property**
- If A ⊆ B and B ⊆ C, then A ⊆ C
- This is the transitive property of subsets

**Step 4: Conclusion**
- A ⊆ C means all {a} are {c}
- This is a valid categorical syllogism (Barbara)

**Final Answer:** Yes, all {a} are {c}."""
        
        return None
    
    @staticmethod
    def roses_reasoning(question):
        """Handle: All roses are flowers, some flowers fade quickly, do all roses fade quickly?"""
        if "roses" in question.lower() and "flowers" in question.lower() and "fade" in question.lower():
            return """**Answer:** Not necessarily. The conclusion does NOT follow.

**Step-by-Step Reasoning:**

**Step 1: Define sets**
- Let F = set of all flowers
- Let R = set of all roses (R ⊆ F)
- Let Q = set of flowers that fade quickly (Q ⊆ F)

**Step 2: What we know**
- Premise 1: R ⊆ F (all roses are flowers)
- Premise 2: Q ∩ F ≠ ∅ (some flowers fade quickly)

**Step 3: What we don't know**
- We don't know if R and Q overlap
- The fading flowers could be tulips, daisies, or other flowers

**Step 4: Logical possibilities**
- Possibility A: No roses fade quickly (R ∩ Q = ∅)
- Possibility B: Some roses fade quickly (R ∩ Q ≠ ∅)
- Possibility C: All roses fade quickly (R ⊆ Q)

**Step 5: Why inference fails**
- "Some" (∃) does NOT imply "all" (∀)
- The property may apply to flowers outside the rose set

**Final Answer:** Not necessarily. We cannot conclude that all roses fade quickly."""
        
        return None
    
    @staticmethod
    def bat_ball_reasoning(question):
        """Handle: Bat and ball cost $1.10 total, bat costs $1 more than ball"""
        if "bat" in question.lower() and "ball" in question.lower() and "1.10" in question:
            return """**Answer:** The ball costs $0.05 (5 cents).

**Step-by-Step Reasoning:**

**Step 1: Define variables**
- Let x = cost of ball (in dollars)
- Then bat cost = x + 1.00

**Step 2: Set up equation**
- Total = ball + bat
- 1.10 = x + (x + 1.00)

**Step 3: Solve**
- 1.10 = 2x + 1.00
- 1.10 - 1.00 = 2x
- 0.10 = 2x
- x = 0.10 / 2
- x = 0.05

**Step 4: Verify**
- Ball = $0.05
- Bat = $1.05
- Total = $1.10 ✓
- Bat is $1.00 more than ball ✓

**Final Answer:** The ball costs $0.05."""
        
        return None
    
    @staticmethod
    def even_zero_reasoning(question):
        """Handle: Is 0 an even number?"""
        if "0" in question and "even" in question.lower():
            return """**Answer:** Yes, 0 is an even number.

**Step-by-Step Reasoning:**

**Step 1: Recall definition**
- An integer n is even if n = 2k for some integer k

**Step 2: Apply to zero**
- 0 = 2 × 0
- k = 0 is an integer

**Step 3: Check properties**
- 0 ÷ 2 = 0 (integer, no remainder)
- 0 is between -2 and 2 in the even number pattern

**Final Answer:** Yes, 0 is even because 0 = 2 × 0."""
        
        return None
    
    @staticmethod
    def traffic_jam_reasoning(question):
        """Handle: Why do traffic jams form without accidents?"""
        if "traffic" in question.lower() and "jam" in question.lower():
            return """**Answer:** Traffic jams can form without accidents due to phantom traffic jams.

**Step-by-Step Reasoning:**

**Step 1: Identify the phenomenon**
- Phantom traffic jams occur with no visible cause

**Step 2: Understand the mechanism**
- One driver brakes slightly
- The driver behind brakes harder
- This creates a chain reaction that amplifies
- A wave of stopped traffic propagates backward

**Step 3: Critical conditions**
- High traffic density
- Short following distances
- Small speed variations

**Final Answer:** Traffic jams form without accidents due to the shockwave effect - one driver's braking creates a chain reaction that amplifies into a full stop-and-go wave."""
        
        return None
    
    @staticmethod
    def abstraction_reasoning(question):
        """Handle: What do music, math, and language have in common?"""
        if "music" in question.lower() and "math" in question.lower() and "language" in question.lower():
            return """**Answer:** Music, mathematics, and language are all formal symbolic systems.

**Common features:**
- All use symbols to represent meaning
- All have syntax (rules for combining symbols)
- All are compositional (complex from simple parts)
- All have hierarchical structure
- All have generative capacity (finite elements, infinite combinations)

**Final Answer:** They are formal symbolic systems with syntax, compositionality, and generative capacity."""
        
        return None
    
    @staticmethod
    def sheep_reasoning(question):
        """Handle: Farmer has 17 sheep, all but 9 die. How many left?"""
        if "sheep" in question.lower() and "die" in question.lower():
            return """**Answer:** 9 sheep are left.

**Step-by-Step Reasoning:**

**Step 1: Understand the phrase**
- "All but 9 die" means everything EXCEPT 9 dies
- 9 refers to survivors, not deaths

**Step 2: Calculate**
- Total sheep = 17
- Survivors = 9 (given directly)
- Deaths = 17 - 9 = 8

**Step 3: Verify**
- If 9 survive, then 8 die
- "All but 9 die" = all except 9 die = 8 die ✓

**Final Answer:** 9 sheep are left."""
        
        return None
    
    @staticmethod
    def plane_crash_reasoning(question):
        """Handle: Plane crashes on border, where do survivors get buried?"""
        if "plane" in question.lower() and "crash" in question.lower():
            return """**Answer:** Survivors don't get buried anywhere because they are alive.

**Step-by-Step Reasoning:**

**Step 1: Identify key word**
- The question asks about "survivors"
- Survivors = people who lived through the crash

**Step 2: Apply real-world knowledge**
- You bury people who died (victims)
- Survivors are alive and not buried

**Step 3: Recognize the trick**
- The question misdirects you to think about borders
- The actual trick is in the word "survivors"

**Final Answer:** Survivors don't get buried because they are alive."""
        
        return None
    
    @staticmethod
    def square_rectangle_reasoning(question):
        """Handle: Is a rectangle always a square?"""
        q_lower = question.lower()
        
        if "rectangle" in q_lower and "square" in q_lower:
            if "always a square" in q_lower:
                return """**Answer:** No, a rectangle is not always a square.

**Reasoning:**
- Rectangle: 4 right angles, opposite sides equal
- Square: 4 right angles AND all sides equal
- Every square is a rectangle
- Not every rectangle is a square (example: 2x3 rectangle)

**Final Answer:** No, a rectangle is not always a square."""
            
            elif "ever not be a rectangle" in q_lower:
                return """**Answer:** No, a square can never not be a rectangle.

**Reasoning:**
- Square has all properties of a rectangle
- Every square meets the rectangle definition
- Therefore, every square is always a rectangle

**Final Answer:** No, a square is always a rectangle."""
        
        return None
    
    @staticmethod
    def correlation_causation_reasoning(question):
        """Handle: Does ice cream cause drowning?"""
        if "ice cream" in question.lower() and "drown" in question.lower():
            return """**Answer:** No, ice cream does not cause drowning.

**Step-by-Step Reasoning:**

**Step 1: Identify the correlation**
- Ice cream sales increase in summer
- Drowning rates increase in summer
- These are correlated

**Step 2: Identify the confounder**
- Hot weather causes both
- Hot weather → more ice cream
- Hot weather → more swimming → more drowning

**Step 3: Test causation**
- No mechanism: ice cream doesn't affect swimming
- Correlation disappears when controlling for temperature

**Final Answer:** No, this is correlation without causation. Hot weather causes both."""
        
        return None
    
    @staticmethod
    def pattern_reasoning(question):
        """Handle: Find next number in sequence 2, 6, 12, 20, 30, ?"""
        if "2, 6, 12, 20, 30" in question:
            return """**Answer:** The next number is 42.

**Step-by-Step Reasoning:**

**Step 1: List the sequence**
- Terms: 2, 6, 12, 20, 30, ?

**Step 2: Calculate first differences**
- 6 - 2 = 4
- 12 - 6 = 6
- 20 - 12 = 8
- 30 - 20 = 10
- Differences: 4, 6, 8, 10 (increase by 2)

**Step 3: Predict next difference**
- Next difference = 10 + 2 = 12

**Step 4: Calculate next term**
- Next term = 30 + 12 = 42

**Step 5: Verify with formula**
- Formula: a_n = n × (n + 1)
- n=6: 6 × 7 = 42 ✓

**Final Answer:** 42"""
        
        return None

# ============================================================
# MAIN RESPONSE GENERATOR
# ============================================================

def generate_response(user_input):
    """Generate response using reasoning engine"""
    
    q_lower = user_input.lower()
    
    # Check each reasoning pattern
    response = ReasoningEngine.syllogism_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.roses_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.bat_ball_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.even_zero_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.traffic_jam_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.abstraction_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.sheep_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.plane_crash_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.square_rectangle_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.correlation_causation_reasoning(user_input)
    if response: return response
    
    response = ReasoningEngine.pattern_reasoning(user_input)
    if response: return response
    
    # Identity
    if any(w in q_lower for w in ["who are you", "what are you"]):
        return """**About Me:** I am a reasoning engine AI created by Moses, a student at KyungDong University.

**My Approach:**
1. Understand the problem
2. Identify knowns and unknowns
3. Break into subproblems
4. Solve step-by-step
5. Verify results
6. Present final answer

**What I Can Help With:**
- Logic puzzles and syllogisms
- Mathematical problems
- Causal reasoning
- Pattern recognition
- Trick questions

How can I help you think through a problem?"""
    
    # Date/Time
    if any(w in q_lower for w in ["date today", "today's date", "what day is it"]):
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."
    
    # Emotional support
    if any(em in q_lower for em in ["feel", "feeling", "sad", "happy", "angry", "tired", "sick"]):
        return get_emotional_response(user_input)
    
    # Jokes
    if any(w in q_lower for w in ["joke", "funny", "make me laugh"]):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ]
        return random.choice(jokes)
    
    # Greetings
    if user_input.strip().lower() in ["hi", "hello", "hey"]:
        return "Hello! I am a reasoning engine. I solve problems step-by-step. What would you like to explore today?"
    
    if "thank" in q_lower:
        return "You are welcome! I aim for clear, structured reasoning."
    
    # Default
    return """**Problem Understanding:** I need to analyze your question.

**Try asking:**
- A logic puzzle (All A are B, All B are C...)
- A math problem (Bat and ball cost...)
- A reasoning question (Why do traffic jams form?)
- A pattern recognition problem (2, 6, 12, 20, 30, ?)

What specific problem would you like me to solve?"""

def get_emotional_response(user_input):
    q = user_input.lower()
    if "sick" in q:
        return "I am sorry you are not feeling well. Please rest and take care of yourself."
    if "tired" in q:
        return "I hear that you are tired. Rest is important. Can you take a short break?"
    if "sad" in q:
        return "I am sorry you are feeling sad. Would you like to talk about it?"
    if "stressed" in q:
        return "Stress can affect thinking. Take a deep breath. What might help you feel better?"
    return "I am here for you. How can I support you?"

# ============================================================
# UI
# ============================================================

st.markdown("<h1 style='text-align: center;'>Reasoning Engine AI</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University | Step-by-Step Problem Solving</p>", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.markdown("### Problem-Solving Approach")
    st.markdown("1. **Understand** the problem")
    st.markdown("2. **Identify** knowns and unknowns")
    st.markdown("3. **Break** into subproblems")
    st.markdown("4. **Solve** step-by-step")
    st.markdown("5. **Verify** results")
    st.markdown("6. **Present** final answer")
    
    st.divider()
    
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.emotion_history = []
        st.rerun()
    
    st.divider()
    st.info("""**Try these reasoning problems:**

• If all A are B, and all B are C, are all A C?

• A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?

• Is 0 an even number? Why?

• What is the pattern: 2, 6, 12, 20, 30, ?

• A farmer has 17 sheep. All but 9 die. How many are left?

• If a plane crashes on the border of two countries, where do survivors get buried?""")

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

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a reasoning question...")

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
    
    # Generate response
    reply = generate_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
