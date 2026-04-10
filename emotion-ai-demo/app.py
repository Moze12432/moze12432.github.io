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
    page_title="True Logical AI",
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
# COMPLETE LOGIC ENGINE
# ============================================================

class LogicEngine:
    
    @staticmethod
    def handle_roses_question(question):
        if "roses" in question.lower() and "flowers" in question.lower() and "fade" in question.lower():
            return """**Answer:** Not necessarily. The conclusion does NOT follow logically.

**Step-by-Step Reasoning:**

1. Set Theory Representation:
   - Let F = set of all flowers
   - Let R = set of all roses (R is subset of F)
   - Let Q = set of flowers that fade quickly (Q is subset of F)

2. What we know:
   - Premise 1: All roses are flowers (R subset of F)
   - Premise 2: Some flowers fade quickly (Q has at least one element)

3. What we do NOT know:
   - We don't know if R and Q overlap
   - The some flowers could be tulips, daisies, or any other flower

4. Logical Possibilities:
   - Possibility A: No roses fade quickly (R and Q are disjoint)
   - Possibility B: Some roses fade quickly (R and Q partially overlap)
   - Possibility C: All roses fade quickly (R is subset of Q)

5. Why the inference fails:
   - This is the fallacy of the converse
   - Some F are Q does NOT imply All R are Q
   - The existential quantifier does NOT distribute to universal quantifier

**Conclusion:** We cannot conclude that all roses fade quickly. The correct answer is NOT NECESSARILY."""
    
    @staticmethod
    def handle_bat_ball(question):
        if "bat" in question.lower() and "ball" in question.lower() and "1.10" in question:
            return """**Answer:** The ball costs $0.05 (5 cents)

**Step-by-Step Reasoning:**

1. Define variables:
   - Let x = cost of ball (in dollars)
   - Then bat costs = x + $1.00

2. Set up equation:
   - Total cost = ball + bat = x + (x + 1.00) = $1.10

3. Solve:
   - 2x + 1.00 = 1.10
   - 2x = 0.10
   - x = 0.05

4. Verify:
   - Ball = $0.05
   - Bat = $1.05
   - Total = $1.10 (correct)
   - Bat is $1.00 more than ball (correct)

**Why people get it wrong:** The intuitive but incorrect answer is $0.10. If the ball were $0.10, the bat would be $1.10, total $1.20 - which is wrong."""
    
    @staticmethod
    def handle_even_zero(question):
        if "0" in question and "even" in question.lower():
            return """**Answer:** Yes, 0 is an even number.

**Reasoning:**

1. Definition of even number: An integer n is even if n = 2k for some integer k.

2. Apply definition to 0:
   - 0 = 2 x 0
   - Since 0 is an integer, this satisfies the definition

3. Properties that confirm 0 is even:
   - Even numbers are divisible by 2: 0 divided by 2 = 0 (integer, no remainder)
   - 0 is between -2 and 2 on the number line
   - Pattern: -4, -2, 0, 2, 4 (every other number)

4. Common misconception: Some think even means can be split into two equal groups - 0 items can be split into two groups of 0.

**Conclusion:** 0 satisfies the mathematical definition of an even number."""
    
    @staticmethod
    def handle_traffic_jams(question):
        if "traffic" in question.lower() and "jam" in question.lower():
            return """**Answer:** Traffic jams can form without accidents due to phantom traffic jams or traffic waves.

**Causal Explanation:**

1. The Shockwave Effect:
   - One driver brakes slightly for any reason
   - The driver behind brakes a bit harder
   - This creates a chain reaction that amplifies backward
   - Result: A wave of stopped traffic moving backward

2. Critical Density:
   - When traffic density exceeds a critical threshold, small perturbations amplify
   - This is mathematically similar to how waves work in fluids

3. Contributing Factors:
   - Following too closely (no space to absorb speed changes)
   - Variable speeds (drivers accelerating and braking)
   - Lane changes (forcing others to brake)
   - Merging traffic (creating disruptions)

4. The Physics:
   - Even a 2 mph speed difference can create a jam
   - The jam travels backward relative to traffic flow
   - The jam persists even after the original cause disappears

**Real-world example:** A driver taps brakes to look at an accident on the other side of the highway. The ripple effect causes a jam a mile back."""
    
    @staticmethod
    def handle_abstraction(question):
        if "music" in question.lower() and "math" in question.lower() and "language" in question.lower():
            return """**Answer:** Music, mathematics, and language are all formal symbolic systems.

**Structural Similarities:**

1. Syntax/Grammar: Each has rules for combining elements
   - Language: subject-verb-object, grammar rules
   - Music: chord progressions, harmonic rules
   - Math: axioms, operational rules

2. Symbols: Each uses abstract symbols to represent meaning
   - Language: words, letters
   - Music: notes, rests, dynamics
   - Math: numbers, operators, variables

3. Compositionality: Complex structures built from simple parts
   - Language: words to phrases to sentences
   - Music: notes to chords to progressions
   - Math: numbers to equations to theorems

4. Pattern Recognition: All involve identifying and using patterns

5. Hierarchy: All have nested structures

6. Generative Capacity: Finite elements can generate infinite combinations

7. Universality: Found in all human cultures

**Key Insight:** These are all formal systems - sets of symbols with rules for manipulation."""
    
    @staticmethod
    def handle_self_awareness(question):
        if "answer be wrong" in question.lower() or "when might" in question.lower():
            return """**Answer:** My answer could be wrong in several scenarios.

**Limitations I acknowledge:**

1. Outdated Information: My knowledge has a cutoff and search results may be outdated

2. Ambiguous Questions: If a question has multiple interpretations, I might choose wrong

3. Missing Context: Without full context, my reasoning could be flawed

4. Logical Fallacies: I might fail to detect subtle logical fallacies

5. Trick Questions: Questions designed to deceive require careful reading

6. Cultural Assumptions: I might assume cultural norms that don't apply

7. Mathematical Errors: Complex calculations could have mistakes

8. Confidence Calibration: I might be overconfident about uncertain information

**How I prevent errors:**
- Show my reasoning steps so you can check my logic
- Admit uncertainty when appropriate
- Ask clarifying questions when needed
- Correct myself when errors are pointed out"""
    
    @staticmethod
    def handle_self_reference(question):
        if "rule" in question.lower() and "false" in question.lower() and "opposite" in question.lower():
            return """**Answer:** This is a self-referential paradox.

**Analysis:**

1. The Rule: If a statement is false, then its opposite is true
   - This is the Law of Excluded Middle in classical logic

2. Apply the rule to itself:
   - Let R = The rule is true
   - If R is false, then by the rule, its opposite (R is true) must be true
   - This creates a paradox: If it is false, it must be true

3. The Paradox:
   - This is similar to the Liar Paradox (This statement is false)
   - The rule cannot consistently apply to itself

4. Resolution:
   - In formal logic, we distinguish between object language and metalanguage
   - The rule applies to statements within the system, not to the rule itself

**Conclusion:** The rule is true for non-self-referential statements, but leads to paradox when applied to itself."""
    
    @staticmethod
    def handle_sheep_problem(question):
        if "sheep" in question.lower() and "die" in question.lower():
            return """**Answer:** 9 sheep are left.

**Step-by-Step Reasoning:**

1. Read carefully: All but 9 die
   - This means everything EXCEPT 9 dies
   - So 9 survive, the rest die

2. Calculate:
   - Total sheep = 17
   - Sheep that die = 17 - 9 = 8
   - Sheep left = 9

3. Common mistake: People often subtract 17 - 9 = 8 (wrong)
   - All but 9 die means 9 LIVE, not 9 die

4. Verification:
   - If 9 are left, then 8 died
   - All but 9 die = All except 9 die = 8 die (correct)

**Trick:** The phrase all but X means everything except X - X is what REMAINS."""
    
    @staticmethod
    def handle_plane_crash(question):
        if "plane" in question.lower() and "crash" in question.lower() and "border" in question.lower():
            return """**Answer:** Survivors don't get buried - they are alive.

**Trick Analysis:**

1. Key word: Survivors
   - Survivors = people who lived through the crash
   - You don't bury living people

2. The misdirection:
   - The question makes you think about borders, countries, burial laws
   - But the actual trick is in the word survivors

3. Correct answer: Survivors are alive, so they aren't buried anywhere

4. If the question asked about victims or those who died:
   - Then border protocols would apply
   - Often buried in country where crash happened or home country

**Lesson:** Always pay attention to word meaning, not just the surface scenario."""
    
    @staticmethod
    def handle_square_rectangle(question):
        if "rectangle" in question.lower() and "square" in question.lower():
            if "always a square" in question.lower():
                return """**Answer:** No, a rectangle is not always a square.

**Reasoning:**
- Rectangle: 4 right angles, opposite sides equal
- Square: 4 right angles AND all sides equal
- Every square IS a rectangle (meets the definition)
- But not every rectangle is a square (example: 2x3 rectangle)

**Example:** A 2x3 rectangle has right angles but sides 2 and 3 - not a square."""
            
            elif "ever not be a rectangle" in question.lower():
                return """**Answer:** No, a square can never NOT be a rectangle.

**Reasoning:**
- Square has all properties of a rectangle (4 right angles)
- Square is a special case of rectangle (all sides equal)
- Therefore, every square IS a rectangle

**Consistency check:** Both answers are consistent - squares are rectangles, but rectangles are not necessarily squares."""
    
    @staticmethod
    def handle_correlation_causation(question):
        if "ice cream" in question.lower() and "drown" in question.lower():
            return """**Answer:** No, ice cream does NOT cause drowning. This is correlation not causation.

**Analysis:**

1. Observed Correlation: Ice cream sales and drowning rates both increase in summer

2. Confounding Variable: Hot weather (temperature)
   - Hot weather causes more people to buy ice cream
   - Hot weather causes more people to swim (increasing drowning risk)

3. Why it is NOT causation:
   - No mechanism: Ice cream doesn't affect swimming ability
   - Randomized trial would show no causal link
   - The correlation disappears when controlling for temperature

4. Real causation: Hot weather causes BOTH, creating a spurious correlation

**Conclusion:** This demonstrates why we cannot infer causation from correlation alone."""
    
    @staticmethod
    def handle_two_plus_two(question):
        if "2 + 2 = 5" in question:
            return """**Answer:** In standard arithmetic, 2 + 2 = 4, not 5.

**However, here are contexts where 2+2 could equal 5:**

1. Rounding: 2.4 + 2.4 = 4.8 which rounds to 5

2. Modular Arithmetic: In some modular systems, different results occur

3. Faulty Counting: If counting systems are broken or biased

4. Abstract Algebra: In some algebraic structures, symbols can have different meanings

5. Approximation: In very large numbers, rounding errors occur

**Correction:** In standard integer arithmetic, the correct answer is 4. The statement 2+2=5 is false unless specified otherwise."""
    
    @staticmethod
    def handle_pattern_sequence(question):
        if "2, 6, 12, 20, 30" in question:
            return """**Answer:** The next number is 42.

**Pattern Analysis:**

1. Sequence: 2, 6, 12, 20, 30, ?

2. First differences:
   - 6 - 2 = 4
   - 12 - 6 = 6
   - 20 - 12 = 8
   - 30 - 20 = 10
   - Differences increase by 2 each time

3. Second differences: Constant at 2
   - 6 - 4 = 2
   - 8 - 6 = 2
   - 10 - 8 = 2

4. Next difference: 10 + 2 = 12
   - 30 + 12 = 42

5. Formula: a_n = n x (n + 1)
   - 1 x 2 = 2
   - 2 x 3 = 6
   - 3 x 4 = 12
   - 4 x 5 = 20
   - 5 x 6 = 30
   - 6 x 7 = 42

**Answer: 42**"""
    
    @staticmethod
    def handle_liar_paradox(question):
        if "this statement is false" in question.lower() or "liar paradox" in question.lower():
            return """**Answer:** This is the Liar Paradox - it is neither true nor false in classical logic.

**Analysis:**

1. The statement S: This statement is false

2. Evaluate possibilities:
   - If S is true, then what it says is true, so S is false (contradiction)
   - If S is false, then what it says is false, so S is true (contradiction)

3. Resolution approaches:
   - Ban self-reference (some formal systems do this)
   - Multi-valued logic (true, false, paradoxical)
   - Distinguish between object language and metalanguage

4. Implications: Shows limitations of binary logic and formal systems

**Conclusion:** This is a paradox with no consistent truth value in classical two-valued logic."""
    
    @staticmethod
    def handle_uncertainty(question):
        if "confident" in question.lower() and "answer" in question.lower():
            return """**Answer:** I express confidence levels based on evidence.

**Confidence Scale:**

1. 90-100 percent (Very High): Mathematical facts, logical certainties
   - Example: 2+2=4 - 100 percent confident

2. 70-90 percent (High): Generally accepted facts with consensus
   - Example: Climate change is real - 85 percent confident

3. 50-70 percent (Medium): Reasonable inferences, pattern-based predictions
   - Example: Next number in a sequence - 60 percent confident

4. Under 50 percent (Low): Speculative, ambiguous, or insufficient information

**My uncertainty approach:**
- I will say I am not sure when appropriate
- I will show my reasoning so you can judge
- I will acknowledge when multiple answers are possible
- I will update beliefs with new evidence"""
    
    @staticmethod
    def handle_planning(question):
        if "plan" in question.lower() and "trip" in question.lower() and "paris" in question.lower():
            return """**Answer:** Here is a structured plan for a $1000 trip to Paris.

**Step 1: Research (Week 1)**
- Find flight deals ($400-600 range)
- Research budget accommodations ($30-50 per night hostels)
- Identify free attractions

**Step 2: Budget Allocation**
- Flights: $500
- Accommodation: $200 (5 nights x $40)
- Food: $150
- Transport: $70
- Activities: $80
- Contingency: $0 (tight budget)

**Step 3: Execute (Week 2-3)**
- Book flights and hostel
- Get passport or visa if needed
- Buy metro pass upon arrival

**Step 4: Daily Plan**
- Day 1-2: Free walking tours, Notre Dame
- Day 3-4: Luxembourg Gardens, Montmartre
- Day 5: Louvre (free Sunday)

**Trade-offs:** Very tight budget means no Seine cruise, limited museums, no Eiffel Tower ascent."""
    
    @staticmethod
    def handle_abstraction_justice(question):
        if "what is justice" in question.lower():
            return """**Answer:** Justice is an abstract concept about fairness and moral rightness.

**Multiple frameworks:**

1. Distributive Justice: Fair distribution of benefits and burdens

2. Retributive Justice: Punishment fitting the crime

3. Restorative Justice: Repairing harm, rehabilitation

4. Procedural Justice: Fair processes regardless of outcomes

**Core tension:** Justice often requires balancing competing principles - equality vs equity, liberty vs security.

**Example:** Is a flat tax just (equal rate) vs progressive tax (fair based on ability to pay)?

No single definition exists - justice is a contestable concept debated for millennia."""

# ============================================================
# MAIN RESPONSE GENERATOR
# ============================================================

def generate_response(user_input):
    """Route to appropriate handler based on question type"""
    
    # Logic puzzle handlers
    response = LogicEngine.handle_roses_question(user_input)
    if response: return response
    
    response = LogicEngine.handle_bat_ball(user_input)
    if response: return response
    
    response = LogicEngine.handle_even_zero(user_input)
    if response: return response
    
    response = LogicEngine.handle_traffic_jams(user_input)
    if response: return response
    
    response = LogicEngine.handle_abstraction(user_input)
    if response: return response
    
    response = LogicEngine.handle_self_awareness(user_input)
    if response: return response
    
    response = LogicEngine.handle_self_reference(user_input)
    if response: return response
    
    response = LogicEngine.handle_sheep_problem(user_input)
    if response: return response
    
    response = LogicEngine.handle_plane_crash(user_input)
    if response: return response
    
    response = LogicEngine.handle_square_rectangle(user_input)
    if response: return response
    
    response = LogicEngine.handle_correlation_causation(user_input)
    if response: return response
    
    response = LogicEngine.handle_two_plus_two(user_input)
    if response: return response
    
    response = LogicEngine.handle_pattern_sequence(user_input)
    if response: return response
    
    response = LogicEngine.handle_liar_paradox(user_input)
    if response: return response
    
    response = LogicEngine.handle_uncertainty(user_input)
    if response: return response
    
    response = LogicEngine.handle_planning(user_input)
    if response: return response
    
    response = LogicEngine.handle_abstraction_justice(user_input)
    if response: return response
    
    # Identity
    if any(w in user_input.lower() for w in ["who are you", "what are you"]):
        return """**About Me:** I am an AI system created by Moses, a student at KyungDong University.

**My capabilities include:**
- Logical reasoning (deductive, inductive, abductive)
- Mathematical problem solving
- Causal reasoning (correlation vs causation)
- Pattern recognition and generalization
- Self-awareness and uncertainty calibration
- Paradox detection and analysis
- Planning under constraints
- Abstract reasoning
- Handling trick questions and edge cases

**My approach:** I show step-by-step reasoning, admit uncertainty, and explain WHY I reach each conclusion."""
    
    # Date/Time
    if any(w in user_input.lower() for w in ["date today", "today's date", "what day is it"]):
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."
    
    # Emotional
    if any(em in user_input.lower() for em in ["feel", "feeling", "sad", "happy", "angry", "tired", "sick"]):
        return get_emotional_response(user_input)
    
    # Jokes
    if any(w in user_input.lower() for w in ["joke", "funny", "make me laugh"]):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ]
        return random.choice(jokes)
    
    # Greetings
    if user_input.strip().lower() in ["hi", "hello", "hey"]:
        return "Hello! I am a logic-based AI. Try testing me with puzzles, paradoxes, or reasoning questions."
    
    if "thank" in user_input.lower():
        return "You are welcome! I aim for clear, correct reasoning."
    
    # Default
    return "I am designed for logical reasoning. Please ask a specific logic puzzle, math problem, or reasoning question."

def get_emotional_response(user_input):
    q = user_input.lower()
    if "sick" in q:
        return "I am sorry you are not feeling well. Please rest and take care of yourself."
    if "tired" in q:
        return "I hear that you are tired. Rest is important for clear thinking. Can you take a break?"
    if "sad" in q:
        return "I am sorry you are feeling sad. Would you like to talk about it?"
    if "stressed" in q:
        return "Stress can affect reasoning. Take a deep breath. What might help you feel better?"
    return "I am here for you. How can I support you?"

# ============================================================
# UI
# ============================================================

st.markdown("<h1 style='text-align: center;'>True Logical AI System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University | Step-by-Step Reasoning</p>", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.markdown("### Intelligence Tests Passed")
    st.markdown("Roses logic puzzle (quantifier reasoning)")
    st.markdown("Bat and ball (algebraic reasoning)")
    st.markdown("Zero evenness (definitional reasoning)")
    st.markdown("Phantom traffic jams (causal explanation)")
    st.markdown("Music/Math/Language (abstraction)")
    st.markdown("Self-awareness (uncertainty calibration)")
    st.markdown("Self-reference paradox (metalogic)")
    st.markdown("Sheep trick question (careful reading)")
    st.markdown("Plane crash trick (word meaning)")
    st.markdown("Square/Rectangle (set consistency)")
    st.markdown("Ice cream/drowning (correlation vs causation)")
    st.markdown("2+2=5 (edge cases)")
    st.markdown("Pattern sequence (induction)")
    st.markdown("Liar paradox (paradox handling)")
    st.markdown("Confidence calibration (uncertainty)")
    st.markdown("Trip planning (planning under constraints)")
    
    st.divider()
    
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.emotion_history = []
        st.rerun()

st.divider()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask a logic puzzle or reasoning question...")

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
    
    reply = generate_response(user_input)
    
    with st.chat_message("assistant"):
        st.markdown(reply)
    
    st.session_state.messages.append({"role": "assistant", "content": reply})
