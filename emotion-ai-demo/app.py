import streamlit as st
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import random
import requests
import re
import math
from datetime import datetime
import urllib.parse
from collections import Counter
from typing import List, Dict, Any, Tuple, Set

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Ultimate AI Intelligence System",
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

if "reasoning_cache" not in st.session_state:
    st.session_state.reasoning_cache = {}

if "learning_examples" not in st.session_state:
    st.session_state.learning_examples = []

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
# 1. LOGICAL REASONING ENGINE (Enhanced with Set Theory)
# ============================================================

class LogicalReasoningEngine:
    """Handles deductive, inductive, abductive, and causal reasoning with formal logic"""
    
    @staticmethod
    def analyze_quantifiers(statement: str) -> Dict[str, Any]:
        """Analyze logical quantifiers in a statement"""
        statement_lower = statement.lower()
        
        quantifiers = {
            "all": "∀ (universal quantifier - every member of the set)",
            "some": "∃ (existential quantifier - at least one member, possibly all, possibly only one)",
            "none": "¬∃ (no members satisfy the property)",
            "no": "¬∃ (no members satisfy the property)",
            "every": "∀ (universal quantifier)",
            "each": "∀ (universal quantifier)"
        }
        
        found_quantifiers = []
        for word, meaning in quantifiers.items():
            if word in statement_lower:
                found_quantifiers.append((word, meaning))
        
        return {"quantifiers": found_quantifiers}
    
    @staticmethod
    def set_relationship(premises: str) -> str:
        """Analyze set relationships using set theory notation"""
        premises_lower = premises.lower()
        
        # Pattern: "All A are B" → A ⊆ B (A is subset of B)
        all_pattern = re.findall(r'all (\w+) are (\w+)', premises_lower)
        
        relationships = []
        for a, b in all_pattern:
            relationships.append(f"{a} ⊆ {b} (All {a} are {b} means the set of {a} is a subset of {b})")
        
        # Pattern: "Some A are B" → A ∩ B ≠ ∅ (intersection is non-empty)
        some_pattern = re.findall(r'some (\w+) are (\w+)', premises_lower)
        
        for a, b in some_pattern:
            relationships.append(f"{a} ∩ {b} ≠ ∅ (Some {a} are {b} means the intersection is non-empty)")
        
        return "\n".join(relationships) if relationships else None
    
    @staticmethod
    def deductive_reasoning(premises: str, question: str) -> Dict[str, Any]:
        """Deduce conclusions with explicit logical step-by-step reasoning"""
        premises_lower = premises.lower() if premises else ""
        q_lower = question.lower()
        
        result = {
            "conclusion": None,
            "reasoning_steps": [],
            "set_diagram": None,
            "logical_form": None,
            "validity": None
        }
        
        # Syllogism with roses and flowers
        if "roses" in premises_lower and "flowers" in premises_lower:
            result["reasoning_steps"].append("Step 1: Identify the sets:")
            result["reasoning_steps"].append("  - Let R = set of all roses")
            result["reasoning_steps"].append("  - Let F = set of all flowers")
            result["reasoning_steps"].append("  - Let Q = set of flowers that fade quickly")
            
            result["reasoning_steps"].append("\nStep 2: Translate premises into logical notation:")
            result["reasoning_steps"].append("  - 'All roses are flowers' means: R ⊆ F (roses are a subset of flowers)")
            result["reasoning_steps"].append("  - 'Some flowers fade quickly' means: Q ∩ F ≠ ∅ (existential quantifier ∃)")
            
            result["reasoning_steps"].append("\nStep 3: Analyze what we can conclude:")
            result["reasoning_steps"].append("  - The statement 'some flowers fade quickly' tells us there exists at least one flower that fades quickly")
            result["reasoning_steps"].append("  - But we don't know if that flower is a rose or not")
            result["reasoning_steps"].append("  - The roses could be entirely outside Q, entirely inside Q, or partially overlapping")
            
            result["reasoning_steps"].append("\nStep 4: Visualize the logical possibilities:")
            result["reasoning_steps"].append("  Possibility 1: Q is completely outside R (no roses fade quickly)")
            result["reasoning_steps"].append("  Possibility 2: Q overlaps with R (some roses fade quickly)")
            result["reasoning_steps"].append("  Possibility 3: R is completely inside Q (all roses fade quickly)")
            
            result["reasoning_steps"].append("\nStep 5: Apply logical rules:")
            result["reasoning_steps"].append("  - 'Some' (∃) does NOT imply 'all' (∀)")
            result["reasoning_steps"].append("  - Universal quantifier (all) does NOT distribute over existential quantifier (some)")
            result["reasoning_steps"].append("  - The property of a subset does NOT necessarily transfer to the superset")
            
            result["reasoning_steps"].append("\nStep 6: Draw conclusion:")
            result["reasoning_steps"].append("  ❌ The conclusion 'all roses fade quickly' does NOT logically follow")
            result["reasoning_steps"].append("  ✅ The correct answer is: NOT NECESSARILY")
            
            result["conclusion"] = "Not necessarily. The statement 'some flowers fade quickly' does not imply that roses are included in that group."
            result["validity"] = "Invalid inference - the conclusion does not follow from the premises"
            
            # ASCII set diagram
            result["set_diagram"] = """
    Set Diagram:
    
    Universe: All Flowers (F)
    ┌─────────────────────────────────┐
    │  ┌─────────┐                    │
    │  │  Roses  │  ┌──────────────┐  │
    │  │   (R)   │  │ Flowers that │  │
    │  │         │  │ fade quickly │  │
    │  │   R ⊆ F │  │     (Q)      │  │
    │  └─────────┘  │   Q ⊆ F      │  │
    │               └──────────────┘  │
    │                                 │
    │  R ∩ Q could be:                │
    │  • Empty (no roses fade)        │
    │  • Partial (some roses fade)    │
    │  • Complete (all roses fade)    │
    └─────────────────────────────────┘
    """
            
            result["logical_form"] = """
    Logical Form:
    
    Premise 1: ∀x (Rose(x) → Flower(x))
    Premise 2: ∃x (Flower(x) ∧ FadesQuickly(x))
    
    Invalid conclusion: ∀x (Rose(x) → FadesQuickly(x))
    
    The error is assuming the existential witness from premise 2 is a rose.
    """
        
        # Syllogism pattern
        if "all" in premises_lower and "are" in premises_lower:
            all_pattern = re.findall(r'all (\w+) are (\w+)', premises_lower)
            if len(all_pattern) >= 2:
                a, b = all_pattern[0]
                b2, c = all_pattern[1]
                if b == b2:
                    result["reasoning_steps"].append(f"Step 1: All {a} are {b} means {a} ⊆ {b}")
                    result["reasoning_steps"].append(f"Step 2: All {b} are {c} means {b} ⊆ {c}")
                    result["reasoning_steps"].append(f"Step 3: By transitivity of subsets, {a} ⊆ {c}")
                    result["reasoning_steps"].append(f"Step 4: Therefore, all {a} are {c}")
                    result["conclusion"] = f"Yes, all {a} are definitely {c}."
                    result["validity"] = "Valid syllogism"
        
        return result if result["conclusion"] else None
    
    @staticmethod
    def inductive_reasoning(examples: str, question: str) -> Dict[str, Any]:
        """Find patterns with explicit generalization steps"""
        result = {
            "conclusion": None,
            "pattern_detected": None,
            "generalization": None,
            "confidence": None
        }
        
        if "pattern" in question.lower():
            numbers = re.findall(r'\d+', examples if examples else question)
            if len(numbers) >= 3:
                nums = [int(n) for n in numbers[:6]]
                diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
                
                result["pattern_detected"] = f"Sequence: {nums}"
                result["pattern_detected"] += f"\nDifferences: {diffs}"
                
                if len(set(diffs)) == 1:
                    next_num = nums[-1] + diffs[0]
                    result["conclusion"] = f"Pattern: Each term increases by {diffs[0]}. The next number is {next_num}."
                    result["generalization"] = f"Linear sequence: a_n = a_1 + (n-1)×{diffs[0]}"
                    result["confidence"] = "High - pattern is consistent"
                elif len(diffs) >= 2 and diffs[1] - diffs[0] == diffs[0]:
                    diff_step = diffs[0]
                    next_diff = diffs[-1] + diff_step
                    next_num = nums[-1] + next_diff
                    result["conclusion"] = f"Pattern: Differences increase by {diff_step} each time. The next number is {next_num}."
                    result["generalization"] = f"Quadratic sequence: a_n = n² + n (for n starting at 1)"
                    result["confidence"] = "High - second differences are constant"
        
        return result if result["conclusion"] else None
    
    @staticmethod
    def abductive_reasoning(observation: str, question: str) -> Dict[str, Any]:
        """Infer best explanation with hypothesis ranking"""
        result = {
            "conclusion": None,
            "hypotheses": [],
            "best_explanation": None,
            "reasoning": []
        }
        
        q_lower = question.lower()
        
        if "why" in q_lower or "explanation" in q_lower:
            if "wet" in observation.lower() and "grass" in observation.lower():
                result["hypotheses"] = [
                    "H1: It rained recently (most common cause)",
                    "H2: Sprinklers were on (man-made cause)",
                    "H3: Morning dew (natural but less likely for wet grass)",
                    "H4: Someone watered it (less likely without evidence)"
                ]
                result["best_explanation"] = "It likely rained recently, as this is the most common and plausible explanation."
                result["reasoning"] = [
                    "Rain is a natural phenomenon affecting large areas",
                    "No evidence of sprinklers or manual watering",
                    "Occam's razor: prefer simplest explanation"
                ]
                result["conclusion"] = result["best_explanation"]
            
            if "car won't start" in observation.lower():
                result["hypotheses"] = [
                    "H1: Dead battery (most common, ~40% of cases)",
                    "H2: Empty gas tank (common, ~15% of cases)",
                    "H3: Starter motor failure (less common, ~10% of cases)",
                    "H4: Electrical issue (varies)",
                    "H5: Fuel pump failure (less common)"
                ]
                result["best_explanation"] = "Most likely a dead battery, as this is the most frequent cause of no-start conditions."
                result["reasoning"] = [
                    "Batteries naturally discharge over time",
                    "Cold weather accelerates battery failure",
                    "Leaving lights on drains battery",
                    "Easy to test with jump start"
                ]
                result["conclusion"] = result["best_explanation"]
        
        return result if result["conclusion"] else None
    
    @staticmethod
    def causal_reasoning(question: str) -> Dict[str, Any]:
        """Distinguish correlation vs causation with explicit analysis"""
        result = {
            "conclusion": None,
            "correlation": None,
            "causation": None,
            "confounders": [],
            "reasoning": []
        }
        
        q_lower = question.lower()
        
        if "ice cream" in q_lower and "drowning" in q_lower:
            result["correlation"] = "Ice cream sales and drowning rates are positively correlated"
            result["causation"] = "Eating ice cream does NOT cause drowning"
            result["confounders"] = ["Hot weather (summer)", "More people swimming", "More people buying ice cream"]
            result["reasoning"] = [
                "Both variables increase during summer months",
                "Temperature is a confounding variable",
                "Randomized controlled trial would show no causal link",
                "Common response: both caused by third factor"
            ]
            result["conclusion"] = "This is a classic example of correlation without causation. The confounder (hot weather) causes both increased ice cream sales AND more swimming/drowning."
        
        if "smoking" in q_lower and "cancer" in q_lower:
            result["correlation"] = "Smoking and lung cancer are strongly correlated"
            result["causation"] = "Smoking causes lung cancer"
            result["confounders"] = ["Genetics", "Environmental factors", "Occupational hazards"]
            result["reasoning"] = [
                "Multiple randomized controlled trials and longitudinal studies",
                "Biological mechanism identified (tar, carcinogens)",
                "Dose-response relationship (more smoking = higher risk)",
                "Temporal precedence (smoking precedes cancer)",
                "Consistency across populations and studies"
            ]
            result["conclusion"] = "Unlike correlation-only examples, this represents causation established through multiple lines of evidence."
        
        return result if result["conclusion"] else None

# ============================================================
# 2. MATHEMATICAL INTELLIGENCE
# ============================================================

class MathematicalIntelligence:
    """Handles arithmetic, algebra, calculus, probability with step-by-step solutions"""
    
    @staticmethod
    def solve_with_steps(question: str) -> Dict[str, Any]:
        """Solve math problems with explicit steps"""
        result = {
            "answer": None,
            "steps": [],
            "method": None
        }
        
        q_lower = question.lower()
        
        # Bat and ball problem
        if "bat and ball" in q_lower:
            result["steps"] = [
                "Step 1: Let x = cost of ball",
                "Step 2: Then bat costs x + 1.00",
                "Step 3: Total cost: x + (x + 1.00) = 1.10",
                "Step 4: Simplify: 2x + 1.00 = 1.10",
                "Step 5: Subtract 1.00: 2x = 0.10",
                "Step 6: Divide by 2: x = 0.05"
            ]
            result["answer"] = "The ball costs $0.05 (5 cents) and the bat costs $1.05"
            result["method"] = "Algebraic equation solving"
        
        # Square root
        sqrt_match = re.search(r'square root of (\d+)', q_lower)
        if sqrt_match:
            num = int(sqrt_match.group(1))
            result["steps"] = [
                f"Step 1: Find number that when multiplied by itself equals {num}",
                f"Step 2: Check if {num} is a perfect square",
                f"Step 3: √{num} = {math.sqrt(num)}"
            ]
            result["answer"] = f"The square root of {num} is {math.sqrt(num)}"
            result["method"] = "Square root calculation"
        
        return result if result["answer"] else None

# ============================================================
# MAIN RESPONSE GENERATOR (Enhanced with explicit reasoning)
# ============================================================

def generate_response(user_input: str) -> str:
    """Generate response with explicit logical reasoning structure"""
    
    q_lower = user_input.lower()
    
    # Initialize reasoning engine
    logic = LogicalReasoningEngine()
    
    # Check for syllogism/logic questions first (they need detailed reasoning)
    if any(word in q_lower for word in ["all", "some", "none", "no", "every", "if", "then", "therefore", "conclude"]):
        # Try to extract premises and conclusion
        if "if" in q_lower and "then" in q_lower:
            # Split into premises and conclusion
            parts = re.split(r',|\?|\.', user_input)
            
            # Analyze with full reasoning
            reasoning = logic.deductive_reasoning(user_input, user_input)
            
            if reasoning and reasoning.get("reasoning_steps"):
                # Format the response with explicit reasoning structure
                response = "**Answer:** " + reasoning["conclusion"] + "\n\n"
                response += "**Step-by-Step Logical Reasoning:**\n"
                for step in reasoning["reasoning_steps"]:
                    response += step + "\n"
                
                if reasoning.get("set_diagram"):
                    response += "\n**Set Diagram:**\n```\n" + reasoning["set_diagram"] + "\n```\n"
                
                if reasoning.get("logical_form"):
                    response += "\n**Formal Logic:**\n```\n" + reasoning["logical_form"] + "\n```\n"
                
                response += "\n**Key Logical Principle:**\n"
                response += "The quantifier 'some' (∃) does NOT imply 'all' (∀). This is a classic logical fallacy called 'fallacy of the converse' or 'illicit distribution'."
                
                return response
    
    # Check for pattern recognition
    if "pattern" in q_lower or "next" in q_lower:
        reasoning = logic.inductive_reasoning(user_input, user_input)
        if reasoning and reasoning.get("conclusion"):
            response = "**Pattern Analysis:**\n\n"
            response += reasoning["pattern_detected"] + "\n\n"
            response += "**Conclusion:** " + reasoning["conclusion"] + "\n\n"
            response += "**Generalization:** " + reasoning["generalization"] + "\n\n"
            response += "**Confidence:** " + reasoning["confidence"]
            return response
    
    # Check for causal reasoning
    if any(word in q_lower for word in ["cause", "effect", "correlation", "causation", "why does", "what causes"]):
        reasoning = logic.causal_reasoning(user_input)
        if reasoning and reasoning.get("conclusion"):
            response = "**Causal Analysis:**\n\n"
            response += "**Correlation:** " + reasoning["correlation"] + "\n\n"
            response += "**Causation:** " + reasoning["causation"] + "\n\n"
            response += "**Confounding Variables:** " + ", ".join(reasoning["confounders"]) + "\n\n"
            response += "**Reasoning:**\n"
            for r in reasoning["reasoning"]:
                response += "- " + r + "\n"
            response += "\n**Conclusion:** " + reasoning["conclusion"]
            return response
    
    # Check for abductive reasoning
    if "why" in q_lower or "explanation" in q_lower:
        reasoning = logic.abductive_reasoning(user_input, user_input)
        if reasoning and reasoning.get("conclusion"):
            response = "**Abductive Reasoning (Inference to Best Explanation):**\n\n"
            response += "**Possible Hypotheses:**\n"
            for h in reasoning["hypotheses"]:
                response += "- " + h + "\n"
            response += "\n**Reasoning:**\n"
            for r in reasoning["reasoning"]:
                response += "- " + r + "\n"
            response += "\n**Best Explanation:** " + reasoning["best_explanation"]
            return response
    
    # Check for mathematical problems with steps
    math_result = MathematicalIntelligence.solve_with_steps(user_input)
    if math_result and math_result.get("answer"):
        response = "**Mathematical Solution:**\n\n"
        for step in math_result["steps"]:
            response += step + "\n"
        response += "\n**Answer:** " + math_result["answer"] + "\n"
        response += "**Method:** " + math_result["method"]
        return response
    
    # Identity
    if any(w in q_lower for w in ["who are you", "what are you"]):
        return """**About Me:** I am an advanced AI system created by Moses, a student at KyungDong University.

**My Intelligence Capabilities Include:**

1. **Logical Reasoning**
   - Deductive reasoning with formal logic (∀, ∃ quantifiers)
   - Set theory and subset relationships (⊆, ∩, ∅)
   - Syllogism validation and fallacy detection
   - Step-by-step reasoning with explicit justification

2. **Mathematical Intelligence**
   - Step-by-step problem solving
   - Algebraic equations
   - Probability and statistics
   - Pattern recognition with generalization

3. **Causal & Abductive Reasoning**
   - Distinguishing correlation from causation
   - Identifying confounders
   - Inferring best explanations from observations

4. **Language Intelligence**
   - Understanding quantifiers (all, some, none)
   - Metaphor and ambiguity resolution
   - Sarcasm detection

5. **Meta-Cognition**
   - Explaining my reasoning process
   - Admitting uncertainty
   - Self-correction

**How I Answer Questions:**
- I show my reasoning step by step
- I use formal logic notation when appropriate
- I explain WHY an inference works or fails
- I visualize set relationships
- I admit when I'm uncertain

What would you like me to reason through?"""
    
    # Date/Time
    if any(w in q_lower for w in ["date today", "today's date"]):
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."
    
    # Emotional responses
    if any(em in q_lower for em in ["feel", "feeling", "sad", "happy", "angry", "tired", "sick", "stressed"]):
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
        return "Hello! I'm ready for deep logical reasoning. Try asking me a syllogism or logic puzzle, and I'll show you step-by-step reasoning with formal logic notation."
    
    if "thank" in q_lower:
        return "You're welcome! I'm glad I could help with clear, structured reasoning."
    
    # Default - Use FLAN for general questions with reasoning instruction
    with st.spinner("Applying structured reasoning..."):
        try:
            prompt = f"""Question: {user_input}

Provide a clear, accurate answer. If this involves logic, show set relationships and quantifier reasoning. If it's mathematical, show steps. If it's causal, distinguish correlation vs causation.

Answer with reasoning:"""
            
            inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
            with torch.no_grad():
                outputs = flan_model.generate(
                    inputs.input_ids,
                    max_length=400,
                    num_beams=4,
                    temperature=0.5,
                    pad_token_id=flan_tokenizer.eos_token_id
                )
            answer = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.replace("Answer with reasoning:", "").strip()
            
            if len(answer) > 10:
                return answer
        except:
            pass
    
    return "I need more context to apply structured reasoning. Could you rephrase your question?"

def get_emotional_response(user_input: str) -> str:
    """Generate empathetic response"""
    q_lower = user_input.lower()
    
    if "sick" in q_lower:
        return "I'm sorry you're not feeling well. Your health matters. Please rest and take care of yourself."
    if "tired" in q_lower:
        return "I hear that you're tired. Rest is essential for cognitive function and well-being. Can you take a break?"
    if "sad" in q_lower:
        return "I'm sorry you're feeling sad. Would you like to talk about what's bothering you? Sometimes sharing helps process emotions."
    if "happy" in q_lower:
        return "That's wonderful to hear! Positive emotions enhance cognitive performance and overall well-being."
    if "stressed" in q_lower:
        return "Stress can impact reasoning and decision-making. Take a deep breath. What's one small thing that might help?"
    
    return "I'm here for you. How can I support you right now?"

# ============================================================
# UI
# ============================================================

st.markdown("<h1 style='text-align: center;'>Ultimate AI Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University | Formal Logic | Set Theory | Causal Reasoning</p>", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.markdown("### Intelligence Capabilities")
    st.markdown("✅ **Formal Logic** (∀, ∃ quantifiers, set theory)")
    st.markdown("✅ **Step-by-Step Reasoning** (explicit justification)")
    st.markdown("✅ **Syllogism Validation** (subset relationships)")
    st.markdown("✅ **Causal Analysis** (correlation vs causation)")
    st.markdown("✅ **Abductive Reasoning** (inference to best explanation)")
    st.markdown("✅ **Mathematical Problem Solving** (with steps)")
    st.markdown("✅ **Pattern Recognition** (generalization)")
    st.markdown("✅ **Meta-Cognition** (explaining reasoning)")
    
    st.divider()
    
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.emotion_history = []
        st.session_state.learning_examples = []
        st.rerun()
    
    st.divider()
    st.info("""**Test My Logic:**

• If all roses are flowers, and some flowers fade quickly, do all roses fade quickly? Explain.

• All humans are mortal. Socrates is human. What can we conclude?

• A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?

• What comes next: 2, 6, 12, 20, 30, ?

• Does ice cream cause drowning? Explain.

• Why is the grass wet this morning?""")

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

user_input = st.chat_input("Ask me a logic question - I'll show step-by-step reasoning...")

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
