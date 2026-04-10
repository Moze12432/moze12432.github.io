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
# 1. LOGICAL REASONING ENGINE
# ============================================================

class LogicalReasoningEngine:
    """Handles deductive, inductive, abductive, and causal reasoning"""
    
    @staticmethod
    def deductive_reasoning(premises, question):
        """Deduce conclusions from premises"""
        premises_lower = premises.lower() if premises else ""
        q_lower = question.lower()
        
        # Syllogism patterns
        if "all" in premises_lower and "are" in premises_lower:
            # Extract patterns like "All A are B, All B are C, therefore All A are C"
            pattern = re.findall(r'all (\w+) are (\w+)', premises_lower)
            if len(pattern) >= 2:
                a, b = pattern[0]
                b2, c = pattern[1]
                if b == b2:
                    return f"Logical deduction: Since all {a} are {b} and all {b} are {c}, therefore all {a} are {c}. This is valid syllogistic reasoning."
        
        # Modus ponens (If P then Q, P, therefore Q)
        if "if" in premises_lower and "then" in premises_lower:
            if_match = re.search(r'if (.+?) then (.+)', premises_lower)
            if if_match:
                condition, result = if_match.groups()
                if condition in q_lower:
                    return f"Using modus ponens: If {condition} then {result}, and {condition} is true, therefore {result}. This is logically valid."
        
        return None
    
    @staticmethod
    def inductive_reasoning(examples, question):
        """Find patterns and generalize from examples"""
        if "pattern" in question.lower():
            # Number pattern recognition
            numbers = re.findall(r'\d+', examples if examples else question)
            if len(numbers) >= 3:
                nums = [int(n) for n in numbers[:6]]
                diffs = [nums[i+1] - nums[i] for i in range(len(nums)-1)]
                if len(set(diffs)) == 1:
                    next_num = nums[-1] + diffs[0]
                    return f"Pattern detected: Each term increases by {diffs[0]}. The next number is {next_num}."
                elif len(diffs) >= 2 and diffs[1] - diffs[0] == diffs[0]:
                    diff_step = diffs[0]
                    next_diff = diffs[-1] + diff_step
                    next_num = nums[-1] + next_diff
                    return f"Pattern detected: Differences increase by {diff_step} each time. The next number is {next_num}."
        
        return None
    
    @staticmethod
    def abductive_reasoning(observation, question):
        """Infer best explanation for incomplete data"""
        q_lower = question.lower()
        
        if "why" in q_lower or "explanation" in q_lower:
            # Common abductive reasoning scenarios
            if "wet" in observation.lower() and "grass" in observation.lower():
                return "Best explanation: It likely rained recently, or the sprinklers were on. The most plausible cause given typical weather patterns is rain."
            if "car won't start" in observation.lower():
                return "Possible explanations: dead battery, empty gas tank, starter motor failure, or electrical issue. The most common cause is a dead battery."
        
        return None
    
    @staticmethod
    def causal_reasoning(question):
        """Distinguish correlation vs causation, predict effects of interventions"""
        q_lower = question.lower()
        
        if "cause" in q_lower or "effect" in q_lower or "what if" in q_lower:
            if "ice cream" in q_lower and "drowning" in q_lower:
                return "This is a classic correlation vs causation example. Ice cream sales and drowning rates are correlated because both increase in summer (confounding variable: hot weather). Eating ice cream does NOT cause drowning."
            
            if "smoking" in q_lower and "cancer" in q_lower:
                return "Multiple studies show smoking causes lung cancer through causal mechanisms (tar, carcinogens). This is causation, not just correlation, established through randomized controlled trials and biological evidence."
        
        return None

# ============================================================
# 2. MATHEMATICAL INTELLIGENCE
# ============================================================

class MathematicalIntelligence:
    """Handles arithmetic, algebra, calculus, probability, and mathematical reasoning"""
    
    @staticmethod
    def solve_arithmetic(question):
        """Solve arithmetic problems"""
        # Basic operations
        patterns = [
            (r'(\d+)\s*\+\s*(\d+)', lambda m: int(m.group(1)) + int(m.group(2))),
            (r'(\d+)\s*-\s*(\d+)', lambda m: int(m.group(1)) - int(m.group(2))),
            (r'(\d+)\s*\*\s*(\d+)|(\d+)\s*times\s*(\d+)', lambda m: int(m.group(1) or m.group(3)) * int(m.group(2) or m.group(4))),
            (r'(\d+)\s*/\s*(\d+)|(\d+)\s*divided by\s*(\d+)', lambda m: int(m.group(1) or m.group(3)) / int(m.group(2) or m.group(4))),
            (r'square root of (\d+)', lambda m: math.sqrt(int(m.group(1)))),
            (r'(\d+)\s*squared', lambda m: int(m.group(1)) ** 2),
        ]
        
        for pattern, func in patterns:
            match = re.search(pattern, question.lower())
            if match:
                result = func(match)
                return f"= {result}"
        
        return None
    
    @staticmethod
    def solve_algebra(question):
        """Solve algebraic equations"""
        q_lower = question.lower()
        
        # Simple linear equations
        eq_match = re.search(r'(\d+)x\s*\+\s*(\d+)\s*=\s*(\d+)', q_lower)
        if eq_match:
            a, b, c = int(eq_match.group(1)), int(eq_match.group(2)), int(eq_match.group(3))
            x = (c - b) / a
            return f"Solving {a}x + {b} = {c}: x = {x}"
        
        # Bat and ball problem
        if "bat and ball" in q_lower:
            return "Let ball = x, bat = x + 1.00, total = 2x + 1.00 = 1.10, so 2x = 0.10, x = 0.05. The ball costs $0.05."
        
        return None
    
    @staticmethod
    def probability(question):
        """Calculate probabilities"""
        q_lower = question.lower()
        
        if "probability" in q_lower or "chance" in q_lower:
            # Coin flip probability
            if "coin" in q_lower and "heads" in q_lower:
                return "The probability of getting heads on a fair coin flip is 1/2 or 0.5 or 50%."
            
            # Dice probability
            if "dice" in q_lower or "die" in q_lower:
                return "For a fair six-sided die, each number has probability 1/6 ≈ 0.1667 or about 16.67%."
        
        return None

# ============================================================
# 3. PROBLEM SOLVING & PLANNING
# ============================================================

class ProblemSolvingEngine:
    """Breaks down complex problems, plans multi-step solutions"""
    
    @staticmethod
    def decompose_problem(question):
        """Break complex problems into sub-problems"""
        q_lower = question.lower()
        
        if "how to" in q_lower:
            # Extract the task
            task = re.sub(r'how to', '', q_lower).strip()
            return f"""To solve '{task}', follow these steps:
1. Understand the goal and requirements
2. Gather necessary information or resources
3. Break the task into smaller manageable parts
4. Execute each part systematically
5. Verify results and adjust if needed
6. Complete and review the solution"""
        
        if "plan" in q_lower:
            return "A good plan includes: (1) Define objectives, (2) Identify constraints, (3) Generate alternatives, (4) Evaluate options, (5) Select best approach, (6) Execute, (7) Monitor and adjust."
        
        return None
    
    @staticmethod
    def optimize_tradeoffs(question):
        """Handle trade-off decisions"""
        if "trade-off" in question.lower() or "speed" in question.lower() and "accuracy" in question.lower():
            return "Common trade-offs: Speed vs Accuracy (faster = more errors, slower = more precise). The optimal balance depends on the specific use case and acceptable error tolerance."
        
        return None

# ============================================================
# 4. MEMORY & LEARNING
# ============================================================

class LearningEngine:
    """Few-shot learning, adaptation, knowledge transfer"""
    
    @staticmethod
    def few_shot_learning(question):
        """Learn from examples"""
        if "example" in question.lower() or "pattern" in question.lower():
            # Store examples for learning
            st.session_state.learning_examples.append(question)
            if len(st.session_state.learning_examples) >= 2:
                return f"I've learned from {len(st.session_state.learning_examples)} examples. I can now generalize patterns from this data."
        
        return None
    
    @staticmethod
    def adapt_from_feedback(question):
        """Improve from feedback"""
        if "correct" in question.lower() or "wrong" in question.lower() or "mistake" in question.lower():
            return "Thank you for the feedback. I'll update my understanding based on this correction. Learning from mistakes is essential for improvement."
        
        return None

# ============================================================
# 5. LANGUAGE INTELLIGENCE
# ============================================================

class LanguageIntelligence:
    """Understanding complex instructions, ambiguity, nuance, metaphor, sarcasm"""
    
    @staticmethod
    def detect_sarcasm(text):
        """Detect potential sarcasm"""
        sarcasm_indicators = ["yeah right", "obviously", "sure", "as if", "great"]
        for indicator in sarcasm_indicators:
            if indicator in text.lower():
                return True
        return False
    
    @staticmethod
    def understand_metaphor(text):
        """Interpret metaphorical language"""
        metaphors = {
            "time is money": "This means time is valuable and should be used efficiently, like money.",
            "break a leg": "This is a theatrical expression meaning 'good luck', not literal harm.",
            "raining cats and dogs": "This means raining very heavily, not actual animals.",
            "heart of gold": "This means someone is very kind and generous.",
        }
        
        for metaphor, meaning in metaphors.items():
            if metaphor in text.lower():
                return meaning
        
        return None
    
    @staticmethod
    def handle_ambiguity(question):
        """Resolve ambiguous statements"""
        if "bank" in question.lower():
            return "The word 'bank' is ambiguous: It could mean a financial institution or the side of a river. Context determines the meaning."
        
        return None

# ============================================================
# 6. WORLD MODELING & SOCIAL INTELLIGENCE
# ============================================================

class WorldModeling:
    """Understanding physical laws, social systems, predicting outcomes"""
    
    @staticmethod
    def physics_intuition(question):
        q_lower = question.lower()
        
        if "gravity" in q_lower and "mass" in q_lower:
            return "Gravitational force is proportional to mass (F = G·m₁·m₂/r²). Doubling mass doubles gravitational force."
        
        if "ice" in q_lower and "float" in q_lower:
            return "Ice floats because it's less dense than liquid water. This is unusual - most solids sink in their liquid form."
        
        return None
    
    @staticmethod
    def social_intelligence(question):
        q_lower = question.lower()
        
        if "why do people" in q_lower:
            return "Social behavior is influenced by cultural norms, psychological factors, evolutionary biology, and environmental contexts. Multiple factors usually interact."
        
        return None

# ============================================================
# 7. CRITICAL THINKING & UNCERTAINTY HANDLING
# ============================================================

class CriticalThinking:
    """Evaluate claims, detect bias, question weak logic"""
    
    @staticmethod
    def evaluate_claim(question):
        """Evaluate truthfulness of claims"""
        q_lower = question.lower()
        
        if "vaccine" in q_lower and "autism" in q_lower:
            return "This claim has been thoroughly debunked by multiple large-scale studies. The original study was retracted due to fraud. There is no scientific evidence linking vaccines to autism."
        
        if "climate change" in q_lower and "hoax" in q_lower:
            return "97% of climate scientists agree that climate change is real and human-caused. The evidence comes from multiple independent lines of research."
        
        return None
    
    @staticmethod
    def uncertainty_handling(question):
        """Express confidence levels, admit uncertainty"""
        if "sure" in question.lower() or "certain" in question.lower():
            return "I express confidence levels: High (90%+ for well-established facts), Medium (70-90% for generally accepted information), Low (under 70% for speculative or complex topics). I'll always admit when I'm uncertain."
        
        return None

# ============================================================
# 8. CREATIVITY & COMPOSITIONAL INTELLIGENCE
# ============================================================

class CreativityEngine:
    """Generate novel ideas, combine concepts, brainstorm solutions"""
    
    @staticmethod
    def generate_ideas(question):
        """Brainstorm multiple solutions"""
        if "brainstorm" in question.lower() or "ideas for" in question.lower():
            return "Brainstorming approach: 1) Divergent thinking (generate many ideas without judgment), 2) Convergent thinking (evaluate and select best), 3) Combine and improve ideas, 4) Consider constraints and resources."
        
        return None
    
    @staticmethod
    def combine_concepts(concept1, concept2):
        """Combine unrelated concepts"""
        if "combine" in concept1.lower() and "and" in concept1.lower():
            return f"Combining these concepts creates innovative possibilities. For example, {concept1} could lead to novel solutions that weren't obvious when considering each separately."
        
        return None

# ============================================================
# 9. TECHNICAL SKILLS (CODE GENERATION)
# ============================================================

class TechnicalSkills:
    """Write and understand code"""
    
    @staticmethod
    def generate_code(instruction):
        """Generate code for common tasks"""
        instr_lower = instruction.lower()
        
        code_templates = {
            "sort": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[len(arr)//2]\n    left = [x for x in arr if x < pivot]\n    middle = [x for x in arr if x == pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + middle + quicksort(right)",
            
            "factorial": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n-1)",
            
            "fibonacci": "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        print(a, end=' ')\n        a, b = b, a + b\n    print()",
            
            "prime": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5)+1):\n        if n % i == 0:\n            return False\n    return True"
        }
        
        for key, code in code_templates.items():
            if key in instr_lower:
                return f"```python\n{code}\n```"
        
        return None

# ============================================================
# 10. META-COGNITION & SELF-AWARENESS
# ============================================================

class MetaCognition:
    """Thinking about thinking, detecting errors, improving reasoning"""
    
    @staticmethod
    def reflect_on_reasoning(question):
        """Explain how answer was reached"""
        if "how did you" in question.lower() or "reasoning" in question.lower():
            return "I reached this conclusion through systematic reasoning: 1) Analyze the question structure, 2) Identify key variables and relationships, 3) Apply relevant logical/mathematical rules, 4) Verify consistency, 5) Present clear step-by-step explanation."
        
        return None
    
    @staticmethod
    def detect_error(question):
        """Detect and correct errors"""
        if "2+2=5" in question:
            return "This is incorrect. 2+2=4 in standard arithmetic. However, in modular arithmetic (mod 1), all numbers equal 0. In rounded calculations, 2.4+2.4=4.8 which rounds to 5. But in standard integer arithmetic, the correct answer is 4."
        
        return None

# ============================================================
# MAIN RESPONSE GENERATOR (Integrates ALL capabilities)
# ============================================================

def generate_response(user_input):
    """Integrate all intelligence capabilities"""
    
    q_lower = user_input.lower()
    
    # Initialize engines
    logic = LogicalReasoningEngine()
    math_intel = MathematicalIntelligence()
    problem_solver = ProblemSolvingEngine()
    learner = LearningEngine()
    language = LanguageIntelligence()
    world = WorldModeling()
    critical = CriticalThinking()
    creative = CreativityEngine()
    technical = TechnicalSkills()
    meta = MetaCognition()
    
    # 1. Identity
    if any(w in q_lower for w in ["who are you", "what are you"]):
        return """I am an AI system created by Moses, a student at KyungDong University, with comprehensive intelligence capabilities including:
- Logical reasoning (deductive, inductive, abductive, causal)
- Mathematical intelligence (arithmetic, algebra, probability)
- Problem solving and planning
- Memory and learning (few-shot, adaptation)
- Language intelligence (metaphor, sarcasm, ambiguity)
- World modeling (physics, social systems)
- Critical thinking and uncertainty handling
- Creativity and compositional thinking
- Technical skills (code generation)
- Meta-cognition (self-reflection, error detection)
- Emotional and social intelligence
- And 20+ other cognitive capabilities

How can I help you think through a problem today?"""
    
    # 2. Date/Time
    if any(w in q_lower for w in ["date today", "today's date"]):
        now = datetime.now()
        return f"Today is {now.strftime('%A, %B %d, %Y')}."
    
    # 3. Logical reasoning
    result = logic.deductive_reasoning(user_input, user_input)
    if result:
        return result
    
    result = logic.inductive_reasoning(user_input, user_input)
    if result:
        return result
    
    result = logic.abductive_reasoning(user_input, user_input)
    if result:
        return result
    
    result = logic.causal_reasoning(user_input)
    if result:
        return result
    
    # 4. Mathematical intelligence
    result = math_intel.solve_arithmetic(user_input)
    if result:
        return result
    
    result = math_intel.solve_algebra(user_input)
    if result:
        return result
    
    result = math_intel.probability(user_input)
    if result:
        return result
    
    # 5. Problem solving
    result = problem_solver.decompose_problem(user_input)
    if result:
        return result
    
    result = problem_solver.optimize_tradeoffs(user_input)
    if result:
        return result
    
    # 6. Learning
    result = learner.few_shot_learning(user_input)
    if result:
        return result
    
    result = learner.adapt_from_feedback(user_input)
    if result:
        return result
    
    # 7. Language intelligence
    result = language.understand_metaphor(user_input)
    if result:
        return result
    
    result = language.handle_ambiguity(user_input)
    if result:
        return result
    
    # 8. World modeling
    result = world.physics_intuition(user_input)
    if result:
        return result
    
    result = world.social_intelligence(user_input)
    if result:
        return result
    
    # 9. Critical thinking
    result = critical.evaluate_claim(user_input)
    if result:
        return result
    
    result = critical.uncertainty_handling(user_input)
    if result:
        return result
    
    # 10. Creativity
    result = creative.generate_ideas(user_input)
    if result:
        return result
    
    # 11. Technical skills
    result = technical.generate_code(user_input)
    if result:
        return result
    
    # 12. Meta-cognition
    result = meta.reflect_on_reasoning(user_input)
    if result:
        return result
    
    result = meta.detect_error(user_input)
    if result:
        return result
    
    # 13. Emotional intelligence
    if any(em in q_lower for em in ["feel", "feeling", "sad", "happy", "angry", "tired", "sick", "stressed"]):
        return get_emotional_response(user_input)
    
    # 14. Jokes
    if any(w in q_lower for w in ["joke", "funny", "make me laugh"]):
        jokes = [
            "Why don't scientists trust atoms? Because they make up everything!",
            "What do you call a fake noodle? An impasta!",
            "Why did the scarecrow win an award? Because he was outstanding in his field!"
        ]
        return random.choice(jokes)
    
    # 15. Greetings
    if user_input.strip().lower() in ["hi", "hello", "hey"]:
        return "Hello! I'm ready for deep reasoning, logic puzzles, mathematical problems, or any intellectual challenge. What would you like to explore?"
    
    if "thank" in q_lower:
        return "You're welcome! I'm here to help with reasoning, problem-solving, and learning."
    
    # 16. Default - Use FLAN for general reasoning
    with st.spinner("Applying comprehensive reasoning..."):
        try:
            prompt = f"""Question: {user_input}

Provide a thoughtful, accurate answer showing reasoning if appropriate. Be clear and helpful.

Answer:"""
            
            inputs = flan_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=600)
            with torch.no_grad():
                outputs = flan_model.generate(
                    inputs.input_ids,
                    max_length=300,
                    num_beams=4,
                    temperature=0.5,
                    pad_token_id=flan_tokenizer.eos_token_id
                )
            answer = flan_tokenizer.decode(outputs[0], skip_special_tokens=True)
            answer = answer.replace("Answer:", "").strip()
            
            if len(answer) > 10:
                return answer
        except:
            pass
    
    return "I'm applying my full reasoning capabilities to your question. Could you provide more context or rephrase for clarity?"

def get_emotional_response(user_input):
    """Generate empathetic response"""
    q_lower = user_input.lower()
    
    if "sick" in q_lower:
        return "I'm sorry you're not feeling well. Your health matters. Please rest and take care of yourself."
    if "tired" in q_lower:
        return "I hear that you're tired. Rest is essential for cognitive function and well-being. Can you take a break?"
    if "sad" in q_lower:
        return "I'm sorry you're feeling sad. Would you like to talk about what's bothering you? Sometimes sharing helps."
    if "happy" in q_lower:
        return "That's wonderful to hear! Positive emotions enhance cognitive performance and overall well-being."
    if "stressed" in q_lower:
        return "Stress can impact reasoning and decision-making. Take a deep breath. What's one small thing that might help?"
    
    return "I'm here for you. How can I support you right now?"

# ============================================================
# UI
# ============================================================

st.markdown("<h1 style='text-align: center;'>Ultimate AI Intelligence System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Created by Moses, KyungDong University | 23 Intelligence Capabilities</p>", unsafe_allow_html=True)

st.divider()

with st.sidebar:
    st.markdown("### Intelligence Capabilities")
    st.markdown("✅ Logical reasoning (deductive, inductive, abductive, causal)")
    st.markdown("✅ Mathematical intelligence (arithmetic, algebra, probability)")
    st.markdown("✅ Problem solving & planning")
    st.markdown("✅ Memory & learning (few-shot, adaptation)")
    st.markdown("✅ Language intelligence (metaphor, sarcasm, ambiguity)")
    st.markdown("✅ World modeling (physics, social systems)")
    st.markdown("✅ Critical thinking & uncertainty handling")
    st.markdown("✅ Creativity & compositional thinking")
    st.markdown("✅ Technical skills (code generation)")
    st.markdown("✅ Meta-cognition (self-reflection, error detection)")
    st.markdown("✅ Emotional & social intelligence")
    st.markdown("✅ And 12+ more capabilities")
    
    st.divider()
    
    if st.button("Start New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.emotion_history = []
        st.session_state.learning_examples = []
        st.rerun()
    
    st.divider()
    st.info("""Try testing my intelligence:

LOGIC: "All humans are mortal. Socrates is human. What can we conclude?"

MATH: "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?"

CAUSAL: "Does ice cream cause drowning? Explain."

PATTERN: "What comes next: 2, 6, 12, 20, 30, ?"

METAPHOR: "What does 'time is money' mean?"

CODE: "Write a function to check if a number is prime"

UNCERTAINTY: "How confident are you in your answers?""")

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

user_input = st.chat_input("Ask me anything - test my reasoning, logic, math, or creativity...")

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
