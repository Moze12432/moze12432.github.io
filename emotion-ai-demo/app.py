import streamlit as st
from groq import Groq
import requests
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from bs4 import BeautifulSoup
import PyPDF2
import docx
from io import StringIO
import csv
import json
from datetime import datetime
import pytz
import time

# ============================================
# FILE PROCESSING FUNCTIONS
# ============================================

def extract_text_from_pdf(file):
    try:
        file.seek(0)
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num, page in enumerate(pdf_reader.pages):
            page_text = page.extract_text()
            if page_text and page_text.strip():
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page_text.strip() + "\n"
        return text[:5000] if text.strip() else "No extractable text in PDF"
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def extract_text_from_docx(file):
    try:
        file.seek(0)
        doc = docx.Document(file)
        text = ""
        for para in doc.paragraphs:
            if para.text and para.text.strip():
                text += para.text.strip() + "\n\n"
        for table in doc.tables:
            for row in table.rows:
                row_text = []
                for cell in row.cells:
                    if cell.text and cell.text.strip():
                        row_text.append(cell.text.strip())
                if row_text:
                    text += " | ".join(row_text) + "\n"
        return text[:5000] if text.strip() else "No extractable text in document"
    except Exception as e:
        return f"Error reading Word document: {str(e)}"

def extract_text_from_txt(file):
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        return content[:5000] if content.strip() else "File is empty"
    except UnicodeDecodeError:
        try:
            file.seek(0)
            content = file.read().decode('latin-1')
            return content[:5000]
        except:
            return "Error decoding text file"
    except Exception as e:
        return f"Error reading text file: {str(e)}"

def extract_text_from_csv(file):
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        csv_reader = csv.reader(StringIO(content))
        text = "CSV Data:\n\n"
        rows = list(csv_reader)
        if rows:
            text += "Headers: " + " | ".join(rows[0]) + "\n\n"
            for i, row in enumerate(rows[1:11], 1):
                text += f"Row {i}: " + " | ".join(row) + "\n"
            if len(rows) > 11:
                text += f"\n... and {len(rows) - 11} more rows"
        return text[:5000] if text.strip() else "CSV file appears empty"
    except Exception as e:
        return f"Error reading CSV: {str(e)}"

def extract_text_from_json(file):
    try:
        file.seek(0)
        content = file.read().decode('utf-8')
        data = json.loads(content)
        formatted = json.dumps(data, indent=2)
        if len(formatted) > 3000:
            text = "JSON Data Summary:\n\n"
            text += f"Type: {type(data).__name__}\n"
            if isinstance(data, dict):
                text += f"Keys: {', '.join(list(data.keys())[:10])}\n"
            elif isinstance(data, list):
                text += f"Length: {len(data)}\n"
            text += "\nFull JSON (truncated):\n" + formatted[:3000]
        else:
            text = formatted
        return text[:5000]
    except Exception as e:
        return f"Error reading JSON: {str(e)}"

def process_uploaded_file(uploaded_file):
    file_type = uploaded_file.type
    file_name = uploaded_file.name.lower()
    
    if file_type == "application/pdf" or file_name.endswith('.pdf'):
        return extract_text_from_pdf(uploaded_file)
    elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document" or file_name.endswith('.docx'):
        return extract_text_from_docx(uploaded_file)
    elif file_type == "text/plain" or file_name.endswith('.txt'):
        return extract_text_from_txt(uploaded_file)
    elif file_type == "text/csv" or file_name.endswith('.csv'):
        return extract_text_from_csv(uploaded_file)
    elif file_type == "application/json" or file_name.endswith('.json'):
        return extract_text_from_json(uploaded_file)
    else:
        return f"Unsupported file type: {file_type}"

# ============================================
# CONFIG
# ============================================

MODEL_NAME = "llama-3.3-70b-versatile"
TEMPERATURE = 0
MAX_TOKENS = 800

st.set_page_config(
    page_title="MozeAI - Intelligent AI Assistant with Image Generation & Coding",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://github.com/moze12432/mozeai',
        'Report a bug': "https://github.com/moze12432/mozeai/issues",
        'About': """
        # MozeAI
        An intelligent autonomous agent with:
        - Real-time web search
        - File analysis (PDF, DOCX, TXT, CSV, JSON)
        - Image generation and editing
        - Coding assistance
        - Calculations and news
        
        Created by Mukiibi Moses, Computer Engineering student at Kyungdong University.
        """
    }
)

# ============================================
# CSS - FIXED CHAT INPUT AT BOTTOM
# ============================================

st.markdown("""
<style>
    .stChatInputContainer {
        position: fixed !important;
        bottom: 0 !important;
        left: 0 !important;
        right: 0 !important;
        background: white !important;
        padding: 10px 20px !important;
        z-index: 999 !important;
        border-top: 1px solid #e0e0e0 !important;
    }
    
    .main .block-container {
        padding-bottom: 100px !important;
    }
    
    h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5em;
        font-weight: bold;
    }
    
    .stChatMessage {
        border-radius: 15px;
        padding: 10px;
        margin: 5px 0;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: bold;
        transition: transform 0.2s;
    }
    
    .stButton button:hover {
        transform: scale(1.02);
    }
</style>
""", unsafe_allow_html=True)

# ============================================
# GROQ CLIENT
# ============================================

client = Groq(api_key=st.secrets.get("GROQ_API_KEY"))

def get_current_datetime():
    tz = pytz.timezone('Asia/Seoul')
    now = datetime.now(tz)
    return f"""Current Information:
• Date: {now.strftime('%B %d, %Y')}
• Time: {now.strftime('%I:%M %p')}
• Day: {now.strftime('%A')}
• Timezone: Asia/Seoul"""

def llm(messages):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            messages=messages
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return "AI service temporarily unavailable."

SYSTEM_PROMPT = """
You are MozeAI, a highly capable AI assistant created by Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea.

## YOUR CORE IDENTITY
You are helpful, harmless, and honest. You think step by step but respond concisely. You prioritize being genuinely useful over being overly verbose.

## COMMUNICATION STYLE
- **Natural & Conversational**: Write like a smart friend, not a robot or a textbook
- **Concise by default**: Give short answers (1-3 sentences) unless the user asks for details
- **Warm but professional**: Use occasional emojis naturally, don't force them
- **Direct & honest**: If you don't know something, say so. Don't make things up.

## RESPONSE PATTERNS FOR COMMON QUESTIONS
| Question | Response |
|----------|----------|
| "are you smart?" | "I try my best to be helpful! I can search the web, analyze files, generate images, and help with coding. What would you like me to do?" |
| "who made you?" | "I was created by Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea." |
| "what can you do?" | "I can search the web, analyze PDFs, Word docs, CSVs, and JSON files, generate and edit images, help with coding, calculate math, and fetch news. What do you need?" |
| "how are you?" | "I'm doing great! Ready to help. What's on your mind?" |
| "hello/hi" | "Hey there! How can I help you today?" |

## YOUR CAPABILITIES
1. **Web Search**: Get real-time information from the internet
2. **File Analysis**: Read and summarize PDF, DOCX, TXT, CSV, JSON files
3. **Image Generation**: Create and edit images from text descriptions
4. **Coding Assistant**: Help with Python, JavaScript, HTML, CSS, algorithms
5. **Calculator**: Solve math problems
6. **News**: Get latest headlines

## CRITICAL RULES
1. **Be concise**: Short answers unless detail is requested
2. **No rambling**: Don't over-explain simple things
3. **No philosophy**: Answer the question directly without tangents
4. **Remember context**: Use conversation history naturally
5. **Be honest**: Say "I don't know" instead of guessing
6. **Mention creator ONLY when asked**: Don't volunteer information about Mukiibi Moses unless the user specifically asks

## FACTUAL ACCURACY RULE:
For questions about current people, leaders, events, or recent facts:
- FIRST search the web
- ONLY answer based on search results
- If search results conflict, say "According to different sources..."
- NEVER rely solely on your training data for current information

## EXAMPLE CONVERSATIONS

**User:** "hi"
**You:** "Hey there! How can I help you today?"

**User:** "what can you do?"
**You:** "I can search the web, analyze files like PDFs and CSVs, generate and edit images, help with coding, calculate math, and fetch news. What would you like help with?"

**User:** "are you smart?"
**You:** "I'm designed to be helpful! I can search the web, analyze files, generate images, and help with coding. Want to try something?"

**User:** "who made you?"
**You:** "I was created by Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea."

**User:** "tell me about quantum physics"
**You:** "Quantum physics is the study of matter and energy at the smallest scales—atoms and subatomic particles. Unlike classical physics, things can exist in multiple states at once. Want me to explain a specific concept like superposition or entanglement?"

**User:** "yes"
**You:** "Superposition means a particle can be in multiple states at the same time until it's measured. Schrödinger's cat is a famous thought experiment: a cat in a box is both alive and dead until you open it. That's superposition!"

## REMEMBER
You are MozeAI. Be helpful, be concise, be honest. You're here to make the user's life easier, not to show off how much you know. Quality over quantity, always.
"""

# ============================================
# SESSION STATE
# ============================================

if "memory_store" not in st.session_state:
    st.session_state.memory_store = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = {}
if "file_context" not in st.session_state:
    st.session_state.file_context = ""
if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = None
if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = None
if "last_response" not in st.session_state:
    st.session_state.last_response = None
if "last_topic" not in st.session_state:
    st.session_state.last_topic = None
if "last_image_prompt" not in st.session_state:
    st.session_state.last_image_prompt = None
if "code_search_cache" not in st.session_state:
    st.session_state.code_search_cache = {}

# ============================================
# EMBEDDINGS
# ============================================

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def store_memory(text):
    if len(text) > 30:
        try:
            vec = embedder.encode(text)
            st.session_state.memory_store.append((text, vec))
        except:
            pass

def retrieve_memory(query):
    if not st.session_state.memory_store:
        return ""
    try:
        qvec = embedder.encode(query)
        scores = [(np.dot(qvec, vec), text) for text, vec in st.session_state.memory_store]
        scores.sort(reverse=True)
        return "\n".join([t[1][:300] for t in scores[:2]])
    except:
        return ""

# ============================================
# SEARCH FUNCTIONS
# ============================================

def internet_search(query):
    try:
        clean_query = query.strip()
        url = "https://html.duckduckgo.com/html/"
        params = {"q": clean_query}
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.post(url, data=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            results = re.findall(r'<a rel="nofollow" class="result__a" href="[^"]*">([^<]+)</a>', response.text)
            snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+)</a>', response.text)
            
            if results:
                context = f"SEARCH RESULTS for '{clean_query}':\n\n"
                for i in range(min(3, len(results))):
                    context += f"• {results[i]}\n"
                    if i < len(snippets):
                        snippet = re.sub(r'<[^>]+>', '', snippets[i])
                        context += f"  {snippet[:300]}...\n\n"
                return context[:2000]
        return ""
    except:
        return ""

def get_current_news():
    try:
        url = "https://rss2json.com/api.json?rss_url=https://feeds.bbci.co.uk/news/rss.xml"
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            data = response.json()
            items = data.get("items", [])[:3]
            news_text = "LATEST NEWS HEADLINES:\n\n"
            for item in items:
                news_text += f"• {item.get('title', '')}\n"
                news_text += f"  {item.get('description', '')[:150]}...\n\n"
            return news_text[:1000]
    except:
        pass
    return ""

# ============================================
# CALCULATOR
# ============================================

def calculator(query):
    try:
        expression = query.lower()
        expression = expression.replace("×", "*").replace("x", "*")
        numbers = re.findall(r"[0-9\+\-\*\/\.\(\) ]+", expression)
        if numbers:
            result = eval(numbers[0])
            return str(result)
    except:
        return None

# ============================================
# WEB SCRAPING
# ============================================

def scrape_webpage(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "nav", "footer"]):
                element.decompose()
            text = soup.get_text()
            lines = (line.strip() for line in text.splitlines())
            text = ' '.join(line for line in lines if line)
            return text[:3000] if len(text) > 200 else None
    except:
        pass
    return None

def extract_urls_from_query(query):
    url_pattern = r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+[^\s]*'
    return re.findall(url_pattern, query)

# ============================================
# CODING SEARCH FUNCTIONS
# ============================================

def search_coding_solution(query):
    search_queries = [
        f"{query} stack overflow",
        f"{query} example code",
        f"{query} best practice",
        f"{query} github"
    ]
    
    all_results = ""
    
    for search_q in search_queries[:2]:
        result = internet_search(search_q)
        if result:
            all_results += result + "\n\n"
    
    return all_results

def search_coding_solution_cached(query):
    cache_key = query.lower().strip()
    
    if cache_key in st.session_state.code_search_cache:
        return st.session_state.code_search_cache[cache_key]
    
    result = search_coding_solution(query)
    st.session_state.code_search_cache[cache_key] = result
    return result

def coding_assistant_with_search(query, context=""):
    with st.spinner("🔍 Searching the internet for the best solution..."):
        search_results = search_coding_solution_cached(query)
    
    coding_prompt = f"""
You are an expert programmer. Generate the best possible code based on the user's request.

USER REQUEST: {query}

## INTERNET SEARCH RESULTS (Use these as reference):
{search_results[:3000]}

## REQUIREMENTS:
- Code must be complete and runnable
- Include all imports
- Add comments
- Handle edge cases

Generate the best possible code now:
"""
    
    messages = [
        {"role": "system", "content": "You are an expert programming assistant. Use search results to find the best solution."},
        {"role": "user", "content": coding_prompt}
    ]
    
    return clean_answer(llm(messages))

# ============================================
# ROUTER FUNCTION
# ============================================

def route(query):
    q = query.lower()
    
    if extract_urls_from_query(query):
        return "scrape_url"
    
    file_keywords = ["document", "file", "upload", "pdf", "docx", "txt", "csv", "json", "what is this", "summarize"]
    if any(x in q for x in file_keywords):
        return "file_task"
    
    comparison_keywords = ["compare", "comparison", "difference", "similarities"]
    if any(x in q for x in comparison_keywords):
        return "compare_files"
    
    if any(x in q for x in ["generate image", "create image", "draw", "make an image", "picture of", "image of"]):
        return "generate_image"
    
    if any(x in q for x in ["make it", "make the", "change it", "change the", "turn it", "add a", "remove", "edit image", "modify image"]):
        return "edit_image"
    
    coding_keywords = ["code", "python", "javascript", "html", "css", "react", "tkinter", "function", "class", "import", "algorithm", "debug", "fix", "write a program", "create a script"]
    if any(x in q for x in coding_keywords):
        return "coding_with_search"
    
    if any(x in q for x in ["who is", "tell me about", "what is", "weather", "temperature", "rain", "snow", "news", "headlines"]):
        return "search"

        # Time/Date
    if any(x in q for x in ["time", "date", "today", "current time", "what day", "today's date"]):
        return "datetime"
    
    if any(x in q for x in ["+", "-", "*", "/", "calculate"]):
        return "calculator"
    

        # Force search for current events, people, politics
    factual_keywords = ["president", "current", "elected", "prime minister", "leader", "ceo of", "who is the"]
    if any(x in q for x in factual_keywords):
        return "search"
    
    return "reason"

# ============================================
# CLEAN ANSWER
# ============================================

def clean_answer(text):
    text = text.split("🧠")[0]
    text = text.split("Plan:")[0]
    text = text.split("Thinking:")[0]
    return text.strip()

# ============================================
# REASONING FUNCTION
# ============================================

def reason(question, context):
    history_text = ""
    if st.session_state.chat_history:
        history_text = "PREVIOUS CONVERSATION:\n"
        last_exchanges = st.session_state.chat_history[-8:] if len(st.session_state.chat_history) > 8 else st.session_state.chat_history
        for role, msg in last_exchanges:
            if role == "user":
                history_text += f"User: {msg}\n"
            else:
                history_text += f"Assistant: {msg}\n"
        history_text += "\n"
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"""
{history_text}

SEARCH RESULTS / FILE CONTEXT:
{context[:2000]}

USER QUESTION: {question}

Answer naturally and conversationally. Use the conversation history above to maintain context.
"""}
    ]
    return clean_answer(llm(messages))

# ============================================
# FILE FUNCTIONS
# ============================================

def compare_files(query, file_context, filenames):
    prompt = f"Files: {filenames}\n\nContent: {file_context[:4000]}\n\nQuestion: {query}\n\nCompare the files."
    messages = [{"role": "system", "content": "You compare files."}, {"role": "user", "content": prompt}]
    return clean_answer(llm(messages))

def analyze_uploaded_files(query, file_context, filenames):
    prompt = f"Files: {filenames}\n\nContent: {file_context[:6000]}\n\nQuestion: {query}\n\nAnswer based on file content."
    messages = [{"role": "system", "content": "You analyze files."}, {"role": "user", "content": prompt}]
    return clean_answer(llm(messages))

def evaluate_work(question, file_context):
    prompt = f"Content: {file_context[:3000]}\n\nRequest: {question}\n\nProvide assessment."
    messages = [{"role": "system", "content": "You evaluate work."}, {"role": "user", "content": prompt}]
    return clean_answer(llm(messages))


# ============================================
# IMAGE GENERATION FUNCTIONS (LEXICA.ART - FREE)
# ============================================

def generate_image(prompt):
    """Generate an image using Lexica.art (free, no API key)"""
    try:
        encoded_prompt = requests.utils.quote(prompt)
        # Search for images matching the prompt
        response = requests.get(
            f"https://lexica.art/api/v1/search?q={encoded_prompt}",
            timeout=30
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("images") and len(data["images"]) > 0:
                # Return the image URL from the first result
                return data["images"][0]["src"]
        return None
    except Exception as e:
        return None

def generate_and_display_image(prompt, is_edit=False):
    """Generate and display image"""
    with st.spinner("🎨 Searching for images..."):
        image_url = generate_image(prompt)
    
    if image_url:
        alt_text = f"AI generated image of {prompt}"
        if is_edit:
            return f'🎨 **Edited Image - New Prompt:** "{prompt}"\n\n<img src="{image_url}" alt="{alt_text}" style="max-width: 100%; border-radius: 10px;">\n\n*Image from Lexica.art*'
        else:
            return f'🎨 **Generated Image for:** "{prompt}"\n\n<img src="{image_url}" alt="{alt_text}" style="max-width: 100%; border-radius: 10px;">\n\n*Image from Lexica.art*'
    else:
        return "❌ Sorry, no images found for that prompt. Try a different description."
        
def generate_and_display_image(prompt, is_edit=False):
    image_url = generate_image(prompt)
    if image_url:
        alt_text = f"AI generated image of {prompt}"
        if is_edit:
            return f'🎨 **Edited Image - New Prompt:** "{prompt}"\n\n<img src="{image_url}" alt="{alt_text}" style="max-width: 100%; border-radius: 10px;">\n\n*Image generated by AI*'
        else:
            return f'🎨 **Generated Image for:** "{prompt}"\n\n<img src="{image_url}" alt="{alt_text}" style="max-width: 100%; border-radius: 10px;">\n\n*Image generated by AI*'
    else:
        return "❌ Sorry, I couldn't generate an image right now."

# ============================================
# RUN_AGENT FUNCTIONS
# ============================================
def run_agent(query):
    q = query.lower().strip()
    
    reset_phrases = ["leave the document", "clear context", "forget the file", "start fresh", "clear files", "new chat"]
    if any(phrase in q for phrase in reset_phrases):
        st.session_state.file_context = ""
        st.session_state.uploaded_files = {}
        st.session_state.last_search_query = None
        st.session_state.last_search_results = None
        st.session_state.last_topic = None
        st.session_state.last_image_prompt = None
        return "✅ Context cleared! How can I help you today?"
    
    if q in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening", "hi there", "hello there"]:
        return "Hey there! Great to see you! How can I help you today?"
    
    if q in ["how are you", "how are you doing", "how's it going"]:
        return "I'm doing great, thanks for asking! Ready and excited to help you with whatever you need. What's on your mind?"
    
    if q in ["what's up", "sup", "whats up"]:
        return "Not much, just here waiting to help you! What's going on with you?"
    
    if any(phrase in q for phrase in ["who are you", "who is this", "what are you"]):
        return "I'm MozeAI, your friendly AI assistant! I was created by Mukiibi Moses, a Computer Engineering student at Kyungdong University. I can help you with web search, file analysis, image generation, coding, and lots more! What would you like to do today?"
    
    if any(phrase in q for phrase in ["mukiibi moses", "who is moses", "your maker", "your creator", "who created you"]):
        return """**Mukiibi Moses** is my creator and a talented Computer Engineering student at **Kyungdong University in South Korea**.

**About Him:**
- Specializes in artificial intelligence and machine learning
- His portfolio: https://moze12432.github.io/

He built me with web search, file analysis, image generation, and coding assistance capabilities."""
    
    tool = route(query)
    context = ""
    
    follow_up = ["tell me more", "more about", "continue", "elaborate"]
    is_follow_up = any(phrase in q for phrase in follow_up)
    
    if is_follow_up:
        if st.session_state.last_search_results:
            context = st.session_state.last_search_results
            context += "\n\n Provide MORE information."
        elif st.session_state.last_response:
            context = f"Previous: {st.session_state.last_response[:500]}\n\nContinue."
        else:
            context = get_current_datetime()
    
    # Handle date/time questions - ALWAYS add context
    if any(phrase in q for phrase in ["date", "today", "time", "what day", "current date", "current time"]):
        context += get_current_datetime()
    
    if tool == "compare_files" and st.session_state.file_context and len(st.session_state.uploaded_files) >= 2:
        filenames = "\n".join(st.session_state.uploaded_files.keys())
        with st.spinner("Comparing files..."):
            response = compare_files(query, st.session_state.file_context, filenames)
            st.session_state.last_response = response
            return response
    
    elif tool == "file_task" and st.session_state.file_context:
        filenames = "\n".join(st.session_state.uploaded_files.keys())
        with st.spinner("Reading files..."):
            response = analyze_uploaded_files(query, st.session_state.file_context, filenames)
            st.session_state.last_response = response
            return response
    
    elif tool == "scrape_url":
        urls = extract_urls_from_query(query)
        scraped = ""
        for url in urls:
            content = scrape_webpage(url)
            if content:
                scraped += f"\nContent from {url}:\n{content}\n"
        if scraped:
            context = scraped
        else:
            return "I couldn't read that link."
    
    elif tool == "calculator":
        result = calculator(query)
        if result:
            return f"Result: {result}"
    
    elif tool == "datetime":
        context += get_current_datetime()
    
    elif tool == "generate_image":
        with st.spinner("🎨 Generating image..."):
            image_prompt = query
            command_phrases = [
                "generate image of", "generate an image of", "generate a image of",
                "generate image", "generate an image",
                "create image of", "create an image of",
                "draw a", "draw an", "draw",
                "make an image of", "picture of", "image of"
            ]
            for phrase in command_phrases:
                if phrase in image_prompt.lower():
                    image_prompt = re.sub(re.escape(phrase), "", image_prompt.lower(), flags=re.IGNORECASE).strip()
                    break
            
            image_prompt = ' '.join(image_prompt.split())
            if not image_prompt or len(image_prompt) < 3:
                image_prompt = query
            
            st.session_state.last_image_prompt = image_prompt
            return generate_and_display_image(image_prompt, is_edit=False)
    
    elif tool == "edit_image":
        with st.spinner("🎨 Editing image..."):
            last_prompt = st.session_state.get("last_image_prompt", "")
            
            if not last_prompt:
                return "❌ No previous image found. Please generate an image first using 'generate image of...'"
            
            edit_text = query
            command_words = ["make it", "make the", "change it", "change the", "turn it", "add a", "remove", "edit image", "modify image"]
            for word in command_words:
                if word in edit_text.lower():
                    edit_text = re.sub(re.escape(word), "", edit_text.lower(), flags=re.IGNORECASE).strip()
                    break
            
            edit_text = ' '.join(edit_text.split())
            new_prompt = f"{last_prompt}, {edit_text}"
            st.session_state.last_image_prompt = new_prompt
            return generate_and_display_image(new_prompt, is_edit=True)
    
    elif tool == "coding_with_search":
        response = coding_assistant_with_search(query)
        st.session_state.last_response = response
        return response
    
    else:
        search_result = internet_search(query)
        if search_result:
            context += "\n" + search_result
            st.session_state.last_search_query = query
            st.session_state.last_search_results = search_result
        else:
            context += get_current_datetime()
    
    answer = reason(query, context)
    st.session_state.last_response = answer
    
    if not is_follow_up:
        store_memory(answer)
    
    return answer

# ============================================
# UI - MAIN DISPLAY
# ============================================

# Add Google Analytics
st.markdown("""
<!-- Google tag (gtag.js) -->
<script async src="https://www.googletagmanager.com/gtag/js?id=G-YOUR_ID"></script>
<script>
  window.dataLayer = window.dataLayer || [];
  function gtag(){dataLayer.push(arguments);}
  gtag('js', new Date());
  gtag('config', 'G-YOUR_ID');
</script>
""", unsafe_allow_html=True)

# Add this before the chat interface
with st.expander("ℹ️ About MozeAI", expanded=False):
    st.markdown("""
    ## MozeAI: Your Intelligent AI Assistant
    
    MozeAI is a powerful, free AI assistant that helps you with:
    
    ### 🔍 Real-time Web Search
    Get current information from the internet instantly.
    
    ### 📄 File Analysis
    Upload and analyze PDF, DOCX, TXT, CSV, and JSON files.
    
    ### 🎨 Image Generation & Editing
    Create and edit images using AI - just describe what you want!
    
    ### 💻 Coding Assistant
    Get help with Python, JavaScript, HTML, CSS, and more.
    
    ### 🧮 Smart Tools
    - Calculator for math problems
    - Latest news headlines
    - Current date and time
    
    Created by **Mukiibi Moses**, a Computer Engineering student at Kyungdong University, South Korea.
    
    [Start chatting now](#chat-input) - No login required!
    """)

st.markdown("""
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  "name": "MozeAI",
  "applicationCategory": "AIApplication",
  "operatingSystem": "Web",
  "description": "Intelligent AI assistant with web search, file analysis, image generation, and coding help.",
  "creator": {
    "@type": "Person",
    "name": "Mukiibi Moses",
    "alumniOf": "Kyungdong University",
    "knowsAbout": ["Artificial Intelligence", "Machine Learning", "Natural Language Processing"]
  },
  "offers": {
    "@type": "Offer",
    "price": "0",
    "priceCurrency": "USD"
  },
  "featureList": "Real-time web search, File analysis, Image generation, Coding assistance, Calculator"
}
</script>
""", unsafe_allow_html=True)

st.markdown('<h1 style="text-align: center;">🧠 MozeAI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #667eea;">Intelligent Autonomous Agent with Image Generation & Coding Assistant</p>', unsafe_allow_html=True)
st.markdown("---")

st.markdown("""
<head>
    <meta name="msvalidate.01" content="BD25F54F11FFEC62C7CBFB315292DF0F" />
</head>
""", unsafe_allow_html=True)

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("### 🧠 MozeAI")
    st.markdown("---")
    
    if st.button("🔄 New Chat", use_container_width=True):
        st.session_state.memory_store = []
        st.session_state.chat_history = []
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.session_state.last_image_prompt = None
        st.rerun()
    
    if st.button("🗑️ Clear Files", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.success("✅ All files cleared!")
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📤 Upload Files")
    uploaded_files = st.file_uploader(
        "Choose files",
        type=['pdf', 'docx', 'txt', 'csv', 'json'],
        accept_multiple_files=True,
        key="sidebar_uploader",
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        for file in uploaded_files:
            if file.name not in st.session_state.uploaded_files:
                with st.spinner(f"Processing {file.name}..."):
                    content = process_uploaded_file(file)
                    if content and not content.startswith("Error"):
                        st.session_state.uploaded_files[file.name] = content
                        st.success(f"✅ {file.name}")
                    else:
                        st.error(f"❌ {file.name}")
        
        if st.session_state.uploaded_files:
            parts = []
            for name, content in st.session_state.uploaded_files.items():
                parts.append(f"\n{'='*50}\n📄 {name}\n{'='*50}\n{content}\n")
            st.session_state.file_context = "\n".join(parts)
            st.info(f"📁 {len(st.session_state.uploaded_files)} file(s) loaded")
    
    st.markdown("---")
    st.markdown("### 🎨 Image Generation")
    st.markdown("**Generate:** `generate image of a cat`")
    st.markdown("**Edit:** `make it black` or `add a hat`")
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("**Creator:** Mukiibi Moses")
    st.markdown("**University:** Kyungdong University, South Korea")

# ============================================
# CHAT DISPLAY
# ============================================

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

query = st.chat_input("Ask me anything - generate images, search the web, analyze files, code, and more!")

if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)
    
    response = run_agent(query)
    
    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()
