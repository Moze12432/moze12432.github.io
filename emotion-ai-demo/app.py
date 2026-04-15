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
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter
from chromadb import Client as ChromaClient
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi

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

TEMPERATURE = 0
MAX_TOKENS = 800

st.set_page_config(page_title="Mukiibi Moses AI", page_icon="🧠", layout="wide")

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

# ============================================
# ADVANCED RAG PIPELINE WITH CHUNKING & RERANKING
# ============================================

class AdvancedRAG:
    def __init__(self):
        # Initialize chunking strategy
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", ".", " ", ""],
            length_function=len
        )
        
        # Initialize ChromaDB for vector storage
        self.chroma_client = ChromaClient()
        self.collection = self.chroma_client.create_collection(
            name="mozeai_memory",
            embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2"
            )
        )
        
        # For BM25 (keyword search)
        self.bm25_corpus = []
        self.bm25_index = None
        self.documents = []
        
    def chunk_document(self, text, metadata=None):
        """Split document into intelligent chunks"""
        chunks = self.text_splitter.split_text(text)
        chunk_metadata = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata.append({
                "chunk_id": i,
                "total_chunks": len(chunks),
                "source": metadata.get("source", "unknown") if metadata else "unknown",
                "timestamp": time.time(),
                "chunk_hash": hashlib.md5(chunk.encode()).hexdigest()
            })
        
        return chunks, chunk_metadata
    
    def add_memory(self, text, metadata=None):
        """Add text to memory with chunking and multiple indices"""
        if len(text) < 50:
            return
            
        # Chunk the document
        chunks, chunk_metadata = self.chunk_document(text, metadata)
        
        # Add to vector database
        ids = []
        embeddings = []
        metadatas = []
        
        for i, (chunk, meta) in enumerate(zip(chunks, chunk_metadata)):
            chunk_id = f"{int(time.time())}_{i}_{hashlib.md5(chunk.encode()).hexdigest()[:8]}"
            ids.append(chunk_id)
            embeddings.append(embedder.encode(chunk))
            metadatas.append({
                "text": chunk,
                "chunk_id": i,
                **meta
            })
        
        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadatas
            )
        except:
            pass
        
        # Add to BM25 corpus for keyword search
        self.bm25_corpus.extend(chunks)
        self.bm25_index = BM25Okapi(self.bm25_corpus)
        
        # Store full documents
        self.documents.append({
            "text": text,
            "metadata": metadata,
            "chunks": chunks,
            "timestamp": time.time()
        })
        
        # Keep only last 50 documents to manage memory
        if len(self.documents) > 50:
            self.documents = self.documents[-50:]
    
    def retrieve_with_reranking(self, query, top_k=5):
        """Retrieve and rerank results using multiple strategies"""
        if not self.documents:
            return []
        
        query_embedding = embedder.encode(query)
        
        # Strategy 1: Vector similarity (semantic search)
        vector_texts = []
        try:
            vector_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k * 2
            )
            
            if vector_results['metadatas']:
                for meta_list in vector_results['metadatas']:
                    for meta in meta_list:
                        if meta and 'text' in meta:
                            vector_texts.append(meta['text'])
        except:
            pass
        
        # Strategy 2: BM25 keyword search
        bm25_texts = []
        if self.bm25_index and self.bm25_corpus:
            try:
                bm25_scores = self.bm25_index.get_scores(query.split())
                top_bm25_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:top_k]
                bm25_texts = [self.bm25_corpus[i] for i in top_bm25_indices if i < len(self.bm25_corpus)]
            except:
                pass
        
        # Combine results (deduplicate)
        all_texts = []
        seen = set()
        
        for text in vector_texts:
            if text not in seen:
                seen.add(text)
                all_texts.append(text)
        
        for text in bm25_texts:
            if text not in seen:
                seen.add(text)
                all_texts.append(text)
        
        # Reranking with hybrid scoring
        reranked_results = []
        query_terms = set(query.lower().split())
        
        for text in all_texts[:top_k * 2]:
            # Simple relevance scoring based on query term matching
            text_terms = set(text.lower().split())
            overlap = len(query_terms & text_terms)
            if overlap > 0:
                score = min(1.0, overlap / max(len(query_terms), 1))
            else:
                score = 0.1
            reranked_results.append((text, score))
        
        # Sort by score
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        return [text for text, score in reranked_results[:top_k]]
    
    def get_context(self, query, max_chunks=3):
        """Get relevant context for query"""
        if not self.documents:
            return ""
        
        retrieved_chunks = self.retrieve_with_reranking(query, top_k=max_chunks)
        
        if retrieved_chunks:
            context = "RELEVANT CONTEXT FROM MEMORY:\n\n"
            for i, chunk in enumerate(retrieved_chunks):
                context += f"[{i+1}] {chunk}\n\n"
            return context
        return ""

# Initialize embedder and RAG
embedder = SentenceTransformer("all-MiniLM-L6-v2")
advanced_rag = AdvancedRAG()

# ============================================
# MEMORY FUNCTIONS (USING ADVANCED RAG)
# ============================================

def store_memory(text):
    """Store memory using advanced RAG pipeline"""
    if len(text) > 50:
        try:
            advanced_rag.add_memory(text, metadata={"type": "conversation"})
        except Exception as e:
            pass

def retrieve_memory(query):
    """Retrieve memory using advanced RAG with reranking"""
    if not advanced_rag.documents:
        return ""
    try:
        return advanced_rag.get_context(query, max_chunks=3)
    except Exception as e:
        return ""

# ============================================
# LLM FUNCTION WITH MULTIPLE MODEL FALLBACKS (70B PRIORITY)
# ============================================

def llm_with_fallback(messages, max_retries=2):
    """Call LLM with automatic fallback to multiple models - 70B prioritized"""
    
    models_to_try = [
        "llama-3.3-70b-versatile",
        "llama-3.1-70b-versatile",
        "mixtral-8x7b-32768",
        "llama-3.1-8b-instant",
        "gemma2-9b-it"
    ]
    
    last_error = None
    
    for model in models_to_try:
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS,
                    messages=messages,
                    timeout=30
                )
                st.session_state.last_model_used = model
                return completion.choices[0].message.content.strip()
                
            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                continue
    
    return f"AI service temporarily unavailable. Last error: {last_error[:100] if last_error else 'Unknown'}"

def llm(messages):
    """Wrapper for backward compatibility"""
    return llm_with_fallback(messages)

# ============================================
# ENHANCED SYSTEM PROMPT
# ============================================

SYSTEM_PROMPT = """
You are MozeAI, an advanced AI assistant with REAL-TIME internet access and file analysis capabilities.

CREATOR INFORMATION (ONLY mention when asked directly):
- Created by Mukiibi Moses, a Computer Engineering student at Kyungdong University, South Korea.

YOUR CAPABILITIES:
- REAL-TIME web search for current information
- File analysis for PDF, DOCX, TXT, CSV, JSON files
- File comparison (compare multiple documents)
- Memory of past conversations (advanced RAG retrieval)
- Image generation and editing
- Calculator and news

CRITICAL RULES:
1. For questions about PEOPLE, PLACES, EVENTS, or ANY topic not related to your creator, USE SEARCH RESULTS
2. ONLY mention your creator when specifically asked
3. Use the conversation history and retrieved memory for context
4. Answer concisely and accurately
5. Be conversational and friendly
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
if "last_image_url" not in st.session_state:
    st.session_state.last_image_url = None
if "generated_images" not in st.session_state:
    st.session_state.generated_images = []
if "current_image_index" not in st.session_state:
    st.session_state.current_image_index = -1
if "code_search_cache" not in st.session_state:
    st.session_state.code_search_cache = {}
if "is_resetting" not in st.session_state:
    st.session_state.is_resetting = False
if "last_model_used" not in st.session_state:
    st.session_state.last_model_used = None

# ============================================
# SEARCH FUNCTIONS
# ============================================

def internet_search(query):
    try:
        clean_query = query.strip()
        url = "https://html.duckduckgo.com/html/"
        params = {"q": clean_query}
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.post(url, data=params, headers=headers, timeout=10)
        
        if response.status_code == 200:
            results = re.findall(r'<a rel="nofollow" class="result__a" href="[^"]*">([^<]+)</a>', response.text)
            snippets = re.findall(r'<a class="result__snippet"[^>]*>([^<]+(?:<[^>]+>[^<]*</[^>]+>)*)</a>', response.text)
            
            if results:
                context = f"SEARCH RESULTS for '{clean_query}':\n\n"
                for i in range(min(5, len(results))):
                    context += f"**{results[i]}**\n"
                    if i < len(snippets):
                        snippet = re.sub(r'<[^>]+>', '', snippets[i])
                        snippet = snippet.replace('&#39;', "'").replace('&quot;', '"').replace('&amp;', '&')
                        context += f"{snippet[:400]}\n\n"
                return context[:3000]
        
        return wikipedia_search(clean_query)
    except Exception as e:
        return wikipedia_search(query)

def wikipedia_search(query):
    try:
        url = "https://en.wikipedia.org/api/rest_v1/page/summary/"
        q = query.strip().replace(" ", "_")
        response = requests.get(url + q, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            title = data.get("title", "")
            extract = data.get("extract", "")
            if extract:
                return f"Wikipedia - {title}:\n{extract[:2000]}"
        
        search_url = "https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": 3
        }
        response = requests.get(search_url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            
            if search_results:
                context = f"Wikipedia search results for '{query}':\n\n"
                for result in search_results[:2]:
                    title = result.get("title", "")
                    r2 = requests.get(url + title.replace(" ", "_"), timeout=10)
                    if r2.status_code == 200:
                        data2 = r2.json()
                        extract = data2.get("extract", "")[:800]
                        if extract:
                            context += f"**{title}**\n{extract}\n\n"
                return context[:2500]
    except:
        pass
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
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            for element in soup(["script", "style", "nav", "footer", "header"]):
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
# ROUTER FUNCTION
# ============================================

def route(query):
    q = query.lower()
    
    if extract_urls_from_query(query):
        return "scrape_url"
    
    file_keywords = [
        "document", "file", "upload", "pdf", "docx", "txt", "csv", "json",
        "what is this", "what does this", "tell me about this", "about this file",
        "the file", "the document", "my file", "my document", "uploaded file",
        "what's in", "what is in", "summarize", "analyze this", "this document",
        "this file", "read the file", "read this", "file content", "document content",
        "what is the file", "what does the file", "file says", "document says"
    ]
    if any(x in q for x in file_keywords):
        return "file_task"
    
    comparison_keywords = ["compare", "comparison", "difference between", "similarities", "versus", "vs", "diff"]
    if any(x in q for x in comparison_keywords):
        return "compare_files"
    
    if any(phrase in q for phrase in ["can you", "do you", "are you able", "how to"]):
        return "reason"
    
    if any(x in q for x in ["generate image", "create image", "draw", "make an image of", "picture of", "image of"]):
        return "generate_image"
    
    if any(x in q for x in ["edit image", "change the image", "modify image", "redraw", "make it", "make the", "add to the image", "remove from image", "brighter", "darker", "different", "make the cat", "turn it", "change it to"]):
        return "edit_image"
    
    people_patterns = ["who is", "tell me about", "what do you know about", "information about"]
    if any(x in q for x in people_patterns):
        return "search"
    
    if any(x in q for x in ["weather", "temperature", "temp", "rain", "snow", "forecast"]):
        return "search"
    
    if any(x in q for x in ["+", "-", "*", "/", "×", "calculate", "=", "math"]):
        return "calculator"
    
    if any(x in q for x in ["time", "date", "today", "current time", "what day"]):
        return "datetime"
    
    if any(x in q for x in ["news", "headlines", "current events", "breaking news"]):
        return "search"
    
    coding_keywords = ["code", "python", "javascript", "html", "css", "react", "tkinter", "function", "class", "import", "algorithm", "debug", "fix", "write a program", "create a script"]
    if any(x in q for x in coding_keywords):
        return "coding_with_search"
    
    if len(q) > 10 and not any(x in q for x in ["how are you", "what is your", "who created"]):
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
    """Generate response with full conversation context and enhanced RAG"""
    
    # Get enhanced memory context
    memory_context = retrieve_memory(question)
    
    # Combine all contexts
    enhanced_context = context
    if memory_context:
        enhanced_context += "\n" + memory_context
    
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

{enhanced_context[:3000]}

USER QUESTION: {question}

Instructions:
- Use the conversation history for context
- Use the retrieved memory for relevant past information
- Answer naturally and conversationally
- If information isn't available, say so

ANSWER:
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
    """Analyze uploaded files with chunking for better understanding"""
    
    # Chunk the file content for better processing
    chunks = advanced_rag.text_splitter.split_text(file_context)
    
    # Find most relevant chunks for the query
    query_embedding = embedder.encode(query)
    chunk_embeddings = [embedder.encode(chunk) for chunk in chunks[:10]]
    
    # Score chunks by relevance
    scored_chunks = []
    for i, chunk in enumerate(chunks[:10]):
        chunk_embedding = embedder.encode(chunk)
        score = np.dot(query_embedding, chunk_embedding) / (np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding))
        scored_chunks.append((chunk, score))
    
    scored_chunks.sort(key=lambda x: x[1], reverse=True)
    relevant_chunks = [chunk for chunk, score in scored_chunks[:3]]
    
    analysis_prompt = f"""
Files: {filenames}

MOST RELEVANT SECTIONS FROM THE FILES:
{chr(10).join(relevant_chunks)}

USER QUESTION: {query}

Answer based on the file content above. Be specific and quote from the relevant sections.
"""
    
    messages = [
        {"role": "system", "content": "You are a file analysis assistant. Answer based on the provided file sections."},
        {"role": "user", "content": analysis_prompt}
    ]
    return clean_answer(llm(messages))

def evaluate_work(question, file_context):
    prompt = f"Content: {file_context[:3000]}\n\nRequest: {question}\n\nProvide assessment."
    messages = [{"role": "system", "content": "You evaluate work."}, {"role": "user", "content": prompt}]
    return clean_answer(llm(messages))

# ============================================
# IMAGE GENERATION FUNCTIONS
# ============================================

def generate_image(prompt):
    try:
        encoded_prompt = requests.utils.quote(prompt)
        timestamp = int(time.time())
        image_url = f"https://image.pollinations.ai/prompt/{encoded_prompt}?width=1024&height=1024&nologo=true&_={timestamp}"
        return image_url
    except Exception as e:
        return None

def generate_and_display_image(prompt):
    image_url = generate_image(prompt)
    
    if image_url:
        if hasattr(st.session_state, 'last_image_prompt') and st.session_state.last_image_prompt and st.session_state.last_image_prompt != prompt:
            return f"🎨 **Edited Image - New Prompt:** '{prompt}'\n\n![Generated Image]({image_url})\n\n*Image generated by AI*"
        else:
            return f"🎨 **Generated Image for:** '{prompt}'\n\n![Generated Image]({image_url})\n\n*Image generated by AI*"
    else:
        return "❌ Sorry, I couldn't generate an image right now."

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
# RUN AGENT FUNCTION
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
        st.session_state.last_image_url = None
        return "✅ Context cleared! How can I help you today?"
    
    if any(phrase in q for phrase in ["who are you", "who is this", "what are you", "tell me about yourself"]):
        return "I am MozeAI, an AI assistant created by Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea. I can search the web, analyze files, compare documents, generate images, and answer questions. How can I help you today?"
    
    if any(phrase in q for phrase in ["mukiibi moses", "who is moses", "your maker", "your creator", "tell me about your maker", "tell me about your creator", "who created you"]):
        return """**Mukiibi Moses** is my creator and a talented Computer Engineering student at **Kyungdong University in South Korea**.

**About Him:**
- Specializes in artificial intelligence, machine learning, and data science
- His portfolio: https://moze12432.github.io/

He built me with web search, file analysis, image generation, and coding assistance capabilities."""
    
    if q in ["is your maker a genius", "is your creator a genius"]:
        return "Yes! Mukiibi Moses is a brilliant Computer Engineering student at Kyungdong University."
    
    if q in ["tell me about your maker", "tell me about your creator"]:
        return "My maker is Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea."
    
    if q in ["who is your maker", "who created you"]:
        return "I was created by Mukiibi Moses, a Computer Engineering student at Kyungdong University in South Korea."
    
    if q in ["can you generate images", "do you generate images", "can you create images", "can you draw"]:
        return "Yes, I can generate images! Just tell me what you want, for example: 'generate image of a cat' or 'draw a beautiful sunset'"
    
    edit_indicators = ["make it", "make the", "turn it", "change it to", "change the", "add a", "add to", "remove", "make the cat", "make the image", "edit the", "modify the"]
    if any(phrase in q for phrase in edit_indicators) and st.session_state.get("last_image_prompt"):
        last_prompt = st.session_state.get("last_image_prompt", "")
        if last_prompt:
            edit_text = query
            for word in ["edit image", "change the image", "modify image", "redraw", "make it", "make the", "change the", "edit the", "turn it", "change it to"]:
                if word in edit_text.lower():
                    edit_text = re.sub(re.escape(word), "", edit_text.lower(), flags=re.IGNORECASE).strip()
                    break
            edit_text = ' '.join(edit_text.split())
            new_prompt = f"{last_prompt}, {edit_text}"
            st.session_state.last_image_prompt = new_prompt
            with st.spinner("🎨 Editing image..."):
                return generate_and_display_image(new_prompt)
    
    tool = route(query)
    context = ""
    
    follow_up_phrases = ["tell me more", "more about", "continue", "go on", "elaborate", "explain further", "his background", "about him", "about her", "about them"]
    is_follow_up = any(phrase in q for phrase in follow_up_phrases)
    continuation_phrases = ["yes", "yeah", "sure", "ok", "continue", "tell me more", "go on", "and?", "then?"]
    
    if is_follow_up or q in continuation_phrases:
        if st.session_state.last_search_results:
            context = st.session_state.last_search_results
            if "background" in q:
                context += "\n\n Provide BACKGROUND details from the search results."
            elif "music" in q or "career" in q:
                context += "\n\n Provide CAREER details from the search results."
            else:
                context += "\n\n Provide MORE information from the search results above."
        elif st.session_state.last_response:
            context = f"Previous response: {st.session_state.last_response[:500]}\n\nContinue naturally."
        else:
            context = get_current_datetime()
    
    if tool == "compare_files" and st.session_state.file_context and len(st.session_state.uploaded_files) >= 2:
        filenames = "\n".join([f"- {name}" for name in st.session_state.uploaded_files.keys()])
        with st.spinner("📊 Comparing files..."):
            response = compare_files(query, st.session_state.file_context, filenames)
            st.session_state.last_response = response
            return response
    
    elif tool == "file_task" and st.session_state.file_context:
        filenames = "\n".join([f"- {name}" for name in st.session_state.uploaded_files.keys()])
        with st.spinner(f"📖 Reading {len(st.session_state.uploaded_files)} file(s)..."):
            response = analyze_uploaded_files(query, st.session_state.file_context, filenames)
            st.session_state.last_response = response
            return response
    
    elif tool == "evaluate" and st.session_state.file_context:
        with st.spinner("📝 Evaluating your work..."):
            response = evaluate_work(query, st.session_state.file_context)
            st.session_state.last_response = response
            return response
    
    elif tool == "scrape_url":
        urls = extract_urls_from_query(query)
        scraped = ""
        for url in urls:
            with st.spinner(f"Reading {url}..."):
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
                "generate image", "generate an image", "generate a image",
                "create image of", "create an image of", "create a image of",
                "create image", "create an image", "create a image",
                "draw a", "draw an", "draw",
                "make an image of", "make a image of", "make image of",
                "picture of", "image of"
            ]
            for phrase in command_phrases:
                if phrase in image_prompt.lower():
                    image_prompt = re.sub(re.escape(phrase), "", image_prompt.lower(), flags=re.IGNORECASE).strip()
                    break
            
            image_prompt = ' '.join(image_prompt.split())
            if not image_prompt or len(image_prompt) < 3:
                image_prompt = query
            
            st.session_state.last_image_prompt = image_prompt
            st.session_state.last_image_url = None
            
            return generate_and_display_image(image_prompt)
    
    elif tool == "edit_image":
        with st.spinner("🎨 Editing image..."):
            last_prompt = st.session_state.get("last_image_prompt", "")
            
            if not last_prompt:
                return "❌ No previous image found. Please generate an image first using 'generate image of...'"
            
            edit_instruction = query
            command_words = ["edit image", "change the image", "modify image", "redraw", "make it", "make the", "change the", "edit the", "turn it", "change it to"]
            for word in command_words:
                if word in edit_instruction.lower():
                    edit_instruction = re.sub(re.escape(word), "", edit_instruction.lower(), flags=re.IGNORECASE).strip()
                    break
            
            edit_instruction = ' '.join(edit_instruction.split())
            new_prompt = f"{last_prompt}, {edit_instruction}"
            st.session_state.last_image_prompt = new_prompt
            st.session_state.last_image_url = None
            
            return generate_and_display_image(new_prompt)
    
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
    
    if not is_follow_up and q not in continuation_phrases:
        store_memory(answer)
    
    return answer

# ============================================
# UI - MAIN DISPLAY
# ============================================

st.markdown('<h1 style="text-align: center;">🧠 Mukiibi-Moses AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #667eea;">Intelligent Autonomous Agent with Real-Time Internet Access</p>', unsafe_allow_html=True)
st.markdown("---")

# ============================================
# SIDEBAR
# ============================================

with st.sidebar:
    st.markdown("### 🧠 MozeAI")
    st.markdown("---")
    
    if st.button("🔄 New Chat", key="new_chat_btn", use_container_width=True):
        if not st.session_state.get("is_resetting", False):
            st.session_state.is_resetting = True
            
            st.session_state.chat_history = []
            st.session_state.memory_store = []
            st.session_state.uploaded_files = {}
            st.session_state.file_context = ""
            st.session_state.last_image_prompt = None
            st.session_state.generated_images = []
            st.session_state.current_image_index = -1
            st.session_state.last_search_query = None
            st.session_state.last_search_results = None
            st.session_state.last_response = None
            st.session_state.last_topic = None
            st.session_state.code_search_cache = {}
            
            # Also clear the advanced RAG memory
            global advanced_rag
            advanced_rag = AdvancedRAG()
            
            st.session_state.is_resetting = False
            st.success("✨ New chat started!")
            st.rerun()
    
    if st.button("🗑️ Clear Files", key="clear_files_btn", use_container_width=True):
        st.session_state.uploaded_files = {}
        st.session_state.file_context = ""
        st.success("✅ All files cleared!")
        st.rerun()
    
    st.markdown("---")
    
    st.markdown("### 📤 Upload Files")
    st.markdown("Supported: PDF, DOCX, TXT, CSV, JSON")
    
    uploaded_files = st.file_uploader(
        "Choose files to analyze",
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
                        st.error(f"❌ {file.name}: {content}")
        
        if st.session_state.uploaded_files:
            file_context_parts = []
            for name, content in st.session_state.uploaded_files.items():
                file_context_parts.append(f"\n{'='*60}\n📄 FILE: {name}\n{'='*60}\n{content}\n")
            st.session_state.file_context = "\n".join(file_context_parts)
            st.info(f"📁 {len(st.session_state.uploaded_files)} file(s) loaded: {', '.join(st.session_state.uploaded_files.keys())}")
    
    if st.session_state.uploaded_files:
        st.markdown("---")
        st.markdown(f"**📄 Loaded Files ({len(st.session_state.uploaded_files)})**")
        for name, content in st.session_state.uploaded_files.items():
            with st.expander(f"📄 {name} (click to preview)"):
                preview = content[:500] + "..." if len(content) > 500 else content
                st.text(preview)
        
        if len(st.session_state.uploaded_files) >= 2:
            st.markdown("---")
            st.markdown("### 🔍 Comparison Tips")
            st.markdown("**Try asking:**")
            st.markdown("- \"Compare all the files\"")
            st.markdown("- \"What are the differences between these documents?\"")
            st.markdown("- \"What is in each file?\"")
        
        st.markdown("---")
        st.markdown("**💡 Try asking:**")
        st.markdown("- \"What is this file about?\"")
        st.markdown("- \"Summarize the key points\"")
        st.markdown("- \"Evaluate my work\"")
    
    with st.expander("🔧 Debug: File Context Preview", expanded=False):
        if st.session_state.file_context:
            st.text(f"Total context length: {len(st.session_state.file_context)} characters")
            st.text("First 1000 characters:")
            st.code(st.session_state.file_context[:1000])
        else:
            st.write("No file context loaded")
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("**Creator:** Mukiibi Moses")
    st.markdown("**Student:** Computer Engineering")
    st.markdown("**University:** Kyungdong University, South Korea")
    st.markdown("**Focus:** AI Agents, LLM Applications, Real-time Systems")
    st.markdown("---")
    st.markdown("### ✨ Features")
    st.markdown("✅ **Real-time web search** (weather, news, facts)")
    st.markdown("✅ **File analysis** (PDF, DOCX, CSV, JSON, TXT)")
    st.markdown("✅ **File comparison** (compare multiple documents)")
    st.markdown("✅ **Image generation & editing**")
    st.markdown("✅ **Advanced RAG memory** (chunking + reranking)")
    st.markdown("✅ **Current news headlines**")
    st.markdown("✅ **Calculator**")
    st.markdown("✅ **Conversation memory**")
    st.markdown("✅ **Work evaluation**")
    if st.session_state.last_model_used:
        st.caption(f"🤖 Model: {st.session_state.last_model_used}")

# ============================================
# CHAT DISPLAY
# ============================================

for role, msg in st.session_state.chat_history:
    with st.chat_message(role):
        st.write(msg)

query = st.chat_input("Ask me anything... I can check weather, compare files, search the web, and more!")

if query:
    st.session_state.chat_history.append(("user", query))
    with st.chat_message("user"):
        st.write(query)
    
    response = run_agent(query)
    
    with st.chat_message("assistant"):
        st.write(response)
    
    st.session_state.chat_history.append(("assistant", response))
    st.rerun()
