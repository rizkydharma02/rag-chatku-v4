import os
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import tempfile
import logging
from requests.exceptions import RequestException

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for API key
GROQ_API_URL = "https://api.groq.com/v1/chat/completions"

# Configure requests session
session = requests.Session()
session.trust_env = False  # Disable environment-based proxy settings

@st.cache_resource
def load_embedding_model(model_name):
    """Load and cache the embedding model"""
    return SentenceTransformer(model_name)

def set_api_key(api_key):
    """Store API key in session state"""
    if api_key:
        st.session_state.api_key = api_key
        return True
    return False

def get_available_models():
    """Get list of available models"""
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama3-8b-8192"]

def read_pdf(file_path):
    """Read text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

def read_docx(file_path):
    """Read text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"Error reading DOCX: {str(e)}")
        raise

def read_file(file_path):
    """Read text from PDF or DOCX file"""
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

@st.cache_data
def read_url(url):
    """Fetch and read text from URL"""
    try:
        response = session.get(url, verify=True)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        logger.error(f"Error reading URL: {str(e)}")
        raise

@st.cache_data(show_spinner=False)
def generate_embedding(text, model):
    """Generate embedding for text using provided model"""
    try:
        return model.encode(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def query_llm(prompt, model_name):
    """Query LLM using Groq API with direct HTTP request"""
    if not st.session_state.get("api_key"):
        return "Error: API key not found. Please configure your API key first."

    try:
        # Setup request headers
        headers = {
            "Authorization": f"Bearer {st.session_state.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
        }
        
        # Setup request payload
        data = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.7,
            "max_tokens": 2048
        }
        
        # Make request using session
        response = session.post(
            GROQ_API_URL,
            headers=headers,
            json=data,
            timeout=60,
            verify=True
        )
        
        # Check for HTTP errors
        response.raise_for_status()
        
        # Parse response
        result = response.json()
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            if "content" in message:
                return message["content"]
                
        return "Error: Unexpected response format from API"
            
    except requests.exceptions.Timeout:
        logger.error("Request timed out")
        return "Error: Request timed out. Please try again."
    except requests.exceptions.HTTPError as e:
        logger.error(f"HTTP Error in query_llm: {str(e)}")
        return f"Error: API request failed (HTTP {response.status_code}). Please check your API key."
    except requests.exceptions.RequestException as e:
        logger.error(f"Request Error in query_llm: {str(e)}")
        return "Error: Failed to connect to API. Please check your internet connection."
    except Exception as e:
        logger.error(f"Error in query_llm: {str(e)}")
        return f"Error: Failed to get response. {str(e)}"

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise

def create_index(embeddings):
    """Create FAISS index from embeddings"""
    try:
        if not embeddings:
            return None
        
        embeddings_array = np.array(embeddings)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array.astype('float32'))
        return index
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise

def search_index(index, query_embedding, k=3):
    """Search FAISS index for similar embeddings"""
    try:
        if index is None:
            return []
        
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        return I[0]
    except Exception as e:
        logger.error(f"Error searching index: {str(e)}")
        raise

def validate_api_key_format(api_key: str) -> bool:
    """Validate GROQ API key format"""
    if not api_key:
        return False
    
    api_key = api_key.strip()
    return api_key.startswith('gsk_') and len(api_key) >= 20