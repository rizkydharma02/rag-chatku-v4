import os
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from groq import Groq
import faiss
import numpy as np
import streamlit as st
import tempfile
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global client
groq_client = None

def handle_error(func):
    """Decorator for error handling"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise type(e)(f"Error in {func.__name__}: {str(e)}")
    return wrapper

@st.cache_resource
def load_embedding_model(model_name):
    """Load and cache the embedding model"""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        raise

@handle_error
def set_api_key(api_key):
    """Set and validate GROQ API key"""
    global groq_client
    
    if not api_key:
        raise ValueError("API key tidak boleh kosong")
    
    # Clean and validate API key format
    api_key = api_key.strip()
    if not api_key.startswith('gsk_'):
        raise ValueError("API key harus dimulai dengan 'gsk_'")
    if len(api_key) < 20:
        raise ValueError("API key tidak valid")
        
    # Initialize GROQ client
    groq_client = Groq(api_key=api_key)
    
    # Test connection with retry
    max_retries = 3
    for attempt in range(max_retries):
        try:
            test_response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=5
            )
            if test_response and test_response.choices:
                logger.info("API key successfully validated")
                return True
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Final API key verification failed: {str(e)}")
                raise ValueError(f"Verifikasi API key gagal setelah {max_retries} percobaan")
            logger.warning(f"API verification attempt {attempt + 1} failed, retrying...")
            time.sleep(1)  # Short delay before retry

@handle_error
def get_available_models():
    """Get list of available models"""
    return ["mixtral-8x7b-32768", "gemma-7b-it"]

@handle_error
def query_llm(prompt, model_name):
    """Query LLM with improved error handling and retry logic"""
    if groq_client is None:
        raise ValueError("GROQ client belum diinisialisasi. Silakan cek API key.")
        
    if not prompt or not prompt.strip():
        raise ValueError("Prompt tidak boleh kosong")
        
    logger.info(f"Querying LLM with model: {model_name}")
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=1000,
                stream=False
            )
            
            if not completion or not completion.choices:
                raise ValueError("Invalid response from API")
                
            response = completion.choices[0].message.content
            if not response or not isinstance(response, str):
                raise ValueError("Invalid response content")
                
            logger.info(f"Successfully generated response (length: {len(response)})")
            return response
            
        except Exception as e:
            if attempt == max_retries - 1:
                logger.error(f"Final LLM query attempt failed: {str(e)}")
                return f"Error generating response: {str(e)}"
            logger.warning(f"LLM query attempt {attempt + 1} failed, retrying...")
            time.sleep(1)

@handle_error
def generate_embedding(text, model):
    """Generate embeddings for text"""
    if not text or not text.strip():
        raise ValueError("Input text is empty")
    return model.encode(text)

@handle_error
def read_file(file_path):
    """Read file based on extension"""
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    
    if ext == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join(page.extract_text() for page in pdf_reader.pages)
    elif ext == '.docx':
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
        
    if not text.strip():
        raise ValueError("Extracted text is empty")
    return text

@handle_error
def read_url(url):
    """Read and extract text from URL"""
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    text = soup.get_text()
    
    if not text.strip():
        raise ValueError("Extracted text is empty")
    return text

@handle_error
def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

@handle_error
def create_index(embeddings):
    """Create FAISS index for embeddings"""
    if not embeddings:
        raise ValueError("No embeddings provided for index creation")
        
    embeddings_array = np.array(embeddings)
    dimension = embeddings_array.shape[1]
    
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array.astype('float32'))
    
    logger.info(f"Created index with {len(embeddings)} embeddings")
    return index

@handle_error
def search_index(index, query_embedding, k=3):
    """Search FAISS index"""
    if index is None:
        raise ValueError("No index provided for search")
        
    D, I = index.search(np.array([query_embedding]).astype('float32'), k)
    return I[0]

@handle_error
def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format"""
    if not api_key or not api_key.strip():
        return False
        
    api_key = api_key.strip()
    return api_key.startswith('gsk_') and len(api_key) >= 20