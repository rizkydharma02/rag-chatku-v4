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

@st.cache_resource
def load_embedding_model(model_name):
    """Load and cache the embedding model"""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        raise

def set_api_key(api_key):
    """Set and validate GROQ API key"""
    global groq_client
    try:
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
        
        # Test connection
        try:
            completion = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": "Test connection"}],
                max_tokens=5
            )
            if not completion or not completion.choices:
                raise ValueError("Gagal memverifikasi API key")
        except Exception as e:
            logger.error(f"API key verification failed: {str(e)}")
            raise ValueError(f"Verifikasi API key gagal: {str(e)}")
            
        logger.info("API key successfully validated and set")
        return True
        
    except Exception as e:
        logger.error(f"Error setting API key: {str(e)}")
        groq_client = None
        raise

def get_available_models():
    """Get list of available models"""
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama3-8b-8192"]

def read_pdf(file_path):
    """Read and extract text from PDF"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join(page.extract_text() for page in pdf_reader.pages)
            if not text.strip():
                raise ValueError("Extracted text is empty")
            return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

def read_docx(file_path):
    """Read and extract text from DOCX"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        if not text.strip():
            raise ValueError("Extracted text is empty")
        return text
    except Exception as e:
        logger.error(f"Error reading DOCX: {str(e)}")
        raise

@st.cache_data
def read_url(url):
    """Read and extract text from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        if not text.strip():
            raise ValueError("Extracted text is empty")
        return text
    except Exception as e:
        logger.error(f"Error reading URL: {str(e)}")
        raise

def read_file(file_path):
    """Read file based on extension"""
    _, file_extension = os.path.splitext(file_path)
    ext = file_extension.lower()
    
    if ext == '.pdf':
        return read_pdf(file_path)
    elif ext == '.docx':
        return read_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

def generate_embedding(text, model):
    """Generate embeddings for text"""
    try:
        if not text or not text.strip():
            raise ValueError("Input text is empty")
        return model.encode(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def query_llm(prompt, model_name):
    """Query LLM with improved error handling"""
    try:
        if groq_client is None:
            raise ValueError("GROQ client belum diinisialisasi. Silakan cek API key.")
            
        if not prompt or not prompt.strip():
            raise ValueError("Prompt tidak boleh kosong")
            
        logger.info(f"Querying LLM with model: {model_name}")
        logger.info(f"Prompt length: {len(prompt)} characters")
        
        # Create chat completion
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
        
        # Validate response
        if not completion or not completion.choices or not completion.choices[0].message:
            raise ValueError("Invalid response from GROQ API")
            
        response = completion.choices[0].message.content
        if not response or not isinstance(response, str):
            raise ValueError("Invalid response content")
            
        logger.info(f"Received valid response of length: {len(response)}")
        return response
        
    except Exception as e:
        error_msg = f"Error in query_llm: {str(e)}"
        logger.error(error_msg)
        return f"An error occurred: {str(e)}"

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def create_index(embeddings):
    """Create FAISS index for embeddings"""
    try:
        if not embeddings:
            logger.warning("No embeddings provided for index creation")
            return None
            
        embeddings_array = np.array(embeddings)
        dimension = embeddings_array.shape[1]
        
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array.astype('float32'))
        
        logger.info(f"Successfully created index with {len(embeddings)} embeddings")
        return index
        
    except Exception as e:
        logger.error(f"Error creating index: {str(e)}")
        raise

def search_index(index, query_embedding, k=3):
    """Search FAISS index"""
    try:
        if index is None:
            logger.warning("No index provided for search")
            return []
            
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        return I[0]
        
    except Exception as e:
        logger.error(f"Error searching index: {str(e)}")
        raise

def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format"""
    try:
        if not api_key or not api_key.strip():
            return False
            
        api_key = api_key.strip()
        if not api_key.startswith('gsk_'):
            return False
            
        if len(api_key) < 20:
            return False
            
        return True
        
    except Exception as e:
        logger.error(f"Error validating API key format: {str(e)}")
        return False