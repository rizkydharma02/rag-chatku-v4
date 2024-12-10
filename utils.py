# utils.py
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

@st.cache_resource
def load_embedding_model(model_name):
    """Load and cache the embedding model"""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        raise

def set_api_key(api_key: str) -> bool:
    """Set GROQ API key and initialize client"""
    try:
        if api_key and api_key.strip():
            api_key = api_key.strip()
            if not api_key.startswith('gsk_'):
                logger.error("Invalid API key format: must start with 'gsk_'")
                return False
                
            # Set environment variable
            os.environ["GROQ_API_KEY"] = api_key
            return True
        return False
    except Exception as e:
        logger.error(f"Error setting API key: {str(e)}")
        return False

def get_available_models() -> list:
    """Get list of available LLM models"""
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama2-70b-chat"]

def query_llm(prompt: str, model_name: str) -> str:
    """Query GROQ LLM with proper error handling"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Error: API key not found. Please set your GROQ API key."

        # Initialize Groq client without proxies
        try:
            client = Groq(api_key=api_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=2048,
                top_p=1,
                stream=False
            )
            
            if completion.choices and completion.choices[0].message:
                return completion.choices[0].message.content
            else:
                return "Error: No response from model"
                
        except Exception as e:
            if "got an unexpected keyword argument 'proxies'" in str(e):
                # Jika error terkait proxies, coba instalasi ulang groq
                os.system("pip install --upgrade groq")
                return "Please try again. System is updating..."
            else:
                return f"Error: {str(e)}"

    except Exception as e:
        logger.error(f"Error in query_llm: {str(e)}")
        return f"Error: {str(e)}"

# [Bagian kode lainnya tetap sama]

def read_pdf(file_path: str) -> str:
    """Read content from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

def read_docx(file_path: str) -> str:
    """Read content from DOCX file"""
    try:
        doc = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        logger.error(f"Error reading DOCX: {str(e)}")
        raise

@st.cache_data
def read_url(url: str) -> str:
    """Read content from URL with caching"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
        return soup.get_text()
    except Exception as e:
        logger.error(f"Error reading URL: {str(e)}")
        raise

def read_file(file_path: str) -> str:
    """Read content from file based on extension"""
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return read_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def generate_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    """Generate embeddings for text"""
    try:
        return model.encode(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def create_index(embeddings: List[np.ndarray]) -> Optional[faiss.IndexFlatL2]:
    """Create FAISS index for embeddings"""
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

def search_index(index: faiss.IndexFlatL2, query_embedding: np.ndarray, k: int = 3) -> List[int]:
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
    try:
        if not api_key:
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