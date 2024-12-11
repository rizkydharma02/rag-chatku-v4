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
from typing import List, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_groq_client():
    """Get or create Groq client using session state"""
    if 'groq_client' not in st.session_state:
        st.session_state.groq_client = None
    return st.session_state.groq_client

@st.cache_resource
def load_embedding_model(model_name: str) -> SentenceTransformer:
    """Load and cache the embedding model"""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        raise

def set_api_key(api_key: str) -> None:
    """Set the GROQ API key and initialize client"""
    if api_key:
        try:
            # Create new client without proxies
            client = Groq(api_key=api_key)
            st.session_state.groq_client = client
            logger.info("GROQ client initialized successfully")
        except Exception as e:
            logger.error(f"Error setting API key: {str(e)}")
            st.session_state.groq_client = None
    else:
        st.session_state.groq_client = None

def get_available_models() -> List[str]:
    """Get list of available LLM models"""
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama2-70b-chat"]

@st.cache_data(show_spinner=False)
def read_pdf(file_path: str) -> str:
    """Read and extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

@st.cache_data(show_spinner=False)
def read_docx(file_path: str) -> str:
    """Read and extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        logger.error(f"Error reading DOCX: {str(e)}")
        raise

@st.cache_data(show_spinner=False)
def read_url(url: str) -> str:
    """Fetch and extract text from URL"""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text(separator="\n", strip=True)
    except Exception as e:
        logger.error(f"Error reading URL: {str(e)}")
        raise

def read_file(file_path: str) -> str:
    """Read file based on its extension"""
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.pdf':
            return read_pdf(file_path)
        elif file_extension.lower() == '.docx':
            return read_docx(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise

@st.cache_data(show_spinner=False)
def generate_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    """Generate embedding for text using specified model"""
    try:
        return model.encode(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def query_llm(prompt: str, model_name: str) -> str:
    """Query the LLM with given prompt"""
    try:
        client = get_groq_client()
        if client is None:
            raise ValueError("API key not set. Please configure your API key first.")
        
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
            top_p=1
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}")
        return f"An error occurred: {str(e)}"

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

def create_index(embeddings: List[np.ndarray]) -> Optional[faiss.Index]:
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

def search_index(index: faiss.Index, query_embedding: np.ndarray, k: int = 3) -> List[int]:
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