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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variable for Groq client
groq_client = None

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

def initialize_groq_client(api_key):
    """Initialize Groq client with API key"""
    global groq_client
    if api_key:
        try:
            # Create new client instance
            groq_client = Groq(api_key=api_key)
            # Test the client with a simple query
            response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": "test"}],
                max_tokens=1
            )
            logger.info("Groq client initialized successfully")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize Groq client: {str(e)}")
            groq_client = None
            return False
    return False

def set_api_key(api_key):
    """Set API key and initialize client"""
    if not api_key:
        logger.warning("No API key provided")
        return False
        
    try:
        st.session_state.api_key = api_key
        success = initialize_groq_client(api_key)
        if success:
            logger.info("API key set successfully")
            return True
        else:
            logger.warning("Failed to initialize client with provided API key")
            return False
    except Exception as e:
        logger.error(f"Error setting API key: {str(e)}")
        return False

def get_available_models():
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama3-8b-8192"]

def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")

@st.cache_data
def read_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        raise Exception(f"Error reading URL: {str(e)}")

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

@st.cache_data(show_spinner=False)
def generate_embedding(text, model):
    try:
        return model.encode(text)
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

def query_llm(prompt, model_name):
    """Query the LLM with given prompt"""
    global groq_client
    
    if not st.session_state.get('api_key'):
        return "Error: API key not found. Please configure your API key first."
        
    try:
        # Reinitialize client if needed
        if groq_client is None:
            if not initialize_groq_client(st.session_state.api_key):
                return "Error: Failed to initialize Groq client. Please check your API key."
        
        # Create completion
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        
        return completion.choices[0].message.content
        
    except Exception as e:
        logger.error(f"Error in query_llm: {str(e)}")
        # Try to reinitialize client on error
        groq_client = None
        return f"Error: Failed to get response from Groq API. Please check your API key and try again."

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        raise Exception(f"Error saving uploaded file: {str(e)}")

def create_index(embeddings):
    try:
        if not embeddings:
            return None
            
        embeddings_array = np.array(embeddings)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array.astype('float32'))
        return index
    except Exception as e:
        raise Exception(f"Error creating index: {str(e)}")

def search_index(index, query_embedding, k=3):
    try:
        if index is None:
            return []
            
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        return I[0]
    except Exception as e:
        raise Exception(f"Error searching index: {str(e)}")

def validate_api_key_format(api_key: str) -> bool:
    """Validate GROQ API key format"""
    try:
        if not api_key:
            return False
            
        api_key = api_key.strip()
        
        if not api_key.startswith('gsk_'):
            logger.warning("Invalid API key format: doesn't start with 'gsk_'")
            return False
            
        if len(api_key) < 20:
            logger.warning("Invalid API key format: too short")
            return False
            
        return True
    except Exception as e:
        logger.error(f"Error validating API key format: {str(e)}")
        return False