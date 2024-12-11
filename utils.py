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
import time

# Global variables
groq_client = None
MAX_RETRIES = 3
RETRY_DELAY = 1

@st.cache_resource
def load_embedding_model(model_name):
    """Load embedding model with improved caching"""
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        print(f"Error loading embedding model: {str(e)}")
        raise

def validate_api_key(api_key):
    """Validate API key format and functionality"""
    try:
        if not api_key:
            return False
            
        api_key = api_key.strip()
        if not api_key.startswith('gsk_'):
            return False
            
        if len(api_key) < 20:
            return False
            
        # Test API key with a minimal query
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10
        )
        
        return response is not None
        
    except Exception as e:
        print(f"API key validation error: {str(e)}")
        return False

def set_api_key(api_key):
    """Set API key with validation and error handling"""
    global groq_client
    try:
        if not api_key:
            raise ValueError("API key cannot be empty")
            
        # Validate API key
        if not validate_api_key(api_key):
            raise ValueError("Invalid API key")
            
        # Initialize client
        groq_client = Groq(api_key=api_key)
        return True
        
    except Exception as e:
        print(f"Error setting API key: {str(e)}")
        groq_client = None
        raise

def get_available_models():
    """Get available models with validation"""
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama3-8b-8192"]

def read_pdf(file_path):
    """Read PDF with improved error handling"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                content = page.extract_text()
                if content:
                    text.append(content)
            return "\n".join(text) if text else ""
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        raise

def read_docx(file_path):
    """Read DOCX with improved error handling"""
    try:
        doc = docx.Document(file_path)
        text = []
        for para in doc.paragraphs:
            if para.text:
                text.append(para.text)
        return "\n".join(text) if text else ""
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        raise

@st.cache_data
def read_url(url):
    """Read URL with improved error handling and retry logic"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        for attempt in range(MAX_RETRIES):
            try:
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                text = soup.get_text()
                return text.strip() if text else ""
            except requests.RequestException as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY)
                
    except Exception as e:
        print(f"Error reading URL: {str(e)}")
        raise

def read_file(file_path):
    """Read file with improved validation"""
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.pdf':
            return read_pdf(file_path)
        elif file_extension.lower() == '.docx':
            return read_docx(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        raise

def generate_embedding(text, model):
    """Generate embedding with improved input validation"""
    try:
        if not text or not text.strip():
            raise ValueError("Empty text input")
            
        if not model:
            raise ValueError("Model not initialized")
            
        return model.encode(text.strip())
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        raise

def query_llm(prompt, model_name):
    """Query LLM with improved reliability and retry logic"""
    try:
        if groq_client is None:
            raise ValueError("GROQ client not initialized")
            
        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt")
            
        if not model_name:
            raise ValueError("Model name not specified")
            
        for attempt in range(MAX_RETRIES):
            try:
                completion = groq_client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=1000,
                    stream=False  # Disable streaming for better reliability
                )
                
                if not completion or not completion.choices:
                    raise ValueError("Invalid API response")
                    
                response = completion.choices[0].message.content
                if not response:
                    raise ValueError("Empty response from API")
                    
                return response
                
            except Exception as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                print(f"Retrying query... (attempt {attempt + 1}/{MAX_RETRIES})")
                time.sleep(RETRY_DELAY)
                
    except Exception as e:
        error_msg = f"Error querying LLM: {str(e)}"
        print(error_msg)
        raise Exception(error_msg)

def save_uploaded_file(uploaded_file):
    """Save uploaded file with improved validation"""
    try:
        if not uploaded_file:
            raise ValueError("No file uploaded")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        print(f"Error saving uploaded file: {str(e)}")
        raise

def create_index(embeddings):
    """Create search index with improved validation"""
    try:
        if not embeddings or len(embeddings) == 0:
            return None
            
        embeddings_array = np.array(embeddings)
        if embeddings_array.size == 0:
            raise ValueError("Empty embeddings array")
            
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array.astype('float32'))
        
        return index
    except Exception as e:
        print(f"Error creating index: {str(e)}")
        raise

def search_index(index, query_embedding, k=3):
    """Search index with improved validation"""
    try:
        if index is None:
            return []
            
        if query_embedding is None or len(query_embedding) == 0:
            raise ValueError("Invalid query embedding")
            
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        return I[0]
    except Exception as e:
        print(f"Error searching index: {str(e)}")
        raise