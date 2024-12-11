import os
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from groq import Groq, AsyncGroq
import faiss
import numpy as np
import streamlit as st
import tempfile
import time

# Global variables
groq_client = None

@st.cache_resource
def load_embedding_model(model_name):
    """Load sentence transformer model"""
    try:
        model = SentenceTransformer(model_name)
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def set_api_key(api_key):
    """Set API key with validation"""
    global groq_client
    try:
        if not api_key or not isinstance(api_key, str):
            return False
            
        api_key = api_key.strip()
        if not api_key.startswith('gsk_'):
            return False
            
        groq_client = Groq(api_key=api_key)
        
        # Test the API key
        test_response = groq_client.chat.completions.create(
            model="mixtral-8x7b-32768",
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10,
            stream=False
        )
        return bool(test_response)
        
    except Exception as e:
        print(f"Error setting API key: {str(e)}")
        groq_client = None
        return False

def get_available_models():
    """Get available models"""
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama3-8b-8192"]

def query_llm(prompt, model_name="mixtral-8x7b-32768", max_retries=3):
    """Query LLM with retries"""
    if not groq_client:
        return "GROQ client not initialized. Please check your API key."

    for attempt in range(max_retries):
        try:
            completion = groq_client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                stream=False
            )
            
            if completion and completion.choices:
                return completion.choices[0].message.content
                
            time.sleep(1)  # Wait before retry
            
        except Exception as e:
            print(f"Error in attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                return f"Error: {str(e)}"
            time.sleep(1)  # Wait before retry
    
    return "Failed to get response after multiple attempts."

def generate_embedding(text, model):
    """Generate embeddings"""
    try:
        if not text or not text.strip():
            return None
            
        if not model:
            return None
            
        embedding = model.encode(text.strip())
        return embedding
        
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def read_pdf(file_path):
    """Read PDF file"""
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
        return None

def read_docx(file_path):
    """Read DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = []
        for para in doc.paragraphs:
            if para.text:
                text.append(para.text)
        return "\n".join(text) if text else ""
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        return None

@st.cache_data
def read_url(url):
    """Read URL content"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        print(f"Error reading URL: {str(e)}")
        return None

def read_file(file_path):
    """Read file based on extension"""
    if not file_path:
        return None
        
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
        return None

def save_uploaded_file(uploaded_file):
    """Save uploaded file"""
    if not uploaded_file:
        return None
        
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return None

def create_index(embeddings):
    """Create FAISS index"""
    try:
        if not embeddings or len(embeddings) == 0:
            return None
            
        embeddings_array = np.array(embeddings)
        dimension = embeddings_array.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_array.astype('float32'))
        return index
        
    except Exception as e:
        print(f"Error creating index: {str(e)}")
        return None

def search_index(index, query_embedding, k=3):
    """Search FAISS index"""
    try:
        if not index or not query_embedding:
            return []
            
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        return I[0]
        
    except Exception as e:
        print(f"Error searching index: {str(e)}")
        return []

def validate_api_key(api_key: str) -> bool:
    """Validate API key format"""
    if not api_key or not isinstance(api_key, str):
        return False
        
    api_key = api_key.strip()
    if not api_key.startswith('gsk_'):
        return False
        
    if len(api_key) < 20:
        return False
        
    return True