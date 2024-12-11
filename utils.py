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

# Global variable for Groq client
groq_client = None

@st.cache_resource
def load_embedding_model(model_name):
    """Load sentence transformer model"""
    return SentenceTransformer(model_name)

def set_api_key(api_key):
    """Set API key with basic validation"""
    global groq_client
    try:
        if api_key and api_key.strip().startswith('gsk_'):
            groq_client = Groq(api_key=api_key)
            return True
    except Exception as e:
        print(f"Error setting API key: {str(e)}")
        groq_client = None
    return False

def get_available_models():
    """Get list of available models"""
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama3-8b-8192"]

def read_pdf(file_path):
    """Read and extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join(page.extract_text() for page in pdf_reader.pages)
            return text
    except Exception as e:
        raise Exception(f"Error reading PDF: {str(e)}")

def read_docx(file_path):
    """Read and extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        return text
    except Exception as e:
        raise Exception(f"Error reading DOCX: {str(e)}")

@st.cache_data
def read_url(url):
    """Read and extract text from URL"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        raise Exception(f"Error reading URL: {str(e)}")

def read_file(file_path):
    """Read file based on extension"""
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

def generate_embedding(text, model):
    """Generate embeddings from text"""
    try:
        return model.encode(text)
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

def query_llm(prompt, model_name):
    """Query LLM with simplified reliable handling"""
    try:
        if not groq_client:
            raise ValueError("GROQ client not initialized")

        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            stream=False  # Disable streaming for production stability
        )

        if not completion.choices:
            raise ValueError("No response from API")

        response = completion.choices[0].message.content
        return response if response else None

    except Exception as e:
        print(f"Error in query_llm: {str(e)}")
        return None

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        raise Exception(f"Error saving uploaded file: {str(e)}")

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
        raise Exception(f"Error creating index: {str(e)}")

def search_index(index, query_embedding, k=3):
    """Search for similar documents in FAISS index"""
    try:
        if index is None:
            return []
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        return I[0]
    except Exception as e:
        raise Exception(f"Error searching index: {str(e)}")

def validate_api_key(api_key: str) -> bool:
    """Validate API key format"""
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
        print(f"Error validating API key: {str(e)}")
        return False

# Helper function for retrying API calls
def retry_api_call(func, max_retries=3, delay=1):
    """Retry API calls with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt == max_retries - 1:
                raise e
            time.sleep(delay * (2 ** attempt))