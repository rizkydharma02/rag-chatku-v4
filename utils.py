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
import torch

# Global variables
groq_client = None

@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name):
    """Load sentence transformer model with better error handling"""
    try:
        # Ensure CUDA is available if needed
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = SentenceTransformer(model_name, device=device)
        return model
    except Exception as e:
        print(f"Error loading embedding model: {str(e)}")
        # Return None instead of raising error to handle gracefully
        return None

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

def query_llm(prompt, model_name):
    """Query LLM with improved reliability"""
    try:
        if not groq_client:
            raise ValueError("GROQ client not initialized")

        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt")

        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            stream=False  # Disable streaming for better reliability
        )

        if not completion.choices:
            raise ValueError("No response from API")

        return completion.choices[0].message.content

    except Exception as e:
        print(f"Error in query_llm: {str(e)}")
        return None

def generate_embedding(text, model):
    """Generate embeddings with improved handling"""
    try:
        if not model:
            print("Embedding model not initialized")
            return None

        if not text or not text.strip():
            print("Empty text input")
            return None

        # Generate embedding
        embedding = model.encode(text.strip(), convert_to_tensor=False)
        return embedding

    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        return None

def create_index(embeddings):
    """Create FAISS index with better validation"""
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
    """Search index with improved validation"""
    try:
        if index is None or query_embedding is None:
            return []

        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        return I[0]

    except Exception as e:
        print(f"Error searching index: {str(e)}")
        return []

def read_pdf(file_path):
    """Read PDF with improved error handling"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            return text.strip()
    except Exception as e:
        print(f"Error reading PDF: {str(e)}")
        return None

def read_docx(file_path):
    """Read DOCX with improved error handling"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
        return text.strip()
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def read_url(url):
    """Read URL content with improved error handling"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text().strip()
    except Exception as e:
        print(f"Error reading URL: {str(e)}")
        return None

def read_file(file_path):
    """Read file based on extension"""
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
    """Save uploaded file with better error handling"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        print(f"Error saving uploaded file: {str(e)}")
        return None

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