# utils.py
import os
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import groq
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
    return SentenceTransformer(model_name)

def get_available_models():
    return ["mixtral-8x7b-32768"]

def set_api_key(api_key):
    """Set GROQ API key"""
    try:
        if api_key and api_key.strip():
            api_key = api_key.strip()
            if not api_key.startswith('gsk_'):
                return False
            os.environ["GROQ_API_KEY"] = api_key
            return True
        return False
    except Exception as e:
        logger.error(f"Error setting API key: {str(e)}")
        return False

def query_llm(prompt, model_name):
    """Query the LLM with proper error handling"""
    try:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Error: API key not found"

        # Initialize groq client
        client = groq.Groq()
        client.api_key = api_key  # Set API key directly

        try:
            # Make API call
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": "You are an AI assistant that provides clear, accurate responses."},
                    {"role": "user", "content": prompt}
                ]
            )

            # Return response content
            if hasattr(response.choices[0], 'message'):
                return response.choices[0].message.content
            else:
                return "Error: Invalid response format"

        except groq.error.GroqError as e:
            logger.error(f"Groq API Error: {str(e)}")
            return f"Error: {str(e)}"
            
        except Exception as e:
            logger.error(f"Error during API call: {str(e)}")
            return "Error: Failed to communicate with Groq API"

    except Exception as e:
        logger.error(f"General error in query_llm: {str(e)}")
        return f"Error: {str(e)}"

# Rest of the utility functions remain the same
def read_pdf(file_path):
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages)
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)
    except Exception as e:
        logger.error(f"Error reading DOCX: {str(e)}")
        raise

@st.cache_data
def read_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except Exception as e:
        logger.error(f"Error reading URL: {str(e)}")
        raise

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

def generate_embedding(text, model):
    try:
        return model.encode(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
        raise

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
        logger.error(f"Error creating index: {str(e)}")
        raise

def search_index(index, query_embedding, k=3):
    try:
        if index is None:
            return []
            
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        return I[0]
    except Exception as e:
        logger.error(f"Error searching index: {str(e)}")
        raise