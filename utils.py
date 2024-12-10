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
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_client = None

def initialize_groq_client(api_key):
    global groq_client
    try:
        if api_key and validate_api_key_format(api_key):
            groq_client = Groq(api_key=api_key)
            # Test the client with a simple request
            test_response = groq_client.chat.completions.create(
                model="mixtral-8x7b-32768",
                messages=[{"role": "user", "content": "test"}],
                temperature=0.5,
                max_tokens=10
            )
            logger.info("GROQ client initialized successfully")
            return bool(test_response)
        logger.error("Invalid API key format")
        return False
    except Exception as e:
        logger.error(f"Error initializing GROQ client: {str(e)}")
        return False

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

def query_llm(prompt, model_name):
    """
    Fungsi untuk mengirim query ke model LLM dan mendapatkan respons
    """
    global groq_client
    try:
        if not groq_client:
            raise ValueError("GROQ client belum diinisialisasi. Pastikan API key valid.")

        logger.info(f"Sending request with model: {model_name}")
        logger.info(f"Prompt: {prompt[:100]}...")  # Print first 100 chars of prompt

        messages = [
            {
                "role": "system",
                "content": "Anda adalah asisten AI yang membantu dengan memberikan informasi yang akurat dan jelas."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        response = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=2048,
            top_p=1,
            stream=False
        )

        logger.info("Response received from API")
        
        if response and hasattr(response.choices[0].message, 'content'):
            result = response.choices[0].message.content
            logger.info(f"Response content: {result[:100]}...")  # Print first 100 chars
            return result
        else:
            logger.error("No valid response content found")
            return "Maaf, tidak bisa mendapatkan respons yang valid dari model."

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in query_llm: {error_msg}")
        return f"Terjadi kesalahan: {error_msg}"

def set_api_key(api_key):
    """
    Fungsi untuk mengatur API key dan menginisialisasi GROQ client
    """
    global groq_client
    try:
        if api_key:
            groq_client = Groq(api_key=api_key)
            logger.info("GROQ client initialized in set_api_key")
            return True
        return False
    except Exception as e:
        logger.error(f"Error setting API key: {str(e)}")
        return False

def get_available_models():
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama2-70b-4096"]

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

def generate_embedding(text, model):
    try:
        return model.encode(text)
    except Exception as e:
        raise Exception(f"Error generating embedding: {str(e)}")

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
    if not api_key:
        return False
    try:
        api_key = api_key.strip()
        return api_key.startswith('gsk_') and len(api_key) >= 20
    except Exception as e:
        logger.error(f"Error validating API key format: {str(e)}")
        return False