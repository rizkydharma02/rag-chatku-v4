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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_client = None

def initialize_groq_client(api_key):
    global groq_client
    try:
        if api_key and validate_api_key_format(api_key):
            groq_client = Groq(api_key=api_key)
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
    try:
        return SentenceTransformer(model_name)
    except Exception as e:
        logger.error(f"Error loading embedding model: {str(e)}")
        raise

def set_api_key(api_key):
    global groq_client
    try:
        if api_key and validate_api_key_format(api_key):
            groq_client = Groq(api_key=api_key)
            logger.info("API key set successfully")
            return True
        logger.error("Invalid API key format in set_api_key")
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
            text = "\n".join(page.extract_text() for page in pdf_reader.pages)
            return text if text.strip() else "No text content found in PDF"
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise Exception(f"Error reading PDF: {str(e)}")

def read_docx(file_path):
    try:
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        return text if text.strip() else "No text content found in DOCX"
    except Exception as e:
        logger.error(f"Error reading DOCX: {str(e)}")
        raise Exception(f"Error reading DOCX: {str(e)}")

@st.cache_data
def read_url(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        text = soup.get_text()
        return text if text.strip() else "No text content found at URL"
    except Exception as e:
        logger.error(f"Error reading URL: {str(e)}")
        raise Exception(f"Error reading URL: {str(e)}")

def read_file(file_path):
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.pdf':
            return read_pdf(file_path)
        elif file_extension.lower() == '.docx':
            return read_docx(file_path)
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        logger.error(f"Error reading file: {str(e)}")
        raise

def generate_embedding(text, model):
    try:
        if not isinstance(text, str):
            raise ValueError("Input text must be a string")
        if not text.strip():
            raise ValueError("Input text cannot be empty")
        return model.encode(text)
    except Exception as e:
        logger.error(f"Error generating embedding: {str(e)}")
        raise Exception(f"Error generating embedding: {str(e)}")

def query_llm(prompt, model_name):
    try:
        if not groq_client:
            logger.error("GROQ client not initialized")
            return "Error: GROQ client not initialized"
            
        if not prompt or not isinstance(prompt, str):
            logger.error(f"Invalid prompt type: {type(prompt)}")
            return "Error: Invalid prompt"
            
        if not model_name or model_name not in get_available_models():
            logger.error(f"Invalid model name: {model_name}")
            return "Error: Invalid model name"

        logger.info(f"Sending request to GROQ API with model: {model_name}")
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant who provides clear and accurate responses."},
            {"role": "user", "content": prompt}
        ]

        logger.info(f"Request details: {json.dumps({'model': model_name, 'messages': messages}, indent=2)}")

        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=0.7,
            max_tokens=1000,
            top_p=1,
            stop=None
        )
        
        if not completion or not hasattr(completion, 'choices'):
            logger.error("Invalid completion response")
            return "Error: Invalid response from model"
            
        if not completion.choices:
            logger.error("No choices in completion")
            return "Error: No response generated"
            
        response = completion.choices[0].message.content
        
        if not response or not isinstance(response, str):
            logger.error("Invalid response content")
            return "Error: Invalid response content"
            
        logger.info(f"Successfully received response of length: {len(response)}")
        return response.strip()
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in query_llm: {error_msg}")
        return f"Error occurred: {error_msg}"

def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        logger.error(f"Error saving uploaded file: {str(e)}")
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
        logger.error(f"Error creating index: {str(e)}")
        raise Exception(f"Error creating index: {str(e)}")

def search_index(index, query_embedding, k=3):
    try:
        if index is None:
            return []
            
        D, I = index.search(np.array([query_embedding]).astype('float32'), k)
        return I[0]
    except Exception as e:
        logger.error(f"Error searching index: {str(e)}")
        raise Exception(f"Error searching index: {str(e)}")

def validate_api_key_format(api_key: str) -> bool:
    try:
        if not api_key:
            return False
            
        api_key = api_key.strip()
        return api_key.startswith('gsk_') and len(api_key) >= 20
    except Exception as e:
        logger.error(f"Error validating API key format: {str(e)}")
        return False