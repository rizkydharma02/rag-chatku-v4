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
from typing import List, Optional, Any, Dict
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables
groq_client = None

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
    global groq_client
    if api_key:
        try:
            groq_client = Groq(api_key=api_key)
            logger.info("GROQ client initialized successfully")
        except Exception as e:
            logger.error(f"Error setting API key: {str(e)}")
            groq_client = None
    else:
        groq_client = None

def get_available_models() -> List[str]:
    """Get list of available LLM models"""
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama2-70b-chat"]

@st.cache_data(show_spinner=False)
def read_pdf(file_path: str) -> str:
    """Read and extract text from PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = "\n".join(page.extract_text() for page in pdf_reader.pages)
            logger.info(f"Successfully read PDF: {file_path}")
            return text
    except Exception as e:
        logger.error(f"Error reading PDF: {str(e)}")
        raise

@st.cache_data(show_spinner=False)
def read_docx(file_path: str) -> str:
    """Read and extract text from DOCX file"""
    try:
        doc = docx.Document(file_path)
        text = "\n".join(para.text for para in doc.paragraphs)
        logger.info(f"Successfully read DOCX: {file_path}")
        return text
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
        text = soup.get_text(separator="\n", strip=True)
        logger.info(f"Successfully read URL: {url}")
        return text
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

async def query_llm_async(prompt: str, model_name: str) -> str:
    """Asynchronously query the LLM with given prompt"""
    try:
        if groq_client is None:
            raise ValueError("API key not set. Please configure your API key first.")
            
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000,
            top_p=1
        )
        
        response = completion.choices[0].message.content
        logger.info("Successfully generated LLM response")
        return response
        
    except Exception as e:
        logger.error(f"Error querying LLM: {str(e)}")
        return f"An error occurred while querying the LLM: {str(e)}"

def query_llm(prompt: str, model_name: str) -> str:
    """Synchronous wrapper for query_llm_async"""
    try:
        return asyncio.run(query_llm_async(prompt, model_name))
    except Exception as e:
        logger.error(f"Error in query_llm: {str(e)}")
        return f"An error occurred: {str(e)}"

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temporary location"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            logger.info(f"Successfully saved uploaded file: {uploaded_file.name}")
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
        
        logger.info("Successfully created FAISS index")
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
        logger.info("Successfully searched index")
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
        
        # Basic format validation
        if not api_key.startswith('gsk_'):
            logger.warning("Invalid API key format: doesn't start with 'gsk_'")
            return False
            
        if len(api_key) < 20:
            logger.warning("Invalid API key format: too short")
            return False
            
        logger.info("API key format validation successful")
        return True
        
    except Exception as e:
        logger.error(f"Error validating API key format: {str(e)}")
        return False