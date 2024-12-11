import os
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import streamlit as st
import tempfile
import logging
from typing import List, Optional
import groq

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@st.cache_resource
def load_embedding_model(model_name: str) -> SentenceTransformer:
    return SentenceTransformer(model_name)

def get_groq_client():
    """Get Groq client with current API key"""
    if not st.session_state.get("api_key"):
        return None
    try:
        client = groq.Groq(
            api_key=st.session_state.api_key
        )
        return client
    except Exception as e:
        logger.error(f"Error creating Groq client: {str(e)}")
        return None

def set_api_key(api_key: str) -> None:
    """Simply store the API key in session state"""
    st.session_state.api_key = api_key

def get_available_models() -> List[str]:
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama2-70b-chat"]

def read_file(file_path: str) -> str:
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            return "\n".join(page.extract_text() for page in pdf_reader.pages)
    elif file_extension.lower() == '.docx':
        doc = docx.Document(file_path)
        return "\n".join(para.text for para in doc.paragraphs)
    else:
        raise ValueError("Unsupported file type")

@st.cache_data
def read_url(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

@st.cache_data(show_spinner=False)
def generate_embedding(text: str, model: SentenceTransformer) -> np.ndarray:
    return model.encode(text)

def query_llm(prompt: str, model_name: str) -> str:
    """Query the LLM with better error handling"""
    try:
        # Get client for each query to ensure fresh state
        client = get_groq_client()
        if client is None:
            return "Error: Please configure your API key first."

        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        return completion.choices[0].message.content

    except groq.error.AuthenticationError:
        logger.error("Authentication failed with Groq API")
        return "Error: Invalid API key. Please check your API key and try again."
    except groq.error.APIError as e:
        logger.error(f"Groq API error: {str(e)}")
        return f"Error communicating with Groq API: {str(e)}"
    except Exception as e:
        logger.error(f"Unexpected error in query_llm: {str(e)}")
        return f"An unexpected error occurred: {str(e)}"

def save_uploaded_file(uploaded_file) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def create_index(embeddings: List[np.ndarray]) -> Optional[faiss.Index]:
    if not embeddings:
        return None
        
    embeddings_array = np.array(embeddings)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array.astype('float32'))
    return index

def search_index(index: faiss.Index, query_embedding: np.ndarray, k: int = 3) -> List[int]:
    if index is None:
        return []
        
    D, I = index.search(np.array([query_embedding]).astype('float32'), k)
    return I[0]

def validate_api_key_format(api_key: str) -> bool:
    """Validate GROQ API key format"""
    if not api_key:
        return False
        
    api_key = api_key.strip()
    # Basic validation that it starts with gsk_ and is of reasonable length
    return api_key.startswith('gsk_') and len(api_key) >= 20