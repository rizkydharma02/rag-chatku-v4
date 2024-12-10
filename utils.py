# utils.py
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_available_models():
    """Get list of available models"""
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama2-70b-chat"]

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

def set_api_key(api_key):
    if api_key and api_key.strip():
        os.environ["GROQ_API_KEY"] = api_key.strip()
        return True
    return False

def query_llm(prompt, model_name):
    """Query the LLM with better error handling"""
    try:
        # Get API key from environment
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return "Error: Please set a valid API key."

        # Create Groq client
        client = Groq(api_key=api_key)

        # Make the API call
        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048,
            top_p=1
        )

        # Extract and return the response
        if hasattr(completion.choices[0], 'message'):
            return completion.choices[0].message.content
        return "Error: No response from model."

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in query_llm: {error_msg}")
        if "api_key" in error_msg.lower():
            return "Error: Invalid API key. Please check your API key configuration."
        if "unexpected keyword" in error_msg.lower():
            return "Error: Please try again in a few moments."
        return f"Error: {error_msg}"

def read_file(file_path):
    _, file_extension = os.path.splitext(file_path)
    if file_extension.lower() == '.pdf':
        return read_pdf(file_path)
    elif file_extension.lower() == '.docx':
        return read_docx(file_path)
    else:
        raise ValueError("Unsupported file type")

def read_pdf(file_path):
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text

def read_docx(file_path):
    doc = docx.Document(file_path)
    return "\n".join(para.text for para in doc.paragraphs)

@st.cache_data
def read_url(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text()

def generate_embedding(text, model):
    return model.encode(text)

def save_uploaded_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())
        return tmp_file.name

def create_index(embeddings):
    if not embeddings:
        return None
        
    embeddings_array = np.array(embeddings)
    dimension = embeddings_array.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_array.astype('float32'))
    return index

def search_index(index, query_embedding, k=3):
    if index is None:
        return []
        
    D, I = index.search(np.array([query_embedding]).astype('float32'), k)
    return I[0]