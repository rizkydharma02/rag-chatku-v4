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

groq_client = None

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)

def set_api_key(api_key):
    global groq_client
    if api_key:
        try:
            groq_client = Groq(api_key=api_key)
        except Exception as e:
            print(f"Error setting API key: {str(e)}")
    else:
        groq_client = None

def get_available_models():
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama3-8b-8192"]

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

def query_llm(prompt, model_name):
    try:
        if groq_client is None:
            raise ValueError("Please enter a valid API key first.")
            
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=1000
        )
        
        return completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while querying the LLM: {str(e)}"

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
        print(f"Error validating API key format: {str(e)}")
        return False