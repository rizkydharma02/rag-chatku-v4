import os
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
from groq import Groq
import streamlit as st
import tempfile

# Global variable for GROQ client
groq_client = None

def set_api_key(api_key):
    """Set GROQ API key"""
    global groq_client
    if api_key and api_key.strip():
        groq_client = Groq(api_key=api_key.strip())
        return True
    return False

def get_available_models():
    """Get available models"""
    return ["mixtral-8x7b-32768"]

def query_llm(prompt, model_name):
    """Query LLM with basic error handling"""
    if not groq_client:
        return None

    try:
        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1000,
            stream=False
        )
        
        if completion and completion.choices:
            return completion.choices[0].message.content
        return None
        
    except Exception as e:
        print(f"Error in query_llm: {str(e)}")
        return None

def read_pdf(file_path):
    """Read PDF file"""
    try:
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = []
            for page in pdf_reader.pages:
                text.append(page.extract_text())
            return "\n".join(text)
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
        return "\n".join(text)
    except Exception as e:
        print(f"Error reading DOCX: {str(e)}")
        return None

def read_url(url):
    """Read URL content"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
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
    """Save uploaded file"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            return tmp_file.name
    except Exception as e:
        print(f"Error saving file: {str(e)}")
        return None

def validate_api_key(api_key: str) -> bool:
    """Validate API key format"""
    if not api_key:
        return False
    api_key = api_key.strip()
    if not api_key.startswith('gsk_'):
        return False
    if len(api_key) < 20:
        return False
    return True