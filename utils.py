import os
import streamlit as st
from groq import Groq
import PyPDF2
import docx
import requests
from bs4 import BeautifulSoup
import tempfile

# Global variables
groq_client = None

def set_api_key(api_key):
    """Set API key with simple validation"""
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
    """Query LLM with simplified handling"""
    try:
        if not groq_client:
            print("GROQ client not initialized")
            return None

        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
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

def read_file(file_path):
    """Read file content based on extension"""
    try:
        _, file_extension = os.path.splitext(file_path)
        if file_extension.lower() == '.pdf':
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text.strip()
        elif file_extension.lower() == '.docx':
            doc = docx.Document(file_path)
            text = "\n".join([para.text for para in doc.paragraphs])
            return text.strip()
        else:
            raise ValueError("Unsupported file type")
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return None

def read_url(url):
    """Read URL content"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text().strip()
    except Exception as e:
        print(f"Error reading URL: {str(e)}")
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
    return api_key.startswith('gsk_') and len(api_key) >= 20