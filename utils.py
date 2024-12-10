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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

groq_client = None

def set_api_key(api_key):
    global groq_client
    if api_key:
        try:
            groq_client = Groq(api_key=api_key)
            return True
        except Exception as e:
            print(f"Warning in setting API key: {str(e)}")
            return False
    return False

def query_llm(prompt, model_name):
    global groq_client
    try:
        if not groq_client:
            return "Silakan masukkan GROQ API key Anda untuk menggunakan fitur chat"

        completion = groq_client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "Anda adalah asisten AI yang membantu memberikan informasi yang akurat dan jelas."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=1024
        )
        
        if completion and hasattr(completion.choices[0].message, 'content'):
            return completion.choices[0].message.content
        return "Maaf, tidak dapat memproses permintaan Anda saat ini"

    except Exception as e:
        logger.error(f"Error in query_llm: {str(e)}")
        return "Terjadi kesalahan saat memproses permintaan. Silakan coba lagi atau periksa API key Anda"

def get_available_models():
    return ["mixtral-8x7b-32768", "gemma-7b-it", "llama2-70b-4096"]