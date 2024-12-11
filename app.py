import streamlit as st
import time
from datetime import datetime

# Must be the first Streamlit command
st.set_page_config(
    page_title="Chatku AI",
    page_icon="🤖",
    layout="wide"
)

import os
from utils import (
    load_embedding_model, read_file, read_url, generate_embedding,
    query_llm, save_uploaded_file, create_index, search_index,
    set_api_key, get_available_models
)
from db_utils import DatabaseManager
from auth_utils import (
    login_user, register_user, get_current_user, logout_user,
    validate_groq_api_key
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state with all required variables"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Initialize basic required variables
    default_values = {
        'chat_history': [],
        'documents': [],
        'processed_files': [],
        'processed_urls': [],
        'selected_model': "mixtral-8x7b-32768",
        'api_key': None,
        'is_processing': False,
        'error_message': None
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Load API key if user is logged in
    if 'user_id' in st.session_state and not st.session_state.api_key:
        api_key = st.session_state.db_manager.get_api_key(st.session_state.user_id)
        if api_key:
            st.session_state.api_key = api_key
            set_api_key(api_key)

def handle_query(query):
    """Handle chat query with improved response handling"""
    if not st.session_state.api_key:
        st.error("Silakan masukkan GROQ API key yang valid")
        return False

    try:
        # Set API key before each query
        set_api_key(st.session_state.api_key)
        
        # Add user message
        with st.chat_message("user"):
            st.write(query)
            st.session_state.chat_history.append(("user", query))
        
        # Get AI response
        with st.spinner("Menghasilkan respons..."):
            response = query_llm(query, st.session_state.selected_model)
            
            if response:
                with st.chat_message("assistant"):
                    st.write(response)
                    st.session_state.chat_history.append(("assistant", response))
                return True
            else:
                st.error("Gagal mendapatkan respons. Silakan coba lagi.")
                return False
                
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        return False

def handle_main_area():
    """Handle main chat area with improved UI"""
    st.title("Chatku AI")
    st.caption("Chatku AI dengan Retrieval Augmented Generation")
    
    # Display chat history
    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)
    
    # Chat input
    if prompt := st.chat_input("Ketik pesan Anda di sini..."):
        if not st.session_state.is_processing:
            try:
                st.session_state.is_processing = True
                success = handle_query(prompt)
                if success:
                    st.rerun()
            finally:
                st.session_state.is_processing = False

def handle_sidebar():
    """Handle sidebar with essential settings"""
    user = get_current_user()
    if not user:
        st.error("User not authenticated")
        return

    st.sidebar.write(f"👤 **{user['email']}**")
    st.sidebar.write("---")
    
    # API Settings
    with st.sidebar.expander("API Settings", expanded=False):
        current_api_key = st.session_state.get('api_key', '')
        
        if current_api_key:
            st.success("API Key sudah terpasang")
        else:
            st.warning("API Key belum diatur")
        
        with st.form("api_key_form"):
            new_api_key = st.text_input(
                "GROQ API Key",
                value=current_api_key,
                type="password",
                placeholder="Masukkan GROQ API Key Anda",
                help="Dapatkan API key dari https://console.groq.com/"
            )
            
            if st.form_submit_button("Simpan API Key"):
                if not new_api_key:
                    st.error("API key tidak boleh kosong")
                elif new_api_key == current_api_key:
                    st.info("API key tidak berubah")
                else:
                    try:
                        success = st.session_state.db_manager.save_api_key(
                            user_id=user["id"],
                            api_key=new_api_key
                        )
                        
                        if success:
                            st.session_state.api_key = new_api_key
                            set_api_key(new_api_key)
                            st.success("API key berhasil disimpan!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Gagal menyimpan API key")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")

    # Document Processing
    with st.sidebar.expander("Document Processing", expanded=False):
        # File Upload
        uploaded_file = st.file_uploader(
            "Upload File (PDF/Word)",
            type=['pdf', 'docx']
        )
        
        # URL Input
        url = st.text_input(
            "URL Input",
            placeholder="Atau masukkan URL"
        )
        
        if st.button("Proses File/URL"):
            with st.spinner("Memproses..."):
                try:
                    if uploaded_file:
                        process_file(uploaded_file)
                    elif url:
                        process_url(url)
                    else:
                        st.warning("Silakan upload file atau masukkan URL")
                except Exception as e:
                    st.error(f"Gagal memproses: {str(e)}")

        st.write("---")
        if st.button("Hapus Semua Data", use_container_width=True):
            clean_session_data()

    st.sidebar.write("---")
    if st.sidebar.button("Logout", use_container_width=True):
        logout_user()
        st.rerun()

def process_file(uploaded_file):
    """Process uploaded file"""
    try:
        file_path = save_uploaded_file(uploaded_file)
        if file_path:
            content = read_file(file_path)
            if content:
                st.session_state.documents.append(content)
                st.session_state.processed_files.append(uploaded_file.name)
                st.success(f"File '{uploaded_file.name}' berhasil diproses!")
            else:
                st.error("Gagal membaca konten file")
    except Exception as e:
        st.error(f"Gagal memproses file: {str(e)}")

def process_url(url):
    """Process URL"""
    try:
        content = read_url(url)
        if content:
            st.session_state.documents.append(content)
            st.session_state.processed_urls.append(url)
            st.success(f"URL '{url}' berhasil diproses!")
        else:
            st.error("Gagal membaca konten URL")
    except Exception as e:
        st.error(f"Gagal memproses URL: {str(e)}")

def clean_session_data():
    """Clean session data"""
    try:
        with st.spinner("Membersihkan data..."):
            st.session_state.documents = []
            st.session_state.processed_files = []
            st.session_state.processed_urls = []
            st.session_state.chat_history = []
            st.session_state.is_processing = False
            st.session_state.error_message = None
            
            st.success("Semua data berhasil dibersihkan!")
            st.rerun()
    except Exception as e:
        st.error(f"Gagal membersihkan data: {str(e)}")

def render_login_page():
    """Render login page"""
    st.title("🤖 Chatku AI")
    st.caption("Chatku AI Dengan Retrieval Augmented Generation")
    
    tab1, tab2 = st.tabs(["Login", "Daftar"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Login")
            login_email = st.text_input("Email", placeholder="Masukkan email anda")
            login_password = st.text_input("Password", type="password", placeholder="Masukkan password anda")
            
            if st.form_submit_button("Login", use_container_width=True):
                if login_email and login_password:
                    try:
                        result = login_user(st.session_state.db_manager, login_email, login_password)
                        if result and result.get("access_token"):
                            if result["user"].get("groq_api_key"):
                                st.session_state.api_key = result["user"]["groq_api_key"]
                                set_api_key(result["user"]["groq_api_key"])
                            st.success("✅ Berhasil login!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Email atau password salah")
                    except Exception as e:
                        st.error(f"Gagal login: {str(e)}")
                else:
                    st.warning("Silakan isi email dan password")
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Daftar Akun Baru")
            signup_email = st.text_input("Email", key="signup_email", placeholder="Masukkan email anda")
            signup_password = st.text_input("Password", type="password", key="signup_pass", placeholder="Buat password anda")
            signup_password_confirm = st.text_input("Konfirmasi Password", type="password", placeholder="Masukkan ulang password")
            signup_groq_api = st.text_input(
                "GROQ API Key",
                type="password",
                help="Dapatkan API key dari https://console.groq.com/",
                placeholder="Masukkan GROQ API key"
            )
            
            if st.form_submit_button("Daftar", use_container_width=True):
                if signup_password != signup_password_confirm:
                    st.error("Password tidak cocok")
                elif not signup_groq_api:
                    st.error("GROQ API Key wajib diisi")
                elif signup_email and signup_password:
                    try:
                        if not validate_groq_api_key(signup_groq_api):
                            st.error("GROQ API Key tidak valid")
                            return

                        user = register_user(
                            st.session_state.db_manager,
                            signup_email,
                            signup_password,
                            signup_groq_api
                        )
                        if user:
                            st.success("✅ Berhasil membuat akun!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Email sudah terdaftar atau terjadi kesalahan")
                    except Exception as e:
                        st.error(f"Gagal mendaftar: {str(e)}")
                else:
                    st.warning("Silakan lengkapi semua field")

def main():
    """Main application function"""
    try:
        initialize_session_state()

        if 'token' not in st.session_state:
            render_login_page()
            return

        user = get_current_user()
        if not user:
            st.session_state.token = None
            st.rerun()
            return

        if not st.session_state.api_key:
            api_key = st.session_state.db_manager.get_api_key(user["id"])
            if api_key:
                st.session_state.api_key = api_key
                set_api_key(api_key)

        with st.sidebar:
            handle_sidebar()
        handle_main_area()

    except Exception as e:
        st.error(f"Terjadi kesalahan aplikasi: {str(e)}")

if __name__ == "__main__":
    main()