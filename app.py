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
    query_llm, set_api_key, get_available_models,
    read_file, read_url, save_uploaded_file,
    validate_api_key
)
from db_utils import DatabaseManager
from auth_utils import (
    login_user, register_user, get_current_user, logout_user
)
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize essential session state"""
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
    
    # Initialize basic required variables
    default_values = {
        'chat_history': [],
        'documents': [],
        'selected_model': get_available_models()[0],
        'api_key': None,
        'is_processing': False,
        'clear_url': False,
        'url': '',
        'processed_files': [],
        'processed_urls': []
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def handle_chat():
    """Handle chat with simplified flow"""
    st.title("Chatku AI")
    st.caption("Chatku AI dengan Retrieval Augmented Generation")
    
    # Display chat history
    for role, message in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(message)
            
    # Chat input
    if prompt := st.chat_input("Ketik pesan Anda di sini..."):
        if not st.session_state.api_key:
            st.error("Silakan masukkan GROQ API key yang valid")
            return
            
        if st.session_state.is_processing:
            return
            
        try:
            st.session_state.is_processing = True
            
            # Display user message
            st.chat_message("user").write(prompt)
            st.session_state.chat_history.append(("user", prompt))
            
            # Get response
            with st.spinner("Menghasilkan respons..."):
                response = query_llm(prompt, st.session_state.selected_model)
                
                if response:
                    # Display assistant response
                    st.chat_message("assistant").write(response)
                    st.session_state.chat_history.append(("assistant", response))
                else:
                    st.error("Gagal mendapatkan respons. Silakan coba lagi.")
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan: {str(e)}")
            
        finally:
            st.session_state.is_processing = False
            st.rerun()

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

    # Model Selection
    with st.sidebar.expander("Model Settings", expanded=False):
        available_models = get_available_models()
        selected_model = st.selectbox(
            "Pilih LLM Model",
            available_models,
            index=available_models.index(st.session_state.selected_model)
        )
        if selected_model != st.session_state.selected_model:
            st.session_state.selected_model = selected_model
            st.success("Model berhasil diperbarui!")

    st.sidebar.write("---")
    if st.sidebar.button("Logout", use_container_width=True):
        logout_user()
        st.rerun()

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
                        if not validate_api_key(signup_groq_api):
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
        handle_chat()

    except Exception as e:
        st.error(f"Terjadi kesalahan aplikasi: {str(e)}")

if __name__ == "__main__":
    main()