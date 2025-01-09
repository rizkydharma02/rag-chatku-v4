import streamlit as st
import time
from datetime import datetime, timedelta

st.set_page_config(
    page_title="Chatku AI",
    page_icon="🤖",
    layout="wide"
)

import os
from utils import (
    load_embedding_model, read_file, read_url, generate_embedding,
    save_uploaded_file, create_index, search_index,
    set_api_key, get_available_models, validate_api_key_format
)
from db_utils import DatabaseManager
from auth_utils import (
    login_user, register_user, get_current_user, logout_user,
    request_password_reset, reset_password  # Add these new imports
)
from chat_manager import ChatManager, initialize_chat_state
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def initialize_state():
    """Initialize session state variables"""
    if 'reset_step' not in st.session_state:
        st.session_state.reset_step = "request"
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()

def render_login_page():
    initialize_state()

    header = st.container()
    with header:
        col1, col2 = st.columns([1, 18])
        with col1:
             st.image("./img/logo-revou.jpg", width=70)
        with col2:
            st.subheader("PT. Revolusi Cita Edukasi")

    st.header("🤖 Chatku AI")
    st.caption("Chatku AI Dengan Retrieval Augmented Generation")
    
    tab1, tab2, tab3 = st.tabs(["Login", "Daftar", "Lupa Password"])
    
    # Login tab
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
                            st.success("✅ Berhasil login!")
                            if result["user"].get("groq_api_key"):
                                st.session_state.api_key = result["user"]["groq_api_key"]
                                set_api_key(result["user"]["groq_api_key"])
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Email atau password salah")
                    except Exception as e:
                        st.error(f"Gagal login: {str(e)}")
                else:
                    st.warning("Silakan isi email dan password")
    
    # Register tab
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

            st.write("**Persyaratan Password:**")
            st.write("- Minimal 6 karakter")
            st.write("- Minimal 1 huruf besar")
            st.write("- Minimal 1 huruf kecil")
            st.write("- Minimal 1 angka")
            
            if st.form_submit_button("Daftar", use_container_width=True):
                if signup_password != signup_password_confirm:
                    st.error("Password tidak cocok")
                elif not signup_groq_api:
                    st.error("GROQ API Key wajib diisi")
                elif signup_email and signup_password:
                    try:
                        if not validate_api_key_format(signup_groq_api):
                            st.error("Format GROQ API Key tidak valid")
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
    
    # Lupa Password tab
    with tab3:
        st.subheader("Reset Password")
        
        if st.session_state.reset_step == "request":
            # Step 1: Request password reset
            with st.form("forgot_password_form"):
                reset_email = st.text_input("Email", placeholder="Masukkan email anda")
                
                if st.form_submit_button("Reset Password", use_container_width=True):
                    if reset_email:
                        if request_password_reset(st.session_state.db_manager, reset_email):
                            st.session_state.reset_step = "reset"
                            st.success("✅ Link reset password telah dikirim!")
                            # For demo purposes, we'll get it from the database
                            user = st.session_state.db_manager.get_user_by_email(reset_email)
                            if user and user.get("reset_token"):
                                st.info("Untuk demo, token akan ditampilkan di sini:")
                                st.code(user["reset_token"])
                                st.rerun()
                        else:
                            st.error("Email tidak ditemukan")
                    else:
                        st.warning("Silakan masukkan email")
        
        else:
            # Step 2: Reset password with token
            with st.form("reset_password_form"):
                reset_token = st.text_input("Token Reset", placeholder="Masukkan token reset")
                new_password = st.text_input("Password Baru", type="password", placeholder="Masukkan password baru")
                confirm_password = st.text_input("Konfirmasi Password", type="password", placeholder="Konfirmasi password baru")
                
                st.write("**Persyaratan Password Baru:**")
                st.write("- Minimal 6 karakter")
                st.write("- Minimal 1 huruf besar")
                st.write("- Minimal 1 huruf kecil")
                st.write("- Minimal 1 angka")
                
                col1, col2 = st.columns([1, 4])
                with col1:
                    if st.form_submit_button("Kembali"):
                        st.session_state.reset_step = "request"
                        st.rerun()
                with col2:
                    if st.form_submit_button("Reset Password", use_container_width=True):
                        if not reset_token or not new_password or not confirm_password:
                            st.warning("Silakan lengkapi semua field")
                        elif new_password != confirm_password:
                            st.error("Password tidak cocok")
                        else:
                            if reset_password(st.session_state.db_manager, reset_token, new_password):
                                st.success("✅ Password berhasil direset!")
                                st.session_state.reset_step = "request"
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Token tidak valid atau sudah kadaluarsa")

def handle_sidebar():
    user = get_current_user()
    if not user:
        st.error("User not authenticated")
        return

    st.sidebar.write(f"👤 **{user['email']}**")
    st.sidebar.write("---")
    
    with st.sidebar.expander("API Settings"):
        current_api_key = st.session_state.get('api_key', '')
        
        st.markdown("""
            ### Cara Mendapatkan GROQ API Key
            1. Kunjungi [console.groq.com](https://console.groq.com/)
            2. Buat akun atau login jika sudah punya akun
            3. Di dashboard, klik "API Keys"
            4. Klik "Create API Key"
            5. Copy API Key yang dihasilkan
        """)
        
        if current_api_key:
            st.success("API Key sudah terpasang")
        else:
            st.warning("API Key belum diatur")
            
        with st.form("api_key_form"):
            new_api_key = st.text_input(
                "GROQ API Key",
                value=current_api_key,
                type="password",
                placeholder="Masukkan GROQ API Key Anda"
            )
            
            if st.form_submit_button("Simpan API Key"):
                if not new_api_key:
                    st.error("API key tidak boleh kosong")
                elif new_api_key == current_api_key:
                    st.info("API key tidak berubah")
                else:
                    try:
                        if validate_api_key_format(new_api_key):
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
                        else:
                            st.error("Format API key tidak valid")
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")

    with st.sidebar.expander("Model Settings"):
        with st.form("model_settings_form"):
            available_models = get_available_models()
            selected_model = st.selectbox(
                "Pilih LLM Model",
                available_models,
                index=available_models.index(st.session_state.selected_model)
                    if st.session_state.selected_model in available_models else 0
            )

            embedding_models = ["all-MiniLM-L6-v2"]
            selected_embedding_model = st.selectbox(
                "Pilih Embedding Model",
                embedding_models,
                index=0
            )
            
            if st.form_submit_button("Simpan Model"):
                st.session_state.selected_model = selected_model
                st.session_state.selected_embedding_model = selected_embedding_model
                st.success("Model berhasil diperbarui!")

    with st.sidebar.expander("Document Processing"):
        uploaded_file = st.file_uploader("Upload File (PDF/Word)", type=['pdf', 'docx'])
        url = st.text_input("URL Input", placeholder="Atau masukkan URL")
        
        if st.button("Proses File/URL"):
            try:
                if uploaded_file:
                    with st.spinner("Memproses file..."):
                        file_path = save_uploaded_file(uploaded_file)
                        content = read_file(file_path)
                        st.session_state.documents.append(content)
                        st.session_state.processed_files.append(uploaded_file.name)
                        st.success(f"File '{uploaded_file.name}' berhasil diproses!")
                elif url:
                    with st.spinner("Memproses URL..."):
                        content = read_url(url)
                        st.session_state.documents.append(content)
                        st.session_state.processed_urls.append(url)
                        st.success(f"URL '{url}' berhasil diproses!")
                else:
                    st.warning("Silakan upload file atau masukkan URL")
            except Exception as e:
                st.error(f"Gagal memproses: {str(e)}")

        if st.button("Generate Embeddings"):
            if not st.session_state.documents:
                st.warning("Tidak ada dokumen yang perlu diproses")
            else:
                try:
                    with st.spinner("Menghasilkan embeddings..."):
                        model = load_embedding_model(st.session_state.selected_embedding_model)
                        st.session_state.embeddings = []
                        for doc in st.session_state.documents:
                            embedding = generate_embedding(doc, model)
                            st.session_state.embeddings.append(embedding)
                        st.success(f"Berhasil menghasilkan embeddings!")
                except Exception as e:
                    st.error(f"Gagal generate embeddings: {str(e)}")
            
        if st.button("Buat Index"):
            if not st.session_state.embeddings:
                st.warning("Harap generate embeddings terlebih dahulu")
            else:
                try:
                    with st.spinner("Membuat index..."):
                        st.session_state.index = create_index(st.session_state.embeddings)
                        st.success("Index berhasil dibuat!")
                except Exception as e:
                    st.error(f"Gagal membuat index: {str(e)}")

        if st.session_state.processed_files or st.session_state.processed_urls:
            st.write("**Processed Items:**")
            for file in st.session_state.processed_files:
                st.write(f"📄 {file}")
            for url in st.session_state.processed_urls:
                st.write(f"🔗 {url}")

        if st.button("Hapus Semua Data"):
            try:
                # Reset document processing data
                st.session_state.documents = []
                st.session_state.embeddings = []
                st.session_state.index = None
                st.session_state.processed_files = []
                st.session_state.processed_urls = []
                
                # Reset chat history
                if 'chat_messages' in st.session_state:
                    st.session_state.chat_messages = []
                
                st.success("Semua data dan riwayat chat berhasil dihapus!")
                st.rerun()
            except Exception as e:
                st.error(f"Gagal menghapus data: {str(e)}")

    st.sidebar.write("---")
    if st.sidebar.button("Logout"):
        logout_user()
        st.rerun()

def main():
    if 'db_manager' not in st.session_state:
        st.session_state.db_manager = DatabaseManager()
        
    initialize_chat_state()

    if 'token' not in st.session_state:
        render_login_page()
        return

    user = get_current_user()
    if not user:
        st.session_state.token = None
        st.rerun()
        return

    with st.sidebar:
        handle_sidebar()
        
    st.header("🤖 Chatku AI")
    st.caption("Chatku AI dengan Retrieval Augmented Generation")
    
    chat_manager = ChatManager()
    chat_manager.handle_chat_interface()

if __name__ == "__main__":
    main()