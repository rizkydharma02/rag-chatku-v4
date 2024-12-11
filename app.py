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
    
    default_values = {
        'documents': [],
        'embeddings': [],
        'chat_history': [],
        'conversation_history': [],
        'index': None,
        'processed_files': [],
        'processed_urls': [],
        'clear_url': False,
        'url': '',
        'selected_model': get_available_models()[0],
        'selected_embedding_model': "all-MiniLM-L6-v2",
        'api_key': None,
        'current_query': "",
        'is_processing': False,
        'error_message': None,
        'user_id': None,
        'email': None,
        'token': None,
        'login_time': None,
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

    # Load API key if user is logged in
    if 'user_id' in st.session_state and not st.session_state.api_key:
        stored_api_key = st.session_state.db_manager.get_api_key(st.session_state.user_id)
        if stored_api_key:
            st.session_state.api_key = stored_api_key
            set_api_key(stored_api_key)

def handle_query(query):
    """Handle chat query with improved response handling"""
    try:
        if not st.session_state.api_key:
            st.error("Silakan masukkan GROQ API key yang valid")
            return False

        # Set API key and validate
        if not set_api_key(st.session_state.api_key):
            st.error("API key tidak valid. Silakan periksa kembali.")
            return False

        # Add user message to history first
        st.session_state.chat_history.append(("user", query))
        st.session_state.conversation_history.append({
            "role": "user", 
            "content": query
        })

        # Prepare context for response
        prompt = query
        if st.session_state.index is not None:
            try:
                model = load_embedding_model(st.session_state.selected_embedding_model)
                if model:
                    query_embedding = generate_embedding(query, model)
                    if query_embedding is not None:
                        relevant_doc_indices = search_index(st.session_state.index, query_embedding)
                        context = "\n".join([st.session_state.documents[i][:1000] for i in relevant_doc_indices])
                        prompt = f"Berdasarkan konteks berikut:\n\n{context}\n\nJawab pertanyaan ini: {query}"
            except Exception as e:
                print(f"Error preparing context: {str(e)}")

        # Get response with retry logic
        response = None
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = query_llm(prompt, st.session_state.selected_model)
                if response and not response.startswith("Error:"):
                    break
                time.sleep(1)
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == max_retries - 1:
                    st.error("Gagal mendapatkan respons setelah beberapa percobaan.")
                    return False

        # Process successful response
        if response and not response.startswith("Error:"):
            # Update conversation history
            st.session_state.chat_history.append(("assistant", response))
            st.session_state.conversation_history.append({
                "role": "assistant",
                "content": response
            })
            return True
        else:
            st.error("Gagal mendapatkan respons yang valid. Silakan coba lagi.")
            return False

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        return False

def handle_main_area():
    """Handle main chat area with improved response handling"""
    st.title("Chatku AI")
    st.caption("Chatku AI dengan Retrieval Augmented Generation")
    
    chat_placeholder = st.empty()
    
    # Display chat history
    with chat_placeholder.container():
        for role, msg in st.session_state.chat_history:
            with st.chat_message(role):
                st.markdown(msg)

    # Chat input
    if prompt := st.chat_input("Ketik pesan Anda di sini..."):
        if not st.session_state.get('is_processing', False):
            try:
                st.session_state.is_processing = True
                
                # Display user message immediately
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                # Handle query with spinner
                with st.spinner("Menghasilkan respons..."):
                    success = handle_query(prompt)
                    if success:
                        # Force refresh to show new response
                        st.rerun()
                    
            finally:
                st.session_state.is_processing = False

def handle_sidebar():
    """Handle sidebar with improved API management"""
    user = get_current_user()
    if not user:
        st.error("User not authenticated")
        return

    st.sidebar.write(f"👤 **{user['email']}**")
    st.sidebar.write("---")
    
    # API Settings
    with st.sidebar.expander("API Settings", expanded=False):
        st.markdown("""
        ### Cara Mendapatkan GROQ API Key
        1. Kunjungi [console.groq.com](https://console.groq.com/)
        2. Buat akun atau login
        3. Di dashboard, klik "API Keys"
        4. Klik "Create API Key"
        5. Copy API Key yang dihasilkan
        
        API Key biasanya dimulai dengan 'gsk_'
        """)
        
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

    # Model Settings
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
            value="" if st.session_state.clear_url else st.session_state.url,
            placeholder="Atau masukkan URL"
        )
        
        if st.button("Proses File/URL"):
            with st.spinner("Memproses..."):
                try:
                    if uploaded_file:
                        process_file(uploaded_file)
                    elif url:
                        st.session_state.url = url
                        process_url(url)
                    else:
                        st.warning("Silakan upload file atau masukkan URL")
                except Exception as e:
                    st.error(f"Gagal memproses: {str(e)}")

        st.write("---")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Generate Embeddings", use_container_width=True):
                generate_embeddings()
        with col2:
            if st.button("Buat Index", use_container_width=True):
                create_search_index()

        # Display processed items
        if st.session_state.processed_files or st.session_state.processed_urls:
            st.write("**Processed Items:**")
            for file in st.session_state.processed_files:
                st.write(f"📄 {file}")
            for url in st.session_state.processed_urls:
                st.write(f"🔗 {url}")

        st.write("---")
        if st.button("Hapus Semua Data", use_container_width=True):
            clean_session_data()

    # Logout button
    st.sidebar.write("---")
    if st.sidebar.button("Logout", use_container_width=True):
        logout_user()
        st.rerun()

def process_file(uploaded_file):
    """Process uploaded file with improved error handling"""
    try:
        with st.spinner("Memproses file..."):
            file_path = save_uploaded_file(uploaded_file)
            if not file_path:
                st.error("Gagal menyimpan file")
                return

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
    """Process URL with improved error handling"""
    try:
        with st.spinner("Memproses URL..."):
            content = read_url(url)
            if content:
                st.session_state.documents.append(content)
                st.session_state.processed_urls.append(url)
                st.success(f"URL '{url}' berhasil diproses!")
            else:
                st.error("Gagal membaca konten URL")
                
    except Exception as e:
        st.error(f"Gagal memproses URL: {str(e)}")

def generate_embeddings():
    """Generate embeddings with improved progress tracking"""
    if not st.session_state.documents:
        st.warning("Tidak ada dokumen yang perlu diproses")
        return
        
    try:
        with st.spinner("Menghasilkan embeddings..."):
            model = load_embedding_model(st.session_state.selected_embedding_model)
            if not model:
                st.error("Gagal memuat model embedding")
                return

            st.session_state.embeddings = []
            progress_bar = st.progress(0)
            
            for i, doc in enumerate(st.session_state.documents):
                embedding = generate_embedding(doc, model)
                if embedding is not None:
                    st.session_state.embeddings.append(embedding)
                progress = (i + 1) / len(st.session_state.documents)
                progress_bar.progress(progress)
            
            if st.session_state.embeddings:
                st.success(f"Berhasil menghasilkan embeddings untuk {len(st.session_state.embeddings)} dokumen")
            else:
                st.error("Gagal menghasilkan embeddings")
            
    except Exception as e:
        st.error(f"Gagal menghasilkan embeddings: {str(e)}")

def create_search_index():
    """Create search index with improved error handling"""
    if not st.session_state.embeddings:
        st.warning("Harap generate embeddings terlebih dahulu")
        return
        
    try:
        with st.spinner("Membuat index pencarian..."):
            st.session_state.index = create_index(st.session_state.embeddings)
            
            if st.session_state.index is not None:
                st.success("Index pencarian berhasil dibuat!")
            else:
                st.error("Gagal membuat index")
                
    except Exception as e:
        st.error(f"Gagal membuat index: {str(e)}")

def clean_session_data():
    """Clean session data with improved state management"""
    try:
        with st.spinner("Membersihkan data..."):
            # Reset all values to default
            st.session_state.documents = []
            st.session_state.embeddings = []
            st.session_state.chat_history = []
            st.session_state.conversation_history = []
            st.session_state.index = None
            st.session_state.processed_files = []
            st.session_state.processed_urls = []
            st.session_state.clear_url = True
            st.session_state.url = ""
            st.session_state.is_processing = False
            st.session_state.error_message = None
            
            st.success("Semua data berhasil dibersihkan!")
            st.rerun()
            
    except Exception as e:
        st.error(f"Gagal membersihkan data: {str(e)}")

def render_login_page():
    """Render login page with improved validation"""
    st.title("🤖 Chatku AI")
    st.caption("Chatku AI Dengan Retrieval Augmented Generation")
    
    tab1, tab2 = st.tabs(["Login", "Daftar"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Login")
            login_email = st.text_input(
                "Email",
                placeholder="Masukkan email anda"
            )
            login_password = st.text_input(
                "Password",
                type="password",
                placeholder="Masukkan password anda"
            )
            
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
            signup_email = st.text_input(
                "Email",
                key="signup_email",
                placeholder="Masukkan email anda"
            )
            signup_password = st.text_input(
                "Password",
                type="password",
                key="signup_pass",
                placeholder="Buat password anda"
            )
            signup_password_confirm = st.text_input(
                "Konfirmasi Password",
                type="password",
                placeholder="Masukkan ulang password"
            )
            signup_groq_api = st.text_input(
                "GROQ API Key",
                type="password",
                help="Dapatkan API key dari https://console.groq.com/",
                placeholder="Masukkan GROQ API key"
            )

            st.markdown("""
            **Persyaratan Password:**
            - Minimal 6 karakter
            - Minimal 1 huruf besar
            - Minimal 1 huruf kecil
            - Minimal 1 angka
            """)
            
            if st.form_submit_button("Daftar", use_container_width=True):
                if signup_password != signup_password_confirm:
                    st.error("Password tidak cocok")
                elif not signup_groq_api:
                    st.error("GROQ API Key wajib diisi")
                elif signup_email and signup_password:
                    try:
                        # Validate API key before registration
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
    """Main application function with improved error handling"""
    try:
        initialize_session_state()

        # Check authentication
        if 'token' not in st.session_state:
            render_login_page()
            return

        # Verify user session
        user = get_current_user()
        if not user:
            st.session_state.token = None
            st.rerun()
            return

        # Load API key if needed
        if not st.session_state.api_key:
            api_key = st.session_state.db_manager.get_api_key(user["id"])
            if api_key:
                st.session_state.api_key = api_key
                set_api_key(api_key)

        # Render main application
        with st.sidebar:
            handle_sidebar()
        handle_main_area()

    except Exception as e:
        st.error(f"Terjadi kesalahan aplikasi: {str(e)}")

if __name__ == "__main__":
    main()