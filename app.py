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
import logging
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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

def initialize_session_state():
    """Initialize session state variables with error handling"""
    try:
        # Initialize DB manager if not exists
        if 'db_manager' not in st.session_state:
            st.session_state['db_manager'] = DatabaseManager()
        
        # Define all required session state variables
        default_state = {
            'messages': [],
            'documents': [],
            'embeddings': [],
            'index': None,
            'processed_files': [],
            'processed_urls': [],
            'clear_url': False,
            'selected_model': get_available_models()[0],
            'selected_embedding_model': "all-MiniLM-L6-v2",
            'api_key': "",
            'thinking': False,
            'user_input_key': 0,
            'api_key_verified': False,
            'initialization_complete': False,
            'error_count': 0
        }
        
        # Initialize or update session state
        for key, default_value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = default_value
        
        st.session_state.initialization_complete = True
        logger.info("Session state initialized successfully")
        return True
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        return False

def verify_api_key():
    """Verify stored API key"""
    try:
        api_key = st.session_state.get('api_key', '')
        if not api_key:
            return False
            
        if not st.session_state.get('api_key_verified'):
            set_api_key(api_key)
            st.session_state.api_key_verified = True
            logger.info("API key verified successfully")
            
        return True
    except Exception as e:
        logger.error(f"API key verification failed: {str(e)}")
        st.session_state.api_key_verified = False
        return False

def handle_query(prompt: str) -> None:
    """Handle user query with improved error handling"""
    if not prompt or not prompt.strip():
        st.error("Silakan masukkan pesan.")
        return

    try:
        # Verify API key first
        if not verify_api_key():
            st.error("Silakan periksa API key Anda di pengaturan sidebar.")
            return
            
        # Initialize messages if needed
        if 'messages' not in st.session_state:
            st.session_state.messages = []
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.status("💭 Thinking...") as status:
            try:
                # Prepare context if needed
                if st.session_state.get('index') is not None:
                    status.update(label="🔍 Searching relevant context...")
                    model = load_embedding_model(st.session_state.selected_embedding_model)
                    query_embedding = generate_embedding(prompt, model)
                    relevant_doc_indices = search_index(st.session_state.index, query_embedding)
                    context = "\n".join([st.session_state.documents[i][:1000] for i in relevant_doc_indices])
                    full_prompt = f"Berdasarkan konteks berikut:\n\n{context}\n\nJawab pertanyaan ini: {prompt}"
                else:
                    full_prompt = prompt
                
                # Generate response
                status.update(label="🤖 Generating response...")
                response = query_llm(full_prompt, st.session_state.selected_model)
                
                if response and not response.startswith("Error"):
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    status.update(label="✅ Done!", state="complete")
                    logger.info("Response generated successfully")
                else:
                    raise Exception(response)
                    
            except Exception as e:
                status.update(label="❌ Error!", state="error")
                error_msg = str(e)
                logger.error(f"Error generating response: {error_msg}")
                
                if "API key" in error_msg.lower():
                    st.error("API key tidak valid. Silakan periksa API key Anda.")
                    st.session_state.api_key_verified = False
                else:
                    st.error(f"Gagal menghasilkan respons: {error_msg}")
                
    except Exception as e:
        logger.error(f"Error in handle_query: {str(e)}")
        st.error("Terjadi kesalahan. Silakan coba lagi.")
    finally:
        # Clean up
        st.session_state.user_input_key = st.session_state.get('user_input_key', 0) + 1
        st.session_state.thinking = False

def render_chat_interface():
    """Render the chat interface with error handling"""
    try:
        st.title("🤖 Chatku AI")
        st.caption("Chatku AI dengan Retrieval Augmented Generation")
        
        # Initialize chat container
        chat_container = st.container()
        
        # Display messages
        with chat_container:
            messages = st.session_state.get('messages', [])
            for message in messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])
        
        # Chat input
        input_key = f"chat_input_{st.session_state.get('user_input_key', 0)}"
        if user_input := st.chat_input("Ketik pesan Anda di sini...", key=input_key):
            if not st.session_state.get('thinking', False):
                st.session_state.thinking = True
                handle_query(user_input)
                
    except Exception as e:
        logger.error(f"Error in chat interface: {str(e)}")
        st.error("Terjadi kesalahan pada tampilan chat. Silakan refresh halaman.")

def render_login_page():
    """Render login page with error handling"""
    try:
        st.title("🤖 Chatku AI")
        st.caption("Chatku AI Dengan Retrieval Augmented Generation")
        
        tab1, tab2 = st.tabs(["Login", "Daftar"])
        
        with tab1:
            with st.form("login_form"):
                st.subheader("Login")
                login_email = st.text_input("Email", placeholder="Masukkan email anda")
                login_password = st.text_input("Password", type="password", placeholder="Masukkan password anda")
                
                if st.form_submit_button("Login", use_container_width=True):
                    try:
                        if not login_email or not login_password:
                            st.warning("Silakan isi email dan password")
                            return
                            
                        result = login_user(st.session_state.db_manager, login_email, login_password)
                        if result and result.get("access_token"):
                            st.success("✅ Berhasil login!")
                            
                            # Set API key if available
                            if result["user"].get("groq_api_key"):
                                st.session_state.api_key = result["user"]["groq_api_key"]
                                try:
                                    set_api_key(result["user"]["groq_api_key"])
                                    st.session_state.api_key_verified = True
                                except Exception as e:
                                    logger.warning(f"Failed to verify API key on login: {str(e)}")
                                    st.session_state.api_key_verified = False
                                    
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Email atau password salah")
                    except Exception as e:
                        logger.error(f"Login error: {str(e)}")
                        st.error(f"Gagal login: {str(e)}")
        
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
                    try:
                        if not all([signup_email, signup_password, signup_password_confirm, signup_groq_api]):
                            st.warning("Silakan lengkapi semua field")
                            return
                            
                        if signup_password != signup_password_confirm:
                            st.error("Password tidak cocok")
                            return
                            
                        # Verify API key before registration
                        try:
                            set_api_key(signup_groq_api)
                        except Exception as e:
                            st.error(f"API key tidak valid: {str(e)}")
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
                        logger.error(f"Registration error: {str(e)}")
                        st.error(f"Gagal mendaftar: {str(e)}")
    except Exception as e:
        logger.error(f"Error in login page: {str(e)}")
        st.error("Terjadi kesalahan. Silakan refresh halaman.")

def handle_sidebar():
    """Handle sidebar with error handling"""
    try:
        user = get_current_user()
        if not user:
            logger.error("No authenticated user found")
            st.error("User not authenticated")
            return

        st.sidebar.write(f"👤 **{user['email']}**")
        st.sidebar.write("---")
        
        # API Settings
        with st.sidebar.expander("API Settings"):
            current_api_key = st.session_state.get('api_key', '')
            
            st.markdown("""
                ### Cara Mendapatkan GROQ API Key
                1. Kunjungi [console.groq.com](https://console.groq.com/)
                2. Buat akun atau login
                3. Di dashboard, klik "API Keys"
                4. Klik "Create API Key"
                5. Copy API Key yang dihasilkan
            """)
            
            if current_api_key:
                if st.session_state.get('api_key_verified', False):
                    st.success("✅ API Key terverifikasi")
                else:
                    st.warning("API Key belum terverifikasi")
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
                    try:
                        if not new_api_key:
                            st.error("API key tidak boleh kosong")
                        elif new_api_key == current_api_key and st.session_state.get('api_key_verified', False):
                            st.info("API key tidak berubah")
                        else:
                            # Verify new API key first
                            set_api_key(new_api_key)
                            
                            # If verification successful, save to database
                            success = st.session_state.db_manager.save_api_key(
                                user_id=user["id"],
                                api_key=new_api_key
                            )
                            
                            if success:
                                st.session_state.api_key = new_api_key
                                st.session_state.api_key_verified = True
                                st.success("✅ API key berhasil disimpan dan terverifikasi!")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Gagal menyimpan API key ke database")
                    except Exception as e:
                        logger.error(f"Error saving API key: {str(e)}")
                        st.session_state.api_key_verified = False
                        st.error(f"Gagal memverifikasi API key: {str(e)}")

        # Model Settings
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
                    try:
                        st.session_state.selected_model = selected_model
                        st.session_state.selected_embedding_model = selected_embedding_model
                        st.success("✅ Model berhasil diperbarui!")
                    except Exception as e:
                        logger.error(f"Error updating model settings: {str(e)}")
                        st.error("Gagal memperbarui pengaturan model")

        # Document Processing
        with st.sidebar.expander("Document Processing"):
            try:
                uploaded_file = st.file_uploader(
                    "Upload File (PDF/Word)",
                    type=['pdf', 'docx'],
                    help="Upload dokumen untuk diproses"
                )
                
                url = st.text_input(
                    "URL Input",
                    value="" if st.session_state.clear_url else st.session_state.get('url', ''),
                    placeholder="Atau masukkan URL untuk diproses"
                )
                
                if st.button("Proses File/URL"):
                    try:
                        if uploaded_file:
                            process_file(uploaded_file)
                        elif url:
                            process_url(url)
                        else:
                            st.warning("Silakan upload file atau masukkan URL")
                    except Exception as e:
                        logger.error(f"Error processing input: {str(e)}")
                        st.error(f"Gagal memproses: {str(e)}")

                st.write("---")
                
                if st.button("Generate Embeddings", use_container_width=True):
                    generate_embeddings()
                    
                if st.button("Buat Index", use_container_width=True):
                    create_search_index()

                # Display processed items
                if st.session_state.get('processed_files') or st.session_state.get('processed_urls'):
                    st.write("**Processed Items:**")
                    for file in st.session_state.get('processed_files', []):
                        st.write(f"📄 {file}")
                    for url in st.session_state.get('processed_urls', []):
                        st.write(f"🔗 {url}")

                st.write("---")
                if st.button("Hapus Semua Data", use_container_width=True):
                    clean_session_data()
                    
            except Exception as e:
                logger.error(f"Error in document processing: {str(e)}")
                st.error("Gagal menangani pemrosesan dokumen")

        # Logout button
        st.sidebar.write("---")
        if st.sidebar.button("Logout", use_container_width=True):
            try:
                logout_user()
                st.rerun()
            except Exception as e:
                logger.error(f"Error during logout: {str(e)}")
                st.error("Gagal logout")
                
    except Exception as e:
        logger.error(f"Error in sidebar: {str(e)}")
        st.error("Terjadi kesalahan pada sidebar")

def process_file(uploaded_file):
    """Process uploaded file with error handling"""
    try:
        with st.spinner("Memproses file..."):
            file_path = save_uploaded_file(uploaded_file)
            content = read_file(file_path)
            
            if not st.session_state.get('documents'):
                st.session_state.documents = []
            if not st.session_state.get('processed_files'):
                st.session_state.processed_files = []
                
            st.session_state.documents.append(content)
            st.session_state.processed_files.append(uploaded_file.name)
            st.success(f"✅ File '{uploaded_file.name}' berhasil diproses!")
            
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"Gagal memproses file: {str(e)}")

def process_url(url):
    """Process URL content with error handling"""
    try:
        with st.spinner("Memproses URL..."):
            content = read_url(url)
            
            if not st.session_state.get('documents'):
                st.session_state.documents = []
            if not st.session_state.get('processed_urls'):
                st.session_state.processed_urls = []
                
            st.session_state.documents.append(content)
            st.session_state.processed_urls.append(url)
            st.success(f"✅ URL '{url}' berhasil diproses!")
            
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        st.error(f"Gagal memproses URL: {str(e)}")

def generate_embeddings():
    """Generate embeddings with error handling"""
    if not st.session_state.get('documents'):
        st.warning("Tidak ada dokumen yang perlu diproses")
        return
        
    try:
        with st.spinner("Menghasilkan embeddings..."):
            model = load_embedding_model(st.session_state.selected_embedding_model)
            st.session_state.embeddings = []
            
            total_docs = len(st.session_state.documents)
            progress_bar = st.progress(0)
            
            for i, doc in enumerate(st.session_state.documents):
                embedding = generate_embedding(doc, model)
                st.session_state.embeddings.append(embedding)
                progress = (i + 1) / total_docs
                progress_bar.progress(progress)
                
            st.success(f"✅ Berhasil menghasilkan embeddings untuk {total_docs} dokumen!")
            
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        st.error(f"Gagal menghasilkan embeddings: {str(e)}")

def create_search_index():
    """Create search index with error handling"""
    if not st.session_state.get('embeddings'):
        st.warning("Harap generate embeddings terlebih dahulu")
        return
        
    try:
        with st.spinner("Membuat index pencarian..."):
            st.session_state.index = create_index(st.session_state.embeddings)
            st.success("✅ Index pencarian berhasil dibuat!")
    except Exception as e:
        logger.error(f"Error creating search index: {str(e)}")
        st.error(f"Gagal membuat index: {str(e)}")

def clean_session_data():
    """Clean all session data with error handling"""
    try:
        with st.spinner("Membersihkan data..."):
            session_vars = {
                'messages': [],
                'documents': [],
                'embeddings': [],
                'index': None,
                'processed_files': [],
                'processed_urls': [],
                'clear_url': True,
                'thinking': False,
                'api_key_verified': st.session_state.get('api_key_verified', False),
                'api_key': st.session_state.get('api_key', '')
            }
            
            for key, value in session_vars.items():
                st.session_state[key] = value
            
            st.success("✅ Semua data berhasil dibersihkan!")
            st.rerun()
            
    except Exception as e:
        logger.error(f"Error cleaning session data: {str(e)}")
        st.error(f"Gagal membersihkan data: {str(e)}")

def handle_errors():
    """Handle application errors and reset if needed"""
    error_count = st.session_state.get('error_count', 0)
    if error_count > 3:  # Reset after multiple errors
        try:
            clean_session_data()
            st.session_state.error_count = 0
            st.success("Aplikasi telah direset karena terlalu banyak error")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            logger.error(f"Failed to reset application: {str(e)}")
            st.error("Gagal mereset aplikasi. Silakan refresh halaman secara manual.")

def main():
    """Main application entry point with comprehensive error handling"""
    try:
        # Initialize session state
        if not st.session_state.get('initialization_complete'):
            if not initialize_session_state():
                st.error("Gagal menginisialisasi aplikasi. Silakan refresh halaman.")
                return

        # Error handling setup
        handle_errors()

        # Check authentication
        if 'token' not in st.session_state:
            logger.info("No token found, rendering login page")
            render_login_page()
            return

        # Verify user session
        user = get_current_user()
        if not user:
            logger.warning("Invalid or expired session")
            st.session_state.token = None
            st.rerun()
            return

        # Render main application
        try:
            # Create columns for layout
            col1, col2 = st.columns([1, 3])
            
            # Render sidebar in first column
            with col1:
                handle_sidebar()
            
            # Render chat interface in second column
            with col2:
                render_chat_interface()
                
        except Exception as e:
            logger.error(f"Error in main interface: {str(e)}")
            st.error("Terjadi kesalahan pada tampilan utama")
            st.session_state.error_count = st.session_state.get('error_count', 0) + 1
            
    except Exception as e:
        logger.error(f"Critical error in main: {str(e)}")
        st.error("Terjadi kesalahan sistem. Silakan refresh halaman.")
        st.session_state.error_count = st.session_state.get('error_count', 0) + 1

if __name__ == "__main__":
    main()