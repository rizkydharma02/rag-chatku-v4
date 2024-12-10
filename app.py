import streamlit as st
import time
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    """Initialize session state variables"""
    try:
        if 'db_manager' not in st.session_state:
            st.session_state.db_manager = DatabaseManager()
        
        # Define default session state variables
        session_vars = {
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
            'api_key_verified': False
        }
        
        # Initialize missing variables
        for key, value in session_vars.items():
            if key not in st.session_state:
                st.session_state[key] = value
                
    except Exception as e:
        logger.error(f"Error initializing session state: {str(e)}")
        st.error("Failed to initialize application state")

def verify_api_key():
    """Verify stored API key"""
    try:
        if not st.session_state.api_key:
            return False
            
        if not st.session_state.api_key_verified:
            set_api_key(st.session_state.api_key)
            st.session_state.api_key_verified = True
            
        return True
        
    except Exception as e:
        logger.error(f"API key verification failed: {str(e)}")
        st.session_state.api_key_verified = False
        return False

def handle_query(prompt: str) -> None:
    """Handle user query with improved error handling"""
    try:
        if not prompt or not prompt.strip():
            st.error("Please enter a message.")
            return

        # Verify API key first
        if not verify_api_key():
            st.error("Please check your API key in the sidebar settings.")
            return
            
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Use status for better progress indication
        with st.status("💭 Thinking...") as status:
            try:
                # Prepare context if needed
                if st.session_state.index is not None:
                    status.update(label="🔍 Searching relevant context...")
                    model = load_embedding_model(st.session_state.selected_embedding_model)
                    query_embedding = generate_embedding(prompt, model)
                    relevant_doc_indices = search_index(st.session_state.index, query_embedding)
                    context = "\n".join([st.session_state.documents[i][:1000] for i in relevant_doc_indices])
                    full_prompt = f"Berdasarkan konteks berikut:\n\n{context}\n\nJawab pertanyaan ini: {prompt}"
                else:
                    full_prompt = prompt
                
                status.update(label="🤖 Generating response...")
                response = query_llm(full_prompt, st.session_state.selected_model)
                
                if response.startswith("An error occurred"):
                    raise Exception(response)
                    
                # Add assistant response
                st.session_state.messages.append({"role": "assistant", "content": response})
                status.update(label="✅ Done!", state="complete")
                
            except Exception as e:
                status.update(label="❌ Error!", state="error")
                raise e
                
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error in handle_query: {error_msg}")
        if "API key" in error_msg.lower():
            st.error("Please check your API key in the sidebar settings.")
        else:
            st.error(f"An error occurred: {error_msg}")
            
    finally:
        st.rerun()

def chat_interface():
    """Render the chat interface"""
    st.title("🤖 Chatku AI")
    st.caption("Chatku AI dengan Retrieval Augmented Generation")
    
    # Display messages
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
    
    # Chat input
    if user_input := st.chat_input("Ketik pesan Anda di sini...", key=f"chat_input_{st.session_state.user_input_key}"):
        if not st.session_state.thinking:
            st.session_state.thinking = True
            handle_query(user_input)
            st.session_state.thinking = False

def handle_sidebar():
    """Handle sidebar UI and functionality"""
    user = get_current_user()
    if not user:
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
        API Key biasanya dimulai dengan 'gsk_'
        """)
        
        if current_api_key:
            if st.session_state.api_key_verified:
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
                    elif new_api_key == current_api_key and st.session_state.api_key_verified:
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
                st.session_state.selected_model = selected_model
                st.session_state.selected_embedding_model = selected_embedding_model
                st.success("✅ Model berhasil diperbarui!")

    # Document Processing
    with st.sidebar.expander("Document Processing"):
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
    """Process uploaded file"""
    try:
        with st.spinner("Memproses file..."):
            file_path = save_uploaded_file(uploaded_file)
            content = read_file(file_path)
            st.session_state.documents.append(content)
            st.session_state.processed_files.append(uploaded_file.name)
            st.success(f"✅ File '{uploaded_file.name}' berhasil diproses!")
    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        st.error(f"Gagal memproses file: {str(e)}")

def process_url(url):
    """Process URL content"""
    try:
        with st.spinner("Memproses URL..."):
            content = read_url(url)
            st.session_state.documents.append(content)
            st.session_state.processed_urls.append(url)
            st.success(f"✅ URL '{url}' berhasil diproses!")
    except Exception as e:
        logger.error(f"Error processing URL: {str(e)}")
        st.error(f"Gagal memproses URL: {str(e)}")

def generate_embeddings():
    """Generate embeddings for documents"""
    if not st.session_state.documents:
        st.warning("Tidak ada dokumen yang perlu diproses")
        return
        
    try:
        with st.spinner("Menghasilkan embeddings..."):
            model = load_embedding_model(st.session_state.selected_embedding_model)
            st.session_state.embeddings = []
            
            progress_bar = st.progress(0)
            for i, doc in enumerate(st.session_state.documents):
                embedding = generate_embedding(doc, model)
                st.session_state.embeddings.append(embedding)
                progress_bar.progress((i + 1) / len(st.session_state.documents))
                
            st.success(f"✅ Berhasil menghasilkan embeddings untuk {len(st.session_state.embeddings)} dokumen!")
    except Exception as e:
        logger.error(f"Error generating embeddings: {str(e)}")
        st.error(f"Gagal menghasilkan embeddings: {str(e)}")

def create_search_index():
    """Create search index from embeddings"""
    if not st.session_state.embeddings:
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
    """Clean all session data"""
    try:
        with st.spinner("Membersihkan data..."):
            # Reset session variables
            session_vars = {
                'messages': [],
                'documents': [],
                'embeddings': [],
                'index': None,
                'processed_files': [],
                'processed_urls': [],
                'clear_url': True,
                'thinking': False
            }
            
            for key, value in session_vars.items():
                if key in st.session_state:
                    st.session_state[key] = value
            
            st.success("✅ Semua data berhasil dibersihkan!")
            st.rerun()
            
    except Exception as e:
        logger.error(f"Error cleaning session data: {str(e)}")
        st.error(f"Gagal membersihkan data: {str(e)}")

def main():
    """Main application entry point"""
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

        # Render main application
        col1, col2 = st.columns([1, 3])
        
        with col1:
            handle_sidebar()
        
        with col2:
            chat_interface()

    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        st.error("An unexpected error occurred. Please try refreshing the page.")

if __name__ == "__main__":
    main()