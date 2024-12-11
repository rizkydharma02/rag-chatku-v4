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
    """Initialize session state with improved state management"""
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
        'selected_model': get_available_models()[0],
        'selected_embedding_model': "all-MiniLM-L6-v2",
        'api_key': "",
        'current_query': "",
        'messages': [],  # New addition for better chat history tracking
        'last_response': None,  # Track last response for better UI updates
        'error_message': None  # Track error messages
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def render_login_page():
    """Render login page with improved UI and validation"""
    st.title("🤖 Chatku AI")
    st.caption("Chatku AI Dengan Retrieval Augmented Generation")
    
    # Create tabs with consistent width
    tab_container = st.container()
    with tab_container:
        tab1, tab2 = st.tabs(["Login", "Daftar"])
        
        with tab1:
            with st.form("login_form"):
                st.subheader("Login")
                login_col1, login_col2 = st.columns([3, 1])
                with login_col1:
                    login_email = st.text_input(
                        "Email",
                        placeholder="Masukkan email anda",
                        key="login_email"
                    )
                    login_password = st.text_input(
                        "Password",
                        type="password",
                        placeholder="Masukkan password anda",
                        key="login_password"
                    )
                
                if st.form_submit_button("Login", use_container_width=True):
                    if login_email and login_password:
                        with st.spinner("Memproses..."):
                            try:
                                result = login_user(
                                    st.session_state.db_manager,
                                    login_email,
                                    login_password
                                )
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
        
        with tab2:
            with st.form("register_form"):
                st.subheader("Daftar Akun Baru")
                reg_col1, reg_col2 = st.columns([3, 1])
                with reg_col1:
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
                    with st.spinner("Memproses..."):
                        if signup_password != signup_password_confirm:
                            st.error("Password tidak cocok")
                        elif not signup_groq_api:
                            st.error("GROQ API Key wajib diisi")
                        elif signup_email and signup_password:
                            try:
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

def display_chat_history():
    """Display chat history with improved UI and performance"""
    chat_container = st.container()
    
    with chat_container:
        # Create a scrollable container for chat history
        chat_area = st.empty()
        with chat_area.container():
            for i, (role, message) in enumerate(st.session_state.chat_history):
                # Use columns for better message layout
                msg_container = st.container()
                with msg_container:
                    cols = st.columns([0.8, 0.2])
                    
                    with cols[0]:
                        if role == "user":
                            st.markdown(f"🧑‍💻 **You**")
                            st.markdown(message)
                        else:
                            st.markdown(f"🤖 **Assistant**")
                            st.markdown(message)
                    
                    with cols[1]:
                        if i == len(st.session_state.chat_history) - 1:
                            st.caption(datetime.now().strftime("%H:%M"))
                    
                    st.markdown("---")
            
            # Add empty space for better scrolling
            st.empty()

def handle_main_area():
    """Handle main chat area with improved UI and state management"""
    st.title("Chatku AI")
    st.caption("Chatku AI dengan Retrieval Augmented Generation")
    
    # Main chat container
    main_container = st.container()
    
    with main_container:
        # Chat history display
        chat_placeholder = st.empty()
        with chat_placeholder.container():
            display_chat_history()
        
        # Chat input form
        with st.form(key="chat_form", clear_on_submit=True):
            cols = st.columns([0.8, 0.2])
            
            with cols[0]:
                query = st.text_input(
                    "Message",
                    placeholder="Ketik pesan Anda di sini...",
                    key="input_message",
                    label_visibility="collapsed"
                )
            
            with cols[1]:
                submit_button = st.form_submit_button(
                    "Kirim",
                    use_container_width=True
                )
            
            if submit_button and query:
                # Update state before processing
                st.session_state.messages.append({
                    "role": "user",
                    "content": query
                })
                
                # Process query with error handling
                with st.spinner("Memproses..."):
                    try:
                        handle_query(query)
                        st.session_state.current_query = ""
                        st.rerun()
                    except Exception as e:
                        st.error(f"Terjadi kesalahan: {str(e)}")

def handle_sidebar():
    """Handle sidebar with improved UI and functionality"""
    user = get_current_user()
    if not user:
        st.error("User not authenticated")
        return

    st.sidebar.write(f"👤 **{user['email']}**")
    st.sidebar.write("---")
    
    # API Settings
    with st.sidebar.expander("API Settings", expanded=False):
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
                with st.spinner("Menyimpan..."):
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
                with st.spinner("Menyimpan..."):
                    st.session_state.selected_model = selected_model
                    st.session_state.selected_embedding_model = selected_embedding_model
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
            value="" if st.session_state.clear_url else st.session_state.get('url', ''),
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

def handle_query(query):
    """Handle query with improved error handling and state management"""
    try:
        if not st.session_state.api_key:
            st.error("Silakan masukkan GROQ API key yang valid")
            return

        # Add user message to history
        st.session_state.conversation_history.append({
            "role": "user",
            "content": query
        })

        # Prepare prompt with context if available
        if st.session_state.index is not None:
            with st.spinner("Mencari konteks yang relevan..."):
                model = load_embedding_model(st.session_state.selected_embedding_model)
                query_embedding = generate_embedding(query, model)
                relevant_doc_indices = search_index(st.session_state.index, query_embedding)
                context = "\n".join([st.session_state.documents[i][:1000] for i in relevant_doc_indices])
                prompt = f"Berdasarkan konteks berikut:\n\n{context}\n\nJawab pertanyaan ini: {query}"
        else:
            prompt = query

        # Generate response
        set_api_key(st.session_state.api_key)
        with st.spinner("Menghasilkan respons..."):
            response = query_llm(prompt, st.session_state.selected_model)

            if response and not response.startswith("An error occurred"):
                # Update state with successful response
                st.session_state.conversation_history.append({
                    "role": "assistant",
                    "content": response
                })
                st.session_state.chat_history.append(("user", query))
                st.session_state.chat_history.append(("assistant", response))
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response
                })
                st.session_state.last_response = response
                st.session_state.error_message = None
            else:
                st.session_state.error_message = "Gagal mendapatkan respons. Silakan cek API key Anda dan coba lagi."
                st.error(st.session_state.error_message)

    except Exception as e:
        error_msg = f"Terjadi kesalahan: {str(e)}"
        st.session_state.error_message = error_msg
        st.error(error_msg)

def process_file(uploaded_file):
    """Process uploaded file with improved error handling"""
    try:
        with st.spinner("Memproses file..."):
            file_path = save_uploaded_file(uploaded_file)
            content = read_file(file_path)
            
            # Update session state
            if content:
                st.session_state.documents.append(content)
                st.session_state.processed_files.append(uploaded_file.name)
                
                success_msg = f"File '{uploaded_file.name}' berhasil diproses!"
                st.success(success_msg)
                
                # Clear any existing error message
                st.session_state.error_message = None
            else:
                raise ValueError("Konten file kosong")
                
    except Exception as e:
        error_msg = f"Gagal memproses file: {str(e)}"
        st.session_state.error_message = error_msg
        st.error(error_msg)

def process_url(url):
    """Process URL with improved error handling"""
    try:
        with st.spinner("Memproses URL..."):
            content = read_url(url)
            
            # Update session state
            if content:
                st.session_state.documents.append(content)
                st.session_state.processed_urls.append(url)
                
                success_msg = f"URL '{url}' berhasil diproses!"
                st.success(success_msg)
                
                # Clear any existing error message
                st.session_state.error_message = None
            else:
                raise ValueError("Konten URL kosong")
                
    except Exception as e:
        error_msg = f"Gagal memproses URL: {str(e)}"
        st.session_state.error_message = error_msg
        st.error(error_msg)

def generate_embeddings():
    """Generate embeddings with improved progress tracking"""
    if not st.session_state.documents:
        st.warning("Tidak ada dokumen yang perlu diproses")
        return
        
    try:
        with st.spinner("Menghasilkan embeddings..."):
            model = load_embedding_model(st.session_state.selected_embedding_model)
            st.session_state.embeddings = []
            
            # Create progress bar
            progress_text = "Menghasilkan embeddings..."
            progress_bar = st.progress(0, text=progress_text)
            
            for i, doc in enumerate(st.session_state.documents):
                embedding = generate_embedding(doc, model)
                st.session_state.embeddings.append(embedding)
                
                # Update progress
                progress = (i + 1) / len(st.session_state.documents)
                progress_bar.progress(
                    progress,
                    text=f"{progress_text} ({i + 1}/{len(st.session_state.documents)})"
                )
            
            success_msg = f"Berhasil menghasilkan embeddings untuk {len(st.session_state.embeddings)} dokumen"
            st.success(success_msg)
            
            # Clear any existing error message
            st.session_state.error_message = None
            
    except Exception as e:
        error_msg = f"Gagal menghasilkan embeddings: {str(e)}"
        st.session_state.error_message = error_msg
        st.error(error_msg)

def create_search_index():
    """Create search index with improved error handling"""
    if not st.session_state.embeddings:
        st.warning("Harap generate embeddings terlebih dahulu")
        return
        
    try:
        with st.spinner("Membuat index pencarian..."):
            st.session_state.index = create_index(st.session_state.embeddings)
            
            if st.session_state.index is not None:
                success_msg = "Index pencarian berhasil dibuat!"
                st.success(success_msg)
                
                # Clear any existing error message
                st.session_state.error_message = None
            else:
                raise ValueError("Gagal membuat index")
                
    except Exception as e:
        error_msg = f"Gagal membuat index: {str(e)}"
        st.session_state.error_message = error_msg
        st.error(error_msg)

def clean_session_data():
    """Clean session data with improved feedback"""
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
            st.session_state.messages = []
            st.session_state.last_response = None
            st.session_state.error_message = None
            
            success_msg = "Semua data berhasil dibersihkan!"
            st.success(success_msg)
            
            # Refresh the UI
            st.rerun()
            
    except Exception as e:
        error_msg = f"Gagal membersihkan data: {str(e)}"
        st.session_state.error_message = error_msg
        st.error(error_msg)

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

        # Render main application
        with st.sidebar:
            handle_sidebar()
        handle_main_area()

    except Exception as e:
        st.error(f"Terjadi kesalahan aplikasi: {str(e)}")

if __name__ == "__main__":
    main()