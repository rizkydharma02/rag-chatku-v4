import streamlit as st
import time
from datetime import datetime

# Must be the first Streamlit command
st.set_page_config(
    page_title="Chatku AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
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

def local_css(file_name):
    with open(file_name, 'r') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def initialize_session_state():
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
        'last_timestamp': None
    }
    
    for key, default_value in default_values.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

def render_login_page():
    # Remove margins/padding
    st.markdown("""
        <style>
        .block-container {
            padding: 1rem !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Logo and welcome message
    st.markdown("""
        <div style='text-align: center; padding: 2rem 0;'>
            <h1 style='margin-bottom: 0.5rem; color: #333;'>🤖 Chatku AI</h1>
            <p style='color: #666; font-size: 1.1em;'>Chatku AI Dengan Retrieval Augmented Generation</p>
        </div>
    """, unsafe_allow_html=True)
    
    # Center container
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        tabs = st.tabs(["Login", "Daftar"])
        
        with tabs[0]:
            with st.form("login_form"):
                st.write("### Login")
                
                login_email = st.text_input(
                    "📧 Email",
                    key="login_email",
                    placeholder="Masukkan email anda"
                )
                
                login_password = st.text_input(
                    "🔒 Password",
                    type="password",
                    key="login_password",
                    placeholder="Masukkan password anda"
                )
                
                submitted = st.form_submit_button("Login", type="primary", use_container_width=True)
                if submitted:
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
        
        with tabs[1]:
            with st.form("register_form"):
                st.write("### Daftar Akun Baru")
                
                signup_email = st.text_input(
                    "📧 Email",
                    key="signup_email_input",
                    placeholder="Masukkan email anda"
                )
                
                signup_password = st.text_input(
                    "🔒 Password",
                    type="password",
                    key="signup_password_input",
                    placeholder="Buat password anda"
                )
                
                signup_password_confirm = st.text_input(
                    "🔒 Konfirmasi Password",
                    type="password",
                    key="signup_password_confirm_input",
                    placeholder="Masukkan ulang password"
                )
                
                signup_groq_api = st.text_input(
                    "🔑 GROQ API Key",
                    type="password",
                    key="signup_groq_api_input",
                    help="Dapatkan API key dari https://console.groq.com/",
                    placeholder="Masukkan GROQ API key"
                )

                # Password requirements
                st.markdown("""
                    <div style='background: #f8f9fa; padding: 1rem; border-radius: 5px; margin: 1rem 0;'>
                        <p style='margin: 0 0 0.5rem 0; color: #666;'><strong>Persyaratan Password:</strong></p>
                        <ul style='margin: 0; padding-left: 1.5rem; color: #666; font-size: 0.9em;'>
                            <li>Minimal 6 karakter</li>
                            <li>Minimal 1 huruf besar</li>
                            <li>Minimal 1 huruf kecil</li>
                            <li>Minimal 1 angka</li>
                        </ul>
                    </div>
                """, unsafe_allow_html=True)
                
                submitted = st.form_submit_button("Daftar", type="primary", use_container_width=True)
                if submitted:
                    if signup_password != signup_password_confirm:
                        st.error("Password tidak cocok")
                    elif not signup_groq_api:
                        st.error("GROQ API Key wajib diisi")
                    elif signup_email and signup_password:
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
                    else:
                        st.warning("Silakan lengkapi semua field")


def get_timestamp():
    if not st.session_state.last_timestamp or \
       (datetime.now() - st.session_state.last_timestamp).seconds > 60:
        st.session_state.last_timestamp = datetime.now()
        return st.session_state.last_timestamp.strftime("%H:%M")
    return None

def display_chat_history():
    current_date = None
    
    for role, message in st.session_state.chat_history:
        message_time = get_timestamp()
        
        # Show date divider if it's a new day
        message_date = datetime.now().strftime("%d %B %Y")
        if current_date != message_date:
            current_date = message_date
            st.markdown(
                f"""
                <div class="date-divider">
                    <span>{message_date}</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        if role == "user":
            st.markdown(
                f"""
                <div class="chat-message user">
                    <div class="message-bubble">
                        <div class="message-text">{message}</div>
                        <div class="message-info">
                            <span class="message-time">{message_time if message_time else ''}</span>
                            <span class="message-status">✓✓</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="chat-message assistant">
                    <div class="avatar">🤖</div>
                    <div class="message-bubble">
                        <div class="message-text">{message}</div>
                        <div class="message-info">
                            <span class="message-time">{message_time if message_time else ''}</span>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

def handle_main_area():
    # Remove default streamlit margins and padding
    st.markdown("""
        <style>
        .block-container {
            padding-top: 0 !important;
            padding-bottom: 0 !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        .element-container {
            margin: 0 !important;
        }
        .stMarkdown {
            margin-bottom: 0 !important;
        }
        header[data-testid="stHeader"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0;">
            <h1 style="color: #333; margin: 0; font-size: 2rem;">Chatku AI</h1>
            <p style="color: #666; margin: 0.5rem 0 0 0; font-size: 1.1rem;">Chatku AI dengan Retrieval Augmented Generation</p>
        </div>
    """, unsafe_allow_html=True)

    # Display chat history
    display_chat_history()
    
    # Form container
    with st.form(key="chat_form", clear_on_submit=True):
        st.markdown("""
            <style>
            .stForm {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                padding: 1rem;
                background-color: white;
                box-shadow: 0 -2px 10px rgba(0,0,0,0.1);
                z-index: 1000;
            }
            </style>
        """, unsafe_allow_html=True)
        
        cols = st.columns([6, 1])
        
        with cols[0]:
            query = st.text_input(
                label="Message Input",
                placeholder="Ketik pesan Anda di sini...",
                label_visibility="collapsed",
                key="current_query"
            )
        
        with cols[1]:
            submitted = st.form_submit_button(
                "Kirim",
                type="primary",
                use_container_width=True
            )
        
        if submitted and query:
            handle_query(query)
            st.rerun()

def handle_sidebar():
    user = get_current_user()
    if not user:
        st.error("User not authenticated")
        return

    # User profile section
    st.sidebar.markdown("""
        <div style='padding: 1rem; margin-bottom: 1rem;'>
            <div style='display: flex; align-items: center; margin-bottom: 0.5rem;'>
                <div style='font-size: 1.5rem; margin-right: 0.5rem;'>👤</div>
                <div>
                    <div style='font-weight: 500;'>{}</div>
                    <div style='font-size: 0.8rem; color: #4caf50;'>Online</div>
                </div>
            </div>
        </div>
    """.format(user["email"]), unsafe_allow_html=True)
    
    st.sidebar.markdown("---")
    
    # API Key section
    with st.sidebar.expander("API Settings", expanded=True):
        st.write("#### GROQ API Key Configuration")
        
        # Show current API key status
        current_api_key = st.session_state.get('api_key', '')
        if current_api_key:
            st.success("API Key sudah terpasang")
        else:
            st.warning("API Key belum diatur")
            
        with st.form("api_key_form"):
            # API Key input
            new_api_key = st.text_input(
                label="GROQ API Key",
                value=current_api_key,
                type="password",
                help="Dapatkan API key dari https://console.groq.com",
                placeholder="Masukkan GROQ API Key Anda",
                key="api_key_input"
            )
            
            # Submit button
            submitted = st.form_submit_button(
                "Simpan API Key",
                type="primary",
                use_container_width=True
            )
            
            if submitted:
                try:
                    # Validate input
                    if not new_api_key:
                        st.error("API key tidak boleh kosong")
                        return
                        
                    if new_api_key == current_api_key:
                        st.info("API key tidak berubah")
                        return
                    
                    # Update API key
                    with st.spinner("Menyimpan API key..."):
                        # Try to save to database
                        success = st.session_state.db_manager.save_api_key(
                            user_id=user["id"],
                            api_key=new_api_key
                        )
                        
                        if success:
                            # Update session state and client
                            st.session_state.api_key = new_api_key
                            set_api_key(new_api_key)
                            
                            st.success("API key berhasil disimpan!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Gagal menyimpan API key ke database")
                        
                except Exception as e:
                    st.error(f"Terjadi kesalahan: {str(e)}")
                    logger.error(f"Error saving API key: {str(e)}")


    # Model Selection
    with st.sidebar.expander("Model Settings", expanded=False):
        with st.form("model_settings_form"):
            available_models = get_available_models()
            selected_model = st.selectbox(
                "Pilih LLM Model:",
                available_models,
                index=available_models.index(st.session_state.selected_model) if st.session_state.selected_model in available_models else 0,
                key="model_selector"
            )

            embedding_models = ["all-MiniLM-L6-v2"]
            selected_embedding_model = st.selectbox(
                "Pilih Embedding Model:",
                embedding_models,
                index=0,
                key="embedding_model_selector"
            )
            
            submitted = st.form_submit_button("Simpan Model", type="primary")
            if submitted:
                st.session_state.selected_model = selected_model
                st.session_state.selected_embedding_model = selected_embedding_model
                st.success("Model berhasil diperbarui!")

    # Document Processing
    with st.sidebar.expander("Document Processing", expanded=False):
        with st.form("document_processing_form"):
            uploaded_file = st.file_uploader(
                "Upload File (PDF/Word)",
                type=['pdf', 'docx'],
                key="file_uploader"
            )
            
            url = st.text_input(
                label="URL Input",
                value="" if st.session_state.clear_url else st.session_state.get('url', ''),
                placeholder="Atau masukkan URL",
                key="url_input",
                label_visibility="visible"
            )
            
            submitted = st.form_submit_button("Proses File/URL", type="primary")
            if submitted:
                if uploaded_file:
                    process_file(uploaded_file)
                elif url:
                    process_url(url)
                else:
                    st.warning("Silakan upload file atau masukkan URL")

        # Separate form for embeddings and index
        with st.form("embeddings_form"):
            col1, col2 = st.columns(2)
            with col1:
                generate_embeddings_button = st.form_submit_button(
                    "Generate Embeddings",
                    use_container_width=True
                )
            with col2:
                create_index_button = st.form_submit_button(
                    "Buat Index",
                    use_container_width=True
                )
            
            if generate_embeddings_button:
                generate_embeddings()
            if create_index_button:
                create_search_index()

        if st.session_state.processed_files or st.session_state.processed_urls:
            display_processed_items()

        with st.form("clear_data_form"):
            if st.form_submit_button("Clear Data", type="primary", use_container_width=True):
                clean_session_data()

    # Logout button
    st.sidebar.markdown("---")
    with st.sidebar.form("logout_form"):
        if st.form_submit_button("Logout", type="primary", use_container_width=True):
            logout_user()
            st.rerun()

def process_file(uploaded_file):
    try:
        with st.spinner("Memproses file..."):
            file_path = save_uploaded_file(uploaded_file)
            content = read_file(file_path)
            st.session_state.documents.append(content)
            st.session_state.processed_files.append(uploaded_file.name)
            st.success(f"File '{uploaded_file.name}' berhasil diproses!")
    except Exception as e:
        st.error(f"Gagal memproses file: {str(e)}")

def process_url(url):
    try:
        with st.spinner("Memproses URL..."):
            content = read_url(url)
            st.session_state.documents.append(content)
            st.session_state.processed_urls.append(url)
            st.success(f"URL '{url}' berhasil diproses!")
    except Exception as e:
        st.error(f"Gagal memproses URL: {str(e)}")

def handle_query(query):
    try:
        if not st.session_state.api_key:
            st.error("Silakan masukkan GROQ API key yang valid")
            return

        st.session_state.conversation_history.append({"role": "user", "content": query})

        if st.session_state.index is not None:
            with st.spinner("Mencari konteks yang relevan..."):
                model = load_embedding_model(st.session_state.selected_embedding_model)
                query_embedding = generate_embedding(query, model)
                relevant_doc_indices = search_index(st.session_state.index, query_embedding)
                
                context = "\n".join([st.session_state.documents[i][:1000] for i in relevant_doc_indices])
                prompt = f"Berdasarkan konteks berikut:\n\n{context}\n\nJawab pertanyaan ini: {query}"
        else:
            prompt = query

        set_api_key(st.session_state.api_key)

        with st.spinner("Menghasilkan respons..."):
            response = query_llm(prompt, st.session_state.selected_model)

        if response and not response.startswith("An error occurred"):
            st.session_state.conversation_history.append({"role": "assistant", "content": response})
            st.session_state.chat_history.append(("user", query))
            st.session_state.chat_history.append(("assistant", response))
        else:
            st.error("Gagal mendapatkan respons. Silakan cek API key Anda dan coba lagi.")

    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")

def generate_embeddings():
    if st.session_state.documents:
        with st.spinner("Menghasilkan embeddings..."):
            model = load_embedding_model(st.session_state.selected_embedding_model)
            st.session_state.embeddings = []
            progress_text = "Operasi sedang berlangsung. Mohon tunggu."
            progress_bar = st.progress(0, text=progress_text)
            
            for i, doc in enumerate(st.session_state.documents):
                embedding = generate_embedding(doc, model)
                st.session_state.embeddings.append(embedding)
                progress_bar.progress(
                    (i + 1) / len(st.session_state.documents),
                    text=f"Memproses dokumen {i+1} dari {len(st.session_state.documents)}"
                )
            
            st.success(f"Berhasil menghasilkan embeddings untuk {len(st.session_state.embeddings)} dokumen")
    else:
        st.warning("Tidak ada dokumen yang perlu diproses")

def create_search_index():
    if len(st.session_state.embeddings) > 0:
        with st.spinner("Membuat index pencarian..."):
            st.session_state.index = create_index(st.session_state.embeddings)
            st.success("Index pencarian berhasil dibuat!")
    else:
        st.warning("Harap generate embeddings terlebih dahulu")

def display_processed_items():
    if st.session_state.processed_files:
        st.markdown("#### File yang Diproses:")
        for file in st.session_state.processed_files:
            st.markdown(f"- {file}")

    if st.session_state.processed_urls:
        st.markdown("#### URL yang Diproses:")
        for url in st.session_state.processed_urls:
            st.markdown(f"- {url}")

def clean_session_data():
    with st.spinner("Membersihkan data..."):
        st.session_state.documents = []
        st.session_state.embeddings = []
        st.session_state.chat_history = []
        st.session_state.conversation_history = []
        st.session_state.index = None
        st.session_state.processed_files = []
        st.session_state.processed_urls = []
        st.session_state.clear_url = True
        st.session_state.query_input = ""
        st.success("Data berhasil dibersihkan")
        st.rerun()

def main():
    initialize_session_state()
    local_css("style.css")

    # Show login page if not authenticated
    if 'token' not in st.session_state:
        render_login_page()
        return

    # Verify user session
    user = get_current_user()
    if not user:
        st.session_state.token = None
        st.rerun()
        return

    # Sidebar
    with st.sidebar:
        handle_sidebar()

    # Main chat area
    handle_main_area()

if __name__ == "__main__":
    main()