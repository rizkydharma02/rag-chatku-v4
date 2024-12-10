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

def get_timestamp():
    if not st.session_state.last_timestamp or \
       (datetime.now() - st.session_state.last_timestamp).seconds > 60:
        st.session_state.last_timestamp = datetime.now()
        return st.session_state.last_timestamp.strftime("%H:%M")
    return None

def display_chat_history():
    for role, message in st.session_state.chat_history:
        message_time = get_timestamp()
        
        if role == "user":
            message_container = st.container()
            with message_container:
                st.write(f"**You**: {message}")
                if message_time:
                    st.caption(f"{message_time} ✓✓")
        else:
            message_container = st.container()
            with message_container:
                st.write(f"**🤖 Assistant**: {message}")
                if message_time:
                    st.caption(message_time)
        st.write("---")

def handle_main_area():
    st.title("Chatku AI")
    st.caption("Chatku AI dengan Retrieval Augmented Generation")
    
    # Display chat history
    display_chat_history()
    
    # Chat input
    with st.form(key="chat_form", clear_on_submit=True):
        query = st.text_input("Message", placeholder="Ketik pesan Anda di sini...", key="current_query")
        if st.form_submit_button("Kirim"):
            if query:
                handle_query(query)
                st.rerun()

def handle_sidebar():
    user = get_current_user()
    if not user:
        st.error("User not authenticated")
        return

    st.sidebar.write(f"👤 **{user['email']}**")
    st.sidebar.write("---")
    
    with st.sidebar.expander("API Settings"):
        current_api_key = st.session_state.get('api_key', '')
        
        # Tambahkan informasi tentang GROQ API Key
        st.markdown("""
            ### Cara Mendapatkan GROQ API Key
            1. Kunjungi [console.groq.com](https://console.groq.com/)
            2. Buat akun atau login jika sudah punya akun
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
        # File Upload
        uploaded_file = st.file_uploader("Upload File (PDF/Word)", type=['pdf', 'docx'])
        
        # URL Input
        url = st.text_input(
            "URL Input",
            value="" if st.session_state.clear_url else st.session_state.get('url', ''),
            placeholder="Atau masukkan URL"
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
                st.error(f"Gagal memproses: {str(e)}")

        st.write("---")
        
        if st.button("Generate Embeddings", use_container_width=True):
            generate_embeddings()
            
        if st.button("Buat Index", use_container_width=True):
            create_search_index()

        # Display processed items
        if st.session_state.processed_files or st.session_state.processed_urls:
            st.write("**Processed Items:**")
            if st.session_state.processed_files:
                for file in st.session_state.processed_files:
                    st.write(f"📄 {file}")
            if st.session_state.processed_urls:
                for url in st.session_state.processed_urls:
                    st.write(f"🔗 {url}")

        st.write("---")
        if st.button("Hapus Semua Data", use_container_width=True):
            clean_session_data()

    st.sidebar.write("---")
    if st.sidebar.button("Logout", use_container_width=True):
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
                
            st.success(f"Berhasil menghasilkan embeddings untuk {len(st.session_state.embeddings)} dokumen")
    except Exception as e:
        st.error(f"Gagal menghasilkan embeddings: {str(e)}")

def create_search_index():
    if not st.session_state.embeddings:
        st.warning("Harap generate embeddings terlebih dahulu")
        return
        
    try:
        with st.spinner("Membuat index pencarian..."):
            st.session_state.index = create_index(st.session_state.embeddings)
            st.success("Index pencarian berhasil dibuat!")
    except Exception as e:
        st.error(f"Gagal membuat index: {str(e)}")

def clean_session_data():
    """
    Membersihkan semua data dari session state dan mengembalikan aplikasi ke keadaan awal
    """
    try:
        with st.spinner("Membersihkan data..."):
            # Reset semua nilai ke default
            st.session_state.documents = []
            st.session_state.embeddings = []
            st.session_state.chat_history = []
            st.session_state.conversation_history = []
            st.session_state.index = None
            st.session_state.processed_files = []
            st.session_state.processed_urls = []
            st.session_state.clear_url = True
            
            # Tampilkan pesan sukses
            st.success("Semua data berhasil dibersihkan!")
            
            # Refresh tampilan
            st.rerun()
            
    except Exception as e:
        st.error(f"Gagal membersihkan data: {str(e)}")

def main():
    """
    Fungsi utama untuk menjalankan aplikasi
    """
    initialize_session_state()

    # Cek autentikasi
    if 'token' not in st.session_state:
        render_login_page()
        return

    # Verifikasi user session
    user = get_current_user()
    if not user:
        st.session_state.token = None
        st.rerun()
        return

    # Render main application
    with st.sidebar:
        handle_sidebar()
    handle_main_area()

if __name__ == "__main__":
    main()