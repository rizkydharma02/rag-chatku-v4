import streamlit as st
from datetime import datetime
from typing import Optional
from utils import query_llm, generate_embedding, load_embedding_model, search_index
from db_utils import DatabaseManager
import logging

logger = logging.getLogger(__name__)

class ChatManager:
    def __init__(self):
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        self.db_manager = DatabaseManager()
            
    def add_message(self, role: str, content: str):
        if not content:
            return
            
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        }
        st.session_state.chat_messages.append(message)
        
    def get_context(self, query: str) -> Optional[str]:
        try:
            if st.session_state.get('index') is not None:
                model = load_embedding_model(st.session_state.selected_embedding_model)
                query_embedding = generate_embedding(query, model)
                relevant_doc_indices = search_index(st.session_state.index, query_embedding)
                return "\n".join([st.session_state.documents[i][:1000] for i in relevant_doc_indices])
            return None
        except Exception as e:
            st.error(f"Error getting context: {str(e)}")
            return None
            
    def handle_chat_interface(self):
        for msg in st.session_state.chat_messages:
            with st.container():
                if msg['role'] == 'user':
                    st.write(f"**You**: {msg['content']}")
                    st.caption(f"{msg['timestamp'].strftime('%H:%M')} ✓✓")
                else:
                    st.write(f"**🤖 Assistant**: {msg['content']}")
                    st.caption(msg['timestamp'].strftime('%H:%M'))
            st.write("---")

        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Message", key="user_input", 
                                     placeholder="Ketik pesan Anda di sini...")
            submit_button = st.form_submit_button("Kirim", use_container_width=True)

            if submit_button and user_input:
                try:
                    # Add user message
                    self.add_message("user", user_input)
                    
                    # Get context if available
                    context = self.get_context(user_input)
                    
                    # Prepare prompt
                    prompt = (f"Berdasarkan konteks berikut:\n\n{context}\n\n"
                            f"Jawab pertanyaan ini: {user_input}") if context else user_input
                    
                    # Get response
                    with st.spinner("Menghasilkan respons..."):
                        response = query_llm(prompt, st.session_state.selected_model)
                    
                    if response and not response.startswith("Error"):
                        # Save to database first
                        try:
                            self.db_manager.save_chat(user_input, response)
                            logger.info("Successfully saved chat to database")
                        except Exception as e:
                            logger.error(f"Failed to save chat to database: {str(e)}")
                        
                        # Then add to session state
                        self.add_message("assistant", response)
                        st.rerun()
                    else:
                        st.error(response)
                    
                except Exception as e:
                    logger.error(f"Chat interface error: {str(e)}")
                    st.error(f"Error: {str(e)}")

    def __del__(self):
        if hasattr(self, 'db_manager'):
            del self.db_manager

def initialize_chat_state():
    default_states = {
        'chat_messages': [],
        'documents': [],
        'embeddings': [],
        'index': None,
        'processed_files': [],
        'processed_urls': [],
        'selected_model': "mixtral-8x7b-32768",
        'selected_embedding_model': "all-MiniLM-L6-v2",
        'api_key': "",
        'clear_url': False
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value