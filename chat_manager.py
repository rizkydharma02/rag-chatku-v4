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
            
    def add_message(self, role: str, content: str, save_to_db: bool = False, user_input: str = None):
        if not content:
            return
            
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        }
        st.session_state.chat_messages.append(message)
        
        if save_to_db and role == "assistant" and user_input:
            try:
                saved = self.db_manager.save_chat(user_input, content)
                if not saved:
                    logger.error("Database save failed")
                    raise Exception("Failed to save to database")
            except Exception as e:
                logger.error(f"Save chat error: {str(e)}")
                raise
            
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
                    self.add_message("user", user_input)
                    context = self.get_context(user_input)
                    prompt = (f"Berdasarkan konteks berikut:\n\n{context}\n\n"
                            f"Jawab pertanyaan ini: {user_input}") if context else user_input
                    
                    with st.spinner("Menghasilkan respons..."):
                        response = query_llm(prompt, st.session_state.selected_model)

                    if response and not response.startswith("Error"):
                        self.add_message("assistant", response, save_to_db=True, user_input=user_input)
                        st.rerun()
                    else:
                        st.error(response)

                except Exception as e:
                    logger.error(f"Chat error: {str(e)}")
                    st.error("Terjadi kesalahan dalam memproses chat")

    def __del__(self):
        if hasattr(self, 'db_manager'):
            del self.db_manager