import streamlit as st
from datetime import datetime
import asyncio
from typing import Optional, Dict, Any
from utils import set_api_key, query_llm, generate_embedding, load_embedding_model, search_index

class ChatManager:
    def __init__(self):
        self._initialize_state()
        
    def _initialize_state(self):
        """Initialize necessary session state variables"""
        if 'chat_messages' not in st.session_state:
            st.session_state.chat_messages = []
        if 'processing' not in st.session_state:
            st.session_state.processing = False
        if 'last_response' not in st.session_state:
            st.session_state.last_response = None
            
    def add_message(self, role: str, content: str):
        """Add a message to chat history with timestamp"""
        if not content:
            return
            
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now()
        }
        
        st.session_state.chat_messages.append(message)
        
    def process_message(self, query: str, context: Optional[str] = None) -> str:
        """Process a message and get response from LLM"""
        try:
            if not st.session_state.get('api_key'):
                return "Error: API key not found. Please check your API key settings."
                
            st.session_state.processing = True
            prompt = f"Berdasarkan konteks berikut:\n\n{context}\n\nJawab pertanyaan ini: {query}" if context else query
            
            try:
                set_api_key(st.session_state.api_key)
                response = query_llm(prompt, st.session_state.selected_model)
                
                if response and not response.startswith("Error"):
                    return response
                return "Failed to get response. Please check your API key and try again."
                
            except Exception as e:
                return f"Error processing query: {str(e)}"
            
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            st.session_state.processing = False
            
    def get_context_for_query(self, query: str) -> Optional[str]:
        """Get relevant context for a query if index exists"""
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
        """Main chat interface handler"""
        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.container():
                if msg['role'] == 'user':
                    st.write(f"**You**: {msg['content']}")
                    st.caption(f"{msg['timestamp'].strftime('%H:%M')} ✓✓")
                else:
                    st.write(f"**🤖 Assistant**: {msg['content']}")
                    st.caption(msg['timestamp'].strftime('%H:%M'))
            st.write("---")

        # Chat input
        with st.form(key="chat_form", clear_on_submit=True):
            user_input = st.text_input("Message", key="user_input", 
                                     placeholder="Ketik pesan Anda di sini...")
            col1, col2 = st.columns([6, 1])
            with col2:
                submit_button = st.form_submit_button("Kirim", use_container_width=True)

            if submit_button and user_input:
                self.add_message("user", user_input)
                
                try:
                    # Get context if available
                    context = self.get_context_for_query(user_input)
                    
                    # Process message
                    response = self.process_message(user_input, context)
                        
                    # Add response to chat
                    if response:
                        self.add_message("assistant", response)
                        st.session_state.last_response = response
                        
                    # Force refresh
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error processing message: {str(e)}")
                
def initialize_chat_state():
    """Initialize all necessary session state variables"""
    default_states = {
        'chat_messages': [],
        'processing': False,
        'last_response': None,
        'documents': [],
        'embeddings': [],
        'index': None,
        'processed_files': [],
        'processed_urls': [],
        'selected_model': "mixtral-8x7b-32768",  # default model
        'selected_embedding_model': "all-MiniLM-L6-v2",
        'api_key': "",
        'clear_url': False
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value