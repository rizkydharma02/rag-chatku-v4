from sqlalchemy import create_engine, Column, String, DateTime, text, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import bcrypt
import os
import streamlit as st
from dotenv import load_dotenv
import logging
import uuid

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_database_url():
    DATABASE_URL = os.getenv("DATABASE_URL")
    if not DATABASE_URL:
        raise Exception("DATABASE_URL not found in environment variables")
    return DATABASE_URL.replace("postgres://", "postgresql://", 1) if DATABASE_URL.startswith("postgres://") else DATABASE_URL

def init_database():
    try:
        engine = create_engine(get_database_url(), connect_args={"sslmode": "require"})
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}")
        raise

engine = init_database()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    groq_api_key = Column(String(255))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            "id": str(self.id) if self.id else None,
            "email": self.email,
            "groq_api_key": self.groq_api_key,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }

class Chatbot(Base):
    __tablename__ = "chatbot"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    question_chat = Column(Text, nullable=False)
    response_chat = Column(Text, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

class DatabaseManager:
    def __init__(self):
        self.db = SessionLocal()

    def __del__(self):
        if hasattr(self, 'db'):
            self.db.close()

    def save_chat(self, question: str, response: str) -> bool:
        try:
            chat = Chatbot(
                question_chat=question,
                response_chat=response
            )
            self.db.add(chat)
            self.db.commit()
            self.db.refresh(chat)
            return True
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to save chat: {str(e)}")
            return False

    def authenticate_user(self, email: str, password: str):
        try:
            user = self.db.query(User).filter(User.email == email).first()
            if not user:
                return None
            if bcrypt.checkpw(password.encode('utf-8'), user.hashed_password.encode('utf-8')):
                user_dict = user.to_dict()
                if user.groq_api_key:
                    st.session_state.api_key = user.groq_api_key
                return user_dict
            return None
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return None

    def create_user(self, email: str, password: str, groq_api_key: str = None):
        try:
            if self.db.query(User).filter(User.email == email).first():
                return None
            salt = bcrypt.gensalt()
            hashed = bcrypt.hashpw(password.encode('utf-8'), salt)
            user = User(
                email=email,
                hashed_password=hashed.decode('utf-8'),
                groq_api_key=groq_api_key
            )
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            return user.to_dict()
        except Exception as e:
            self.db.rollback()
            logger.error(f"Failed to create user: {str(e)}")
            return None

    def get_user_by_email(self, email: str):
        try:
            user = self.db.query(User).filter(User.email == email).first()
            return user.to_dict() if user else None
        except Exception as e:
            logger.error(f"Error getting user by email: {str(e)}")
            return None