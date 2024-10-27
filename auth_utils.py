import jwt
from datetime import datetime, timedelta
import streamlit as st
from typing import Optional, Dict, Any
import os
from db_utils import DatabaseManager
import json
import uuid

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key-here").encode('utf-8')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

class UUIDEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, uuid.UUID):
            return str(obj)
        return super().default(obj)

def create_access_token(data: Dict[str, Any]) -> str:
    """
    Create a new JWT access token.
    
    Args:
        data (dict): The data to encode in the token
        
    Returns:
        str: The encoded JWT token
    """
    try:
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        to_encode.update({"exp": expire.timestamp()})
        
        # Convert data to be JSON serializable
        for key in to_encode:
            if isinstance(to_encode[key], datetime):
                to_encode[key] = to_encode[key].timestamp()
            elif isinstance(to_encode[key], uuid.UUID):
                to_encode[key] = str(to_encode[key])
        
        # Create token using custom JSON encoder
        encoded_jwt = jwt.encode(
            to_encode,
            SECRET_KEY,
            algorithm=ALGORITHM,
            json_encoder=UUIDEncoder
        )
        print(f"Created token successfully")  # Debug print
        return encoded_jwt
    except Exception as e:
        print(f"Error creating token: {str(e)}")
        return None

def decode_token(token: str) -> Dict[str, Any]:
    """
    Decode and validate a JWT token.
    
    Args:
        token (str): The JWT token to decode
        
    Returns:
        dict: The decoded token payload or None if invalid
    """
    if not token:
        print("No token provided")  # Debug print
        return None
    try:
        # Set some leeway for clock skew
        payload = jwt.decode(
            token, 
            SECRET_KEY, 
            algorithms=[ALGORITHM],
            options={"verify_exp": True, "leeway": 10}
        )
        return payload
    except jwt.ExpiredSignatureError:
        print("Token has expired")
        return None
    except Exception as e:
        print(f"Error decoding token: {str(e)}")
        return None

def login_user(db_manager: DatabaseManager, email: str, password: str) -> Dict[str, Any]:
    """
    Authenticate user and create session.
    
    Args:
        db_manager: Database manager instance
        email (str): User's email
        password (str): User's password
        
    Returns:
        dict: Session data including access token and user info
    """
    try:
        print(f"Attempting login for user: {email}")  # Debug print
        user = db_manager.authenticate_user(email, password)
        if not user:
            print("Authentication failed")  # Debug print
            return None

        # Create token data with string conversion for UUID
        token_data = {
            "sub": str(user["email"]),
            "id": str(user["id"]) if isinstance(user["id"], uuid.UUID) else user["id"],
            "iat": datetime.utcnow().timestamp()
        }
        
        access_token = create_access_token(token_data)
        if not access_token:
            print("Failed to create access token")
            return None
        
        # Set session state
        st.session_state.user_id = user["id"]
        st.session_state.email = user["email"]
        st.session_state.token = access_token
        st.session_state.login_time = datetime.utcnow().isoformat()
        
        print(f"Login successful for user: {email}")  # Debug print
        return {"access_token": access_token, "user": user}
    except Exception as e:
        print(f"Login error: {str(e)}")
        return None

def get_current_user() -> Dict[str, Any]:
    """
    Get current authenticated user from session.
    
    Returns:
        dict: User data if authenticated, None otherwise
    """
    try:
        if 'token' not in st.session_state:
            return None

        if 'login_time' in st.session_state:
            login_time = datetime.fromisoformat(st.session_state.login_time)
            if datetime.utcnow() - login_time > timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES):
                print("Session expired")
                logout_user()
                return None
            
        token = st.session_state.token
        payload = decode_token(token)
        if not payload:
            return None
            
        return {
            "id": payload["id"],
            "email": payload["sub"]
        }
    except Exception as e:
        print(f"Error getting current user: {str(e)}")
        return None

def register_user(db_manager: DatabaseManager, email: str, password: str, groq_api_key: str) -> Dict[str, Any]:
    """
    Register a new user.
    
    Args:
        db_manager: Database manager instance
        email (str): User's email
        password (str): User's password
        groq_api_key (str): User's GROQ API key
        
    Returns:
        dict: User data if registration successful, None otherwise
    """
    try:
        if not validate_email(email):
            st.error("Please enter a valid email address")
            return None
            
        is_valid_password, message = validate_password(password)
        if not is_valid_password:
            st.error(message)
            return None
            
        if not validate_groq_api_key(groq_api_key):
            st.error("Please enter a valid GROQ API key")
            return None
            
        user = db_manager.create_user(email, password, groq_api_key)
        if user:
            print(f"Registration successful for: {email}")  # Debug print
            return user
        else:
            print("User creation failed")  # Debug print
            return None
    except Exception as e:
        print(f"Registration error: {str(e)}")
        st.error(f"Registration failed: {str(e)}")
        return None

def logout_user():
    """Clear all session data for logout."""
    keys_to_clear = ['token', 'user_id', 'email', 'api_key', 'user', 'login_time']
    for key in keys_to_clear:
        if key in st.session_state:
            del st.session_state[key]

def validate_email(email: str) -> bool:
    """Validate email format."""
    import re
    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
    return bool(re.match(pattern, email))

def validate_password(password: str) -> tuple[bool, str]:
    """
    Validate password requirements.
    
    Returns:
        tuple: (is_valid, message)
    """
    if len(password) < 6:
        return False, "Kata sandi harus terdiri dari minimal 6 karakter"
    if not any(c.isupper() for c in password):
        return False, "Kata sandi harus mengandung setidaknya satu huruf besar"
    if not any(c.islower() for c in password):
        return False, "Kata sandi harus mengandung setidaknya satu huruf kecil"
    if not any(c.isdigit() for c in password):
        return False, "Kata sandi harus mengandung setidaknya satu nomor"
    return True, "Password is valid"

def validate_groq_api_key(api_key: str) -> bool:
    """
    Validate GROQ API key format.
    
    Args:
        api_key (str): The API key to validate
        
    Returns:
        bool: True if valid, False otherwise
    """
    if not api_key:
        return False
    try:
        # Basic format validation
        api_key = api_key.strip()
        if not api_key.startswith('gsk_'):
            return False
            
        # Length validation
        if len(api_key) < 20:
            return False
            
        return True
    except Exception as e:
        print(f"Error validating API key: {str(e)}")
        return False