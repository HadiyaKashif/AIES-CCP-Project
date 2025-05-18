# components/utils.py
import os
import google.generativeai as genai
from dotenv import load_dotenv
from flask import session, flash
from typing import Dict, List, Any, Optional

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("GEMINI_API_KEY not found in environment variables.")

# Configure Gemini API
genai.configure(api_key=api_key)

# Load the Gemini model
model = genai.GenerativeModel(model_name="models/gemini-1.5-flash-latest")

# Function to generate content
def gemini_llm(prompt: str) -> str:
    """Generate a response from the Gemini model for the given prompt."""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error from Gemini API: {str(e)}"

def get_session_value(key: str, default: Any = None) -> Any:
    """Safely get a value from the Flask session."""
    return session.get(key, default)

def set_session_value(key: str, value: Any) -> None:
    """Safely set a value in the Flask session."""
    session[key] = value

def flash_message(message: str, category: str = 'info') -> None:
    """Flash a message with the given category."""
    flash(message, category)

def init_session_vars(vars_dict: Dict[str, Any]) -> None:
    """Initialize multiple session variables if they don't exist."""
    for key, default in vars_dict.items():
        if key not in session:
            session[key] = default



