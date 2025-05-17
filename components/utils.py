# components/utils.py
import os
import google.generativeai as genai
from dotenv import load_dotenv

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



