import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.tools import Tool

# Load environment variables from .env file
load_dotenv()

# Load environment variables if present, otherwise use provided values
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '')
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY', '')
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY', '')
GOOGLE_CSE_ID = os.environ.get('GOOGLE_CSE_ID', '')
PINECONE_INDEX_NAME = "rag-advanced"

# Print environment variable status (for debugging)
print(f"PINECONE_API_KEY loaded: {'Yes' if PINECONE_API_KEY else 'No'}")
print(f"GEMINI_API_KEY loaded: {'Yes' if GEMINI_API_KEY else 'No'}")
print(f"GOOGLE_API_KEY loaded: {'Yes' if GOOGLE_API_KEY else 'No'}")
print(f"GOOGLE_CSE_ID loaded: {'Yes' if GOOGLE_CSE_ID else 'No'}")

# Initialize services
pc = Pinecone(api_key=PINECONE_API_KEY)
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)
gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

pinecone_index = pc.Index(PINECONE_INDEX_NAME)
# Search tool setup
if GOOGLE_API_KEY and GOOGLE_CSE_ID:
    try:
        search = GoogleSearchAPIWrapper(
            google_api_key=GOOGLE_API_KEY,
            google_cse_id=GOOGLE_CSE_ID
        )
        web_search_tool = Tool(
            name="Google Search",
            description="Search Google for recent results",
            func=lambda query: search.results(query, num_results=3)
        )
        print("Google Search API initialized successfully")
    except Exception as e:
        print(f"Error initializing Google Search API: {str(e)}")
        search = None
        web_search_tool = None
else:
    search = None
    web_search_tool = None
    print("Google Search API not initialized (missing API keys)")