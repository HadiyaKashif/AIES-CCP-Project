# FINAL CODE FOR DOCUMENT PROCESSING

# import os
# import tempfile
# import streamlit as st
# import pytesseract
# from PIL import Image
# from pptx import Presentation
# from docx import Document
# from dotenv import load_dotenv
# from pinecone import Pinecone, ServerlessSpec
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_pinecone import PineconeVectorStore
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# from langchain_core.documents import Document as LangchainDocument

# # Load environment variables
# load_dotenv()

# # Configuration
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
# PINECONE_INDEX_NAME = "rag-practice"

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=768,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

# # Initialize Gemini components
# gemini_embeddings = GoogleGenerativeAIEmbeddings(
#     model="models/embedding-001",
#     google_api_key=GEMINI_API_KEY
# )

# gemini_llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     google_api_key=GEMINI_API_KEY,
#     temperature=0.3
# )

# # Streamlit UI
# st.set_page_config(page_title="Gemini RAG System", page_icon="🤖")
# st.title("🤖 Gemini-Powered Document Chat")
# st.caption("Upload documents and chat with them using Gemini AI")

# if "messages" not in st.session_state:
#     st.session_state.messages = []
# if "processed" not in st.session_state:
#     st.session_state.processed = False

# def extract_text_from_pptx(file_path):
#     text = []
#     prs = Presentation(file_path)
#     for slide in prs.slides:
#         for shape in slide.shapes:
#             if hasattr(shape, "text"):
#                 text.append(shape.text)
#     return "\n".join(text)

# def extract_text_from_docx(file_path):
#     try:
#         doc = Document(file_path)
#         text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
#         return text if text else "No text found in document."
#     except Exception as e:
#         return f"Error extracting text from DOCX: {str(e)}"

# def extract_text_from_image(file_path):
#     image = Image.open(file_path)
#     return pytesseract.image_to_string(image)

# def process_documents(uploaded_files):
#     docs = []
#     for file in uploaded_files:
#         with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
#             tmp.write(file.getvalue())
#             tmp_path = tmp.name

#         try:
#             file_extension = file.name.split('.')[-1].lower()
#             extracted_text = ""

#             if file_extension == "pdf":
#                 loader = PyPDFLoader(tmp_path)
#                 pages = loader.load_and_split()
#                 docs.extend(pages)
#             elif file_extension == "pptx":
#                 extracted_text = extract_text_from_pptx(tmp_path)
#             elif file_extension == "docx":
#                 extracted_text = extract_text_from_docx(tmp_path)
#             elif file_extension in ["png", "jpg", "jpeg"]:
#                 extracted_text = extract_text_from_image(tmp_path)
#             else:
#                 st.error(f"Unsupported file type: {file.name}")
#                 continue

#             if extracted_text:
#                 docs.append(LangchainDocument(page_content=str(extracted_text)))
#         except Exception as e:
#             st.error(f"Error processing {file.name}: {str(e)}")
#         finally:
#             os.unlink(tmp_path)

#     if docs:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=1000,
#             chunk_overlap=200,
#             separators=["\n\n", "\n", " ", ""]
#         )
#         splits = text_splitter.split_documents(docs)

#         try:
#             PineconeVectorStore.from_documents(
#                 splits,
#                 gemini_embeddings,
#                 index_name=PINECONE_INDEX_NAME
#             )
#             st.success(f"✅ Processed {len(splits)} chunks from {len(uploaded_files)} documents")
#             st.session_state.processed = True
#             return True
#         except Exception as e:
#             st.error(f"Failed to store documents: {str(e)}")
#             return False
#     else:
#         st.warning("No valid content was extracted from the documents")
#         return False

# with st.sidebar:
#     st.header("📂 Document Upload")
#     uploaded_files = st.file_uploader("Select files", type=["pdf", "pptx", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
#     if st.button("Process Documents", disabled=not uploaded_files):
#         with st.spinner("Processing documents..."):
#             process_documents(uploaded_files)

# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.markdown(message["content"])

# if prompt := st.chat_input("Ask about your documents..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.markdown(prompt)
#     with st.chat_message("assistant"):
#         if not st.session_state.get("processed", False):
#             st.warning("Please upload and process documents first")
#             st.session_state.messages.append({"role": "assistant", "content": "No documents have been processed yet. Please upload documents first."})
#         else:
#             with st.spinner("Gemini is thinking..."):
#                 try:
#                     if "vectorstore" not in st.session_state:
#                         st.session_state.vectorstore = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=gemini_embeddings)
#                     retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
#                     template = """You are an expert AI assistant powered by Gemini. Answer based only on the following context:\n{context}\nQuestion: {question}\nAnswer:"""
#                     prompt_template = ChatPromptTemplate.from_template(template)
#                     rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt_template | gemini_llm | StrOutputParser())
#                     response = rag_chain.invoke(prompt)
#                     st.markdown(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#                 except Exception as e:
#                     st.error(f"Gemini encountered an error: {str(e)}")

# FINAL CODE FOR WEB SCRAPPING
import os
import tempfile
import streamlit as st
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image
from pptx import Presentation
from docx import Document
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LangchainDocument

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
PINECONE_INDEX_NAME = "rag-practice"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
# if PINECONE_INDEX_NAME not in pc.list_indexes().names():
#     pc.create_index(
#         name=PINECONE_INDEX_NAME,
#         dimension=768,
#         metric="cosine",
#         spec=ServerlessSpec(cloud="aws", region="us-east-1")
#     )

# Check if the index exists, delete it if found
if PINECONE_INDEX_NAME in pc.list_indexes().names():
    print(f"Deleting existing Pinecone index: {PINECONE_INDEX_NAME} ...")
    pc.delete_index(PINECONE_INDEX_NAME)
    print("Index deleted successfully.")

# Check and create index if it doesn't exist
if PINECONE_INDEX_NAME not in pc.list_indexes().names():
    print("Creating Pinecone index...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        dimension=768,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Pinecone index setup complete.")

# Initialize Gemini components
gemini_embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=GEMINI_API_KEY
)

gemini_llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=GEMINI_API_KEY,
    temperature=0.3
)

def scrape_website(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = "\n".join([p.get_text() for p in soup.find_all("p")])
        return text.strip() if text else "No relevant text found on the webpage."
    except Exception as e:
        return f"Error scraping website: {str(e)}"

# Streamlit UI
st.set_page_config(page_title="Gemini RAG System", page_icon="🤖")
st.title("🤖 Gemini-Powered Document & Web Scraping Chat")
st.caption("Upload documents or scrape text from websites to chat with Gemini AI")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False

def process_text_data(text_data, source):
    if text_data:
        doc = LangchainDocument(page_content=text_data, metadata={"source": source})
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        splits = text_splitter.split_documents([doc])
        try:
            PineconeVectorStore.from_documents(
                splits,
                gemini_embeddings,
                index_name=PINECONE_INDEX_NAME
            )
            st.success(f"✅ Processed {len(splits)} chunks from {source}")
            st.session_state.processed = True
        except Exception as e:
            st.error(f"Failed to store documents: {str(e)}")

def process_documents(uploaded_files):
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        
        try:
            file_extension = file.name.split('.')[-1].lower()
            extracted_text = ""
            if file_extension == "pdf":
                loader = PyPDFLoader(tmp_path)
                pages = loader.load_and_split()
                extracted_text = "\n".join([page.page_content for page in pages])
            elif file_extension == "pptx":
                prs = Presentation(tmp_path)
                extracted_text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
            elif file_extension == "docx":
                doc = Document(tmp_path)
                extracted_text = "\n".join([para.text.strip() for para in doc.paragraphs if para.text.strip()])
            elif file_extension in ["png", "jpg", "jpeg"]:
                extracted_text = pytesseract.image_to_string(Image.open(tmp_path))
            else:
                st.error(f"Unsupported file type: {file.name}")
                continue
            
            process_text_data(extracted_text, file.name)
        except Exception as e:
            st.error(f"Error processing {file.name}: {str(e)}")
        finally:
            os.unlink(tmp_path)

with st.sidebar:
    st.header("📂 Upload Documents or Enter Website URL")
    uploaded_files = st.file_uploader("Select files", type=["pdf", "pptx", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
    url = st.text_input("Enter Website URL for Scraping")
    if st.button("Process Documents", disabled=not uploaded_files):
        with st.spinner("Processing documents..."):
            process_documents(uploaded_files)
    if st.button("Scrape Website", disabled=not url):
        with st.spinner("Scraping website..."):
            text_data = scrape_website(url)
            process_text_data(text_data, url)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about your documents or scraped text..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not st.session_state.get("processed", False):
            st.warning("Please upload and process documents or scrape a website first")
        else:
            with st.spinner("Gemini is thinking..."):
                retriever = PineconeVectorStore(index_name=PINECONE_INDEX_NAME, embedding=gemini_embeddings).as_retriever(search_kwargs={"k": 4})
                template = """You are an expert AI assistant powered by Gemini. Answer based only on the following context:\n{context}\nQuestion: {question}\nAnswer:"""
                prompt_template = ChatPromptTemplate.from_template(template)
                rag_chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt_template | gemini_llm | StrOutputParser())
                response = rag_chain.invoke(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
