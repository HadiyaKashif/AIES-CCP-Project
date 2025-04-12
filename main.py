import os
import io
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
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document as LangchainDocument
import time
from typing import List, Dict, Any
from fpdf import FPDF
import time
from collections import defaultdict
from streamlit_quill import st_quill

# MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(page_title="Advanced Gemini RAG System", page_icon="🤖")

# Load environment variables
load_dotenv()

# Configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY") or st.secrets["PINECONE_API_KEY"]
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or st.secrets["GEMINI_API_KEY"]
PINECONE_INDEX_NAME = "rag-advanced"

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

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

def initialize_pinecone():
    try:
        existing_indexes = pc.list_indexes().names()
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            st.warning("Creating new Pinecone index... (may take 1-2 minutes)")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            
            # Wait for index to be ready
            with st.spinner("Waiting for index to be ready..."):
                while True:
                    try:
                        desc = pc.describe_index(PINECONE_INDEX_NAME)
                        if desc.status['ready']:
                            break
                        time.sleep(5)
                    except Exception as e:
                        time.sleep(5)
            
            st.success("Index created and ready to use!")
        else:
            # Check readiness for existing index
            with st.spinner("Checking index status..."):
                while True:
                    try:
                        desc = pc.describe_index(PINECONE_INDEX_NAME)
                        if desc.status['ready']:
                            break
                        time.sleep(5)
                    except Exception as e:
                        time.sleep(5)

            st.info(f"Using existing index: {PINECONE_INDEX_NAME}")

            if "cleared_on_startup" not in st.session_state:
                clear_index()
                st.session_state.cleared_on_startup = True
            
    except Exception as e:
        st.error(f"Pinecone initialization failed: {str(e)}")
        st.stop()

def clear_index():
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        try:
            index.delete(delete_all=True, namespace="")
        except Exception as ns_error:
            pass
        st.session_state.vector_store = None
        st.session_state.processed = False
        st.success("All documents cleared successfully!")
    except Exception as e:
        st.error(f"Error clearing index: {str(e)}")

def scrape_website(url):
    try:
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        text = "\n".join([p.get_text() for p in soup.find_all("p")])
        return text.strip() if text else "No relevant text found on the webpage."
    except Exception as e:
        return f"Error scraping website: {str(e)}"

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
            if st.session_state.vector_store is None:
                st.session_state.vector_store = PineconeVectorStore.from_documents(
                    splits,
                    gemini_embeddings,
                    index_name=PINECONE_INDEX_NAME
                )
            else:
                st.session_state.vector_store.add_documents(splits)
            
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

# Advanced RAG Techniques
def generate_query_variations(original_query: str) -> List[str]:
    """Generate multiple query variations for RAG-Fusion"""
    prompt = f"""Generate 3 different rephrasings of this query while maintaining the core meaning:
    Original: {original_query}
    
    Variations:
    1. """
    response = gemini_llm.invoke(prompt)
    variations = [original_query] + [v.strip() for v in response.content.split("\n") if v.strip()]
    return variations[:4]  # Limit to 4 variations (including original)

def reciprocal_rank_fusion(results: List[List[LangchainDocument]], k=60) -> List[LangchainDocument]:
    """Fuse multiple retrieval results using RRF"""
    fused_scores = defaultdict(float)
    
    for docs in results:
        for rank, doc in enumerate(docs, 1):
            doc_id = doc.page_content  # Simple ID for demo
            fused_scores[doc_id] += 1.0 / (rank + k)
    
    # Sort by fused score
    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Reconstruct documents (simplified - in practice you'd keep full doc objects)
    return [LangchainDocument(page_content=doc_id, metadata={}) for doc_id, _ in sorted_docs[:4]]

def generate_sub_questions(question: str) -> List[str]:
    """Break down complex questions into sub-questions for Multi-Query"""
    prompt = f"""Break this complex question into 2-3 standalone sub-questions:
    Original: {question}
    
    Sub-questions:
    1. """
    response = gemini_llm.invoke(prompt)
    sub_questions = [q.strip() for q in response.content.split("\n") if q.strip()]
    return sub_questions[:3]  # Limit to 3 sub-questions

def generate_reasoning_steps(question: str) -> List[str]:
    """Generate step-by-step reasoning plan for Decomposition"""
    prompt = f"""To answer this question, identify the required information retrieval steps:
    Question: {question}
    
    Steps:
    1. First find: """
    response = gemini_llm.invoke(prompt)
    steps = [s.strip() for s in response.content.split("\n") if s.strip()]
    return steps[:3]  # Limit to 3 steps

# UI Setup
st.title("🤖 Advanced Gemini RAG System")
st.caption("Now with RAG-Fusion, Multi-Query, and Decomposition capabilities")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed" not in st.session_state:
    st.session_state.processed = False
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "rich_notes" not in st.session_state:
    st.session_state.rich_notes = ""

# Initialize Pinecone
initialize_pinecone()

# Sidebar
with st.sidebar:
    st.header("📂 Data Management")
    uploaded_files = st.file_uploader("Upload files", type=["pdf", "pptx", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)
    url = st.text_input("Or enter website URL")
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Process Files", disabled=not uploaded_files):
            with st.spinner("Processing..."):
                process_documents(uploaded_files)
    with col2:
        if st.button("Scrape Site", disabled=not url):
            with st.spinner("Scraping..."):
                text_data = scrape_website(url)
                process_text_data(text_data, url)
    
    if st.button("🧹 Clear All Data", type="secondary"):
        clear_index()

# Chat Interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


if prompt := st.chat_input("Ask about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        if not st.session_state.get("processed", False):
            st.warning("Please process documents first")
        else:
            with st.spinner("Applying advanced RAG techniques (Fusion, Multi-Query, Decomposition)..."):
                try:
                    retriever = st.session_state.vector_store.as_retriever(search_kwargs={"k": 4})
                    
                    # 1. RAG-Fusion - Query Variations
                    variations = generate_query_variations(prompt)
                    fusion_results = []
                    for v in variations:
                        docs = retriever.invoke(v)
                        fusion_results.extend(docs)
                    
                    # 2. Multi-Query - Sub-Questions
                    sub_questions = generate_sub_questions(prompt)
                    multi_query_docs = []
                    for q in sub_questions:
                        docs = retriever.invoke(q)
                        multi_query_docs.extend(docs)
                    
                    # 3. Decomposition - Step-by-Step
                    steps = generate_reasoning_steps(prompt)
                    step_results = []
                    for step in steps:
                        docs = retriever.invoke(step)
                        step_results.extend(docs)
                    
                    # Combine all retrieved documents and remove duplicates
                    all_docs = fusion_results + multi_query_docs + step_results
                    unique_docs = {doc.page_content: doc for doc in all_docs}.values()
                    
                    # Create context from all techniques
                    context = "\n\n".join([
                        f"📄 Source {i+1} (from {doc.metadata.get('source', 'unknown')}):\n{doc.page_content}"
                        for i, doc in enumerate(unique_docs)
                    ])
                    
                    # Generate final response using all context
                
                    template = """You are a helpful AI assistant. Answer the question based only on the context provided below. 

                    - Use bullet points if the answer involves lists, steps, comparisons, or grouped facts.
                    - Do NOT include headings like "Answer", "Summary", or "Comprehensive Answer".
                    - Do NOT include an introduction, explanation of the process, or a conclusion.
                    - Stick to the facts in the context.

                    Context:
                    {context}

                    Question: {question}

                    Answer:"""
           
                    prompt_template = ChatPromptTemplate.from_template(template)
                    
                    rag_chain = (
                        {"context": RunnableLambda(lambda x: context), 
                         "question": RunnablePassthrough()}
                        | prompt_template 
                        | gemini_llm 
                        | StrOutputParser()
                    )
                    
                    
                    process_explanation = """
                    The answer below was generated after thoroughly analyzing and combining relevant information from your uploaded content.
                    """
                    
                    response = rag_chain.invoke(prompt)
                    full_response = f"{process_explanation}\n\n{response}"
                    
                    st.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                except Exception as e:
                    st.error(f"Error generating answer: {str(e)}")

# Notes Section
st.markdown("## 📝 Notes")

editor = st_quill(
    value=st.session_state.rich_notes,
    placeholder="Write your notes here...",
    html=True
)

if editor:
    st.session_state.rich_notes = editor

with st.expander("🔍 Preview Notes", expanded=False):
    st.markdown("### Rendered Notes:")
    st.markdown(st.session_state.rich_notes, unsafe_allow_html=True)


if st.button("💾 Download Notes as PDF"):
    if st.session_state.rich_notes:
        # Step 1: Clean HTML to plain text
        soup = BeautifulSoup(st.session_state.rich_notes, "html.parser")
        clean_text = soup.get_text()

        # Step 2: Generate PDF
        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.set_font("Arial", size=12)

        for line in clean_text.split("\n"):
            pdf.multi_cell(0, 10, line)

        # Step 3: Use BytesIO to stream PDF for download
        pdf_buffer = io.BytesIO()
        pdf.output(pdf_buffer)
        pdf_buffer.seek(0)

        # Step 4: Download button
        st.download_button(
            label="📄 Confirm Download",
            data=pdf_buffer,
            file_name="my_notes.pdf",
            mime="application/pdf"
        )
    else:
        st.warning("No notes to download!")
