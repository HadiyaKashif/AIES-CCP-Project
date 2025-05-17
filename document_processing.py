import os
import tempfile
import pytesseract
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document as LangchainDocument
from pptx import Presentation
from docx import Document
from PIL import Image
import requests
from bs4 import BeautifulSoup
import streamlit as st
from vector_store import get_vector_store

def scrape_website(url):
    try:
        res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(res.text, "html.parser")
        return "\n".join([p.get_text() for p in soup.find_all("p")]) or "No relevant text found."
    except Exception as e:
        return f"Error scraping: {str(e)}"

def process_text_data(text_data, source):
    if text_data:
        doc = LangchainDocument(page_content=text_data, metadata={"source": source})
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents([doc])
        try:
            vector_store = get_vector_store()
            vector_store.add_documents(splits)
            st.success(f"Processed {len(splits)} chunks from {source}")
            st.session_state.processed = True
        except Exception as e:
            st.error(f"Vector store error: {str(e)}")

def process_documents(uploaded_files):
    for file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f".{file.name.split('.')[-1]}") as tmp:
            tmp.write(file.getvalue())
            tmp_path = tmp.name
        try:
            ext = file.name.split('.')[-1].lower()
            text = ""
            if ext == "pdf":
                loader = PyPDFLoader(tmp_path)
                pages = loader.load_and_split()
                text = "\n".join([p.page_content for p in pages])
            elif ext == "pptx":
                prs = Presentation(tmp_path)
                text = "\n".join([shape.text for slide in prs.slides for shape in slide.shapes if hasattr(shape, "text")])
            elif ext == "docx":
                doc = Document(tmp_path)
                text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
            elif ext in ["png", "jpg", "jpeg"]:
                text = pytesseract.image_to_string(Image.open(tmp_path))
            else:
                st.error(f"Unsupported file type: {file.name}")
                continue
            process_text_data(text, file.name)
        except Exception as e:
            st.error(f"Processing error for {file.name}: {str(e)}")
        finally:
            os.unlink(tmp_path)