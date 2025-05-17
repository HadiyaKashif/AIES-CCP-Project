from config import pc, gemini_embeddings
from langchain_pinecone import PineconeVectorStore
import streamlit as st
from pinecone import Pinecone, ServerlessSpec
import time

PINECONE_INDEX_NAME = "rag-advanced"

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

def get_vector_store():
    if "vector_store" not in st.session_state or st.session_state.vector_store is None:
        st.session_state.vector_store = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME,
            embedding=gemini_embeddings,
            namespace="default"
        )
    return st.session_state.vector_store