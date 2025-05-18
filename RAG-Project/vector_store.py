from config import pc, gemini_embeddings
from langchain_pinecone import PineconeVectorStore
from flask import flash, session
from pinecone import Pinecone, ServerlessSpec
import time

PINECONE_INDEX_NAME = "rag-advanced"

def initialize_pinecone():
    try:
        existing_indexes = pc.list_indexes().names()
        
        if PINECONE_INDEX_NAME not in existing_indexes:
            print("Creating new Pinecone index... (may take 1-2 minutes)")
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print("Waiting for index to be ready...")
            while True:
                try:
                    desc = pc.describe_index(PINECONE_INDEX_NAME)
                    if desc.status['ready']:
                        break
                    time.sleep(5)
                except Exception as e:
                    time.sleep(5)
            print("Index created and ready to use!")
        else:
            print("Checking index status...")
            while True:
                try:
                    desc = pc.describe_index(PINECONE_INDEX_NAME)
                    if desc.status['ready']:
                        break
                    time.sleep(5)
                except Exception as e:
                    time.sleep(5)
            print(f"Using existing index: {PINECONE_INDEX_NAME}")
    except Exception as e:
        print(f"Pinecone initialization failed: {str(e)}")
        raise

def clear_index():
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        try:
            index.delete(delete_all=True, namespace="")
        except Exception as ns_error:
            pass
        
        if 'vector_store' in session:
            session.pop('vector_store')
        
        session['processed'] = False
        flash("All documents cleared successfully!", "success")
    except Exception as e:
        flash(f"Error clearing index: {str(e)}", "error")

def get_vector_store():
    # We can't store the actual vector_store object in the session
    # So we'll create a new one each time
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=gemini_embeddings,
        namespace="default"
    )