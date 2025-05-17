# Advanced Gemini RAG System - Flask Version

This is a Flask web application for a Retrieval-Augmented Generation (RAG) system using Google's Gemini model. The application allows users to upload documents, scrape websites, and ask questions about the processed content. It also features web search fallback, notes system, and chat export functionality.

## Features

- Document processing (PDF, DOCX, PPTX, images)
- Website scraping and processing
- Advanced RAG techniques:
  - RAG-Fusion
  - Multi-Query
  - Question Decomposition
- Web search fallback (requires Google API credentials)
- Notes system with rich text editing and PDF export
- Chat history export (JSON, TXT, PDF)

## Setup Instructions

1. Create a `.env` file in the root directory with the following variables:
   ```
   PINECONE_API_KEY=your_pinecone_api_key
   GEMINI_API_KEY=your_gemini_api_key
   GOOGLE_API_KEY=your_google_api_key (optional, for web search)
   GOOGLE_CSE_ID=your_google_cse_id (optional, for web search)
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python app.py
   ```

4. Open a web browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

## Usage

1. Upload documents or enter a website URL in the sidebar
2. Process the files or scrape the website
3. Ask questions in the chat interface
4. Enable web search if needed for questions not found in documents
5. Use the Notes system for taking notes during your research
6. Export chat history or notes as needed

## Technologies Used

- Flask: Web framework
- Pinecone: Vector database for embeddings
- Google Gemini: Language model for embeddings and chat
- LangChain: Framework for RAG implementation
- Various document processing libraries (PyPDF, python-docx, etc.)
- Bootstrap & jQuery: Frontend
