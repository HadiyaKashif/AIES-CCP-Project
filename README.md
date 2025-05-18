# SmartSage – AI-Powered RAG Assistant with Productivity & Learning Features (Flask)

SmartSage is a Flask-based Retrieval-Augmented Generation (RAG) system enhanced with Google's Gemini APIs and Pinecone. It's not just a chatbot — it's a complete intelligent assistant designed to boost learning, productivity, and cognitive engagement.

🔗 **[Watch Demo Video]https://youtu.be/g0nHoZTmlqg?si=hDim9MOgf4QGmKWe**

---

## Key Features

### **AI-Powered Retrieval & Generation**
- Upload documents: **PDF, DOCX, PPTX, images**
- Website scraping via URL input
- Advanced RAG techniques:
  - **RAG Fusion**
  - **Multi-Query Expansion**
  - **Query Decomposition**
- Web search fallback if no answer is found in documents, with **source link references**

### **Smart Interaction**
- **Text and voice-based chat**
- Integrated **Gemini Pro and Gemini Flash** APIs for contextual and fluent answers
- **Speech-to-text** and **text-to-speech** for accessibility

### **Knowledge & Learning Tools**
- **Flashcard Generator** from document or conversation content
- **Memory Anchor System**:
  - Input any concept
  - Generates a unique **anchor**, **summary**, **mnemonic**, and **example** for retention

### **Personal Workspace**
- Rich **Notes Editor** for writing or pasting insights
- Export notes to **TXT or PDF**

### **Focus & Wellness Suite**
- **Pomodoro Timer** (25-minute focus sessions + 5-minute breaks)
- Tracks how many sessions were used per task
- Personalized **motivational quotes** based on task type (e.g., reading, coding)
- **Fatigue Detection System**:
  - Uses webcam to detect **yawning or eye closure**
  - Plays an alert and recommends breaks if fatigue is high

### **Task Management**
- Add, update, complete, and prioritize tasks
- Tracks **task difficulty based on fatigue signals**
- Generates task-based motivational insights

### **Utilities**
- **Chat export** in JSON, TXT, or PDF
- **Dark/light mode**, responsive design with Bootstrap

---

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
- Tesseract / EasyOCR – OCR from images
- OpenCV / Mediapipe – Fatigue detection from webcam
- PyPDF2, python-docx, pdf2image – Document parsing
