from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file
from werkzeug.utils import secure_filename
import os
import re
import json
import io
from datetime import datetime
from bs4 import BeautifulSoup

from config import pc, gemini_embeddings, pinecone_index, PINECONE_INDEX_NAME, gemini_llm, GOOGLE_API_KEY, GOOGLE_CSE_ID, web_search_tool, search
from chat_export import export_chat_history, generate_chat_pdf
from document_processing import process_documents, scrape_website, process_text_data
from vector_store import initialize_pinecone, clear_index, get_vector_store
from rag_techniques import reciprocal_rank_fusion, generate_query_variations, generate_reasoning_steps, generate_sub_questions
from web_scrapping import search_google
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings


# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Pinecone index
initialize_pinecone()

@app.before_request
def initialize_session():
    if 'messages' not in session:
        session['messages'] = []
    if 'processed' not in session:
        session['processed'] = False
    if 'rich_notes' not in session:
        session['rich_notes'] = ""
    if 'show_notes' not in session:
        session['show_notes'] = False

@app.route('/')
def index():
    return render_template('index.html', GOOGLE_API_KEY=GOOGLE_API_KEY, GOOGLE_CSE_ID=GOOGLE_CSE_ID)

#embeddings
embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_embedding(text: str) -> list[float]:
    return embedding.embed_query(text)

@app.route('/process-files', methods=['POST'])
def process_files_route():
    if 'files[]' not in request.files:
        flash('No files provided', 'error')
        return redirect(request.url)
    
    files = request.files.getlist('files[]')
    if not files[0].filename:
        flash('No files selected', 'error')
        return redirect(request.url)
    
    uploaded_files = []
    for file in files:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        uploaded_files.append(file_path)
    
    try:
        process_documents(uploaded_files)
        session['processed'] = True
        flash(f'Successfully processed {len(uploaded_files)} files', 'success')
    except Exception as e:
        flash(f'Error processing files: {str(e)}', 'error')
    
    # Clean up files after processing
    for file_path in uploaded_files:
        if os.path.exists(file_path):
            os.remove(file_path)
    
    return redirect(url_for('index'))

@app.route('/scrape-site', methods=['POST'])
def scrape_site_route():
    url = request.form.get('url')
    if not url:
        flash('No URL provided', 'error')
        return redirect(url_for('index'))
    
    try:
        text_content = scrape_website(url)
        process_text_data(text_content, url)
        session['processed'] = True
        flash('Website successfully scraped and processed', 'success')
    except Exception as e:
        flash(f'Error scraping website: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/clear-data', methods=['POST'])
def clear_data_route():
    try:
        clear_index()
        session['processed'] = False
        flash('All document data cleared successfully', 'success')
    except Exception as e:
        flash(f'Error clearing data: {str(e)}', 'error')
    
    return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    prompt = request.form.get('prompt')
    use_web_search = request.form.get('use_web_search') == 'on'
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'})
    
    # Add user message to session
    messages = session.get('messages', [])
    messages.append({"role": "user", "content": prompt})
    session['messages'] = messages
    
    # Generate response
    if not session.get('processed', False) and not use_web_search:
        response = "Please process or scrape some documents first."
    else:
        try:
            document_answer_found = False
            response = ""
            
            # First try to get answer from documents if processed
            if session.get('processed', False):
                vector_store = get_vector_store()
                retriever = vector_store.as_retriever(search_kwargs={"k": 4})
                queries = generate_query_variations(prompt) + generate_sub_questions(prompt) + generate_reasoning_steps(prompt)
                all_docs = [retriever.invoke(q) for q in queries]
                fused_docs = reciprocal_rank_fusion(all_docs)
                
                if fused_docs:
                    from langchain_core.prompts import ChatPromptTemplate
                    from langchain_core.runnables import RunnablePassthrough, RunnableLambda
                    from langchain_core.output_parsers import StrOutputParser
                    
                    context = "\n\n".join([f"📄 Source {i+1}:\n{doc.page_content}" for i, doc in enumerate(fused_docs)])
                    template = """You are a helpful AI assistant. Answer the question based only on the context below.
                        - If you cannot answer from the context, say EXACTLY: "I couldn't find this information in the documents."
                        - Use bullet points for lists, steps, or comparisons.
                        - Do NOT include any headings or intro text.
                        - Stick strictly to the provided context.
                        Context:
                        {context}

                        Question: {question}

                        Answer:"""
                    prompt_template = ChatPromptTemplate.from_template(template)
                    rag_chain = (
                        {"context": RunnableLambda(lambda x: context), "question": RunnablePassthrough()}
                        | prompt_template
                        | gemini_llm
                        | StrOutputParser()
                    )
                    answer = rag_chain.invoke(prompt)
                    
                    # Check if the answer indicates missing information
                    if "couldn't find this information in the documents" not in answer.lower():
                        response = f"📄 Document Answer:\n\n{answer}"
                        document_answer_found = True
            
            # Handle cases where answer wasn't found
            if not document_answer_found:
                if use_web_search:
                    raw_web_results, formatted_web_results = search_google(prompt)
                    if raw_web_results and "error" not in str(formatted_web_results).lower():
                        # Process web results with Gemini
                        from langchain_core.prompts import ChatPromptTemplate
                        from langchain_core.runnables import RunnablePassthrough
                        from langchain_core.output_parsers import StrOutputParser
                        
                        web_template = """Analyze these web search results and provide a comprehensive answer:
                        - Combine information from multiple sources if needed
                        - Always include source references like [1], [2] with corresponding links
                        - Keep the answer concise but informative
                        
                        Search Results:
                        {results}
                        
                        Question: {question}
                        
                        Answer with inline citations:"""
                        web_prompt = ChatPromptTemplate.from_template(web_template)
                        web_chain = (
                            {"results": RunnablePassthrough(), "question": RunnablePassthrough()}
                            | web_prompt
                            | gemini_llm
                            | StrOutputParser()
                        )
                        
                        processed_results = web_chain.invoke({
                            "results": formatted_web_results, 
                            "question": prompt
                        })
                        
                        # Add numbered source list
                        sources_section = "\n\n### References:\n" + "\n".join(
                            f"{i+1}. [{res.get('title', 'Source')}]({res.get('link', '')})"
                            for i, res in enumerate(raw_web_results)
                        )
                        
                        response = (
                            f"🌐 Web Answer (since not found in documents):\n\n"
                            f"{processed_results}\n"
                            f"{sources_section}"
                        )
                    else:
                        response = "⚠️ Couldn't find relevant information in documents or through web search."
                else:
                    response = (
                        "I couldn't find this information in your documents. "
                        "You can enable web search to search online for answers."
                    )
        
        except Exception as e:
            response = f"Error generating answer: {str(e)}"
    
    # Add assistant message to session
    messages.append({"role": "assistant", "content": response})
    session['messages'] = messages
    
    return jsonify({
        'response': response,
        'messages': messages
    })

ANCHORS_FILE = "memory_store.json"

def load_anchors():
    if not os.path.exists(ANCHORS_FILE):
        return []
    with open(ANCHORS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_anchors(anchors):
    with open(ANCHORS_FILE, "w", encoding="utf-8") as f:
        json.dump(anchors, f, indent=2)
@app.route('/generate-anchor', methods=['POST'])
def generate_anchor():
    data = request.json
    user_context = data.get('context')

    prompt_text = f"""
                    Given the following concept, return a memory anchor in JSON format with these fields:
                    - "term"
                    - "summary"
                    - "mnemonic"
                    - "example"

                    Concept:
                    \"\"\"{user_context}\"\"\"

                    Respond ONLY in JSON.
                """

    try:
        response = gemini_llm.invoke([HumanMessage(content=prompt_text)])
        print("🔍 Gemini raw output:", response)
    except Exception as e:
        return jsonify({"status": "error", "message": f"Gemini call failed: {str(e)}"}), 500

    # Try to extract the actual content
    try:
        raw_output = response.content if hasattr(response, "content") else str(response)
    # Clean markdown code block wrappers
        cleaned_output = re.sub(r"^```json\s*|```$", "", raw_output, flags=re.MULTILINE).strip()
        anchor = json.loads(cleaned_output)
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": f"Invalid JSON from Gemini: {str(e)}",
            "raw_response": raw_output
        }), 500

    # Embed and upsert to Pinecone
    embed = get_embedding(anchor['summary'])
    pinecone_index.upsert([(f"anchor-{anchor['term']}", embed, anchor)])

    # Save locally
    anchors = load_anchors()
    anchor["id"] = len(anchors) + 1
    anchors.append(anchor)
    save_anchors(anchors)

    return jsonify({"status": "success", "anchor": anchor})

@app.route('/save-notes', methods=['POST'])
def save_notes():
    notes_content = request.form.get('notes_content', '')
    session['rich_notes'] = notes_content
    return jsonify({'success': True})

@app.route('/toggle-notes', methods=['POST'])
def toggle_notes():
    session['show_notes'] = not session.get('show_notes', False)
    return jsonify({'show_notes': session.get('show_notes', False)})

@app.route('/export-chat', methods=['POST'])
def export_chat():
    export_format = request.form.get('export_format', 'json')
    messages = session.get('messages', [])
    
    if not messages:
        flash('No chat history to export', 'warning')
        return redirect(url_for('index'))
    
    if export_format == 'json':
        data = json.dumps(messages, indent=2)
        file_name = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        return send_file(
            io.BytesIO(data.encode('utf-8')),
            mimetype='application/json',
            as_attachment=True,
            download_name=file_name
        )
    
    elif export_format == 'txt':
        text = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in messages])
        file_name = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        return send_file(
            io.BytesIO(text.encode('utf-8')),
            mimetype='text/plain',
            as_attachment=True,
            download_name=file_name
        )
    
    elif export_format == 'pdf':
        pdf_buffer = generate_chat_pdf(messages)
        file_name = f"chat_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=file_name
        )
    
    flash('Invalid export format', 'error')
    return redirect(url_for('index'))

@app.route('/export-notes-pdf', methods=['POST'])
def export_notes_pdf():
    rich_notes = session.get('rich_notes', '')
    
    if not rich_notes:
        flash('No notes to export', 'warning')
        return redirect(url_for('index'))
    
    from fpdf import FPDF
    
    # Parse HTML from rich notes
    soup = BeautifulSoup(rich_notes, "html.parser")
    clean_text = soup.get_text()
    
    # Create PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.set_font("Arial", size=12)
    
    for line in clean_text.split("\n"):
        pdf.multi_cell(0, 10, line)
    
    pdf_buffer = io.BytesIO()
    pdf_string = pdf.output(dest='S')
    pdf_buffer = io.BytesIO()
    pdf_buffer.write(pdf_string)
    pdf_buffer.seek(0)
    
    file_name = f"notes_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
    return send_file(
        pdf_buffer,
        mimetype='application/pdf',
        as_attachment=True,
        download_name=file_name
    )

@app.route('/messages')
def get_messages():
    return jsonify({'messages': session.get('messages', [])})

@app.route('/notes')
def get_notes():
    return jsonify({
        'notes': session.get('rich_notes', ''),
        'show_notes': session.get('show_notes', False)
    })

@app.route('/debug')
def debug():
    # Only accessible in debug mode for security
    if not app.debug:
        return "Debug mode is disabled", 403
    
    # Check environment variables
    api_status = {
        "PINECONE_API_KEY": bool(PINECONE_API_KEY),
        "GEMINI_API_KEY": bool(GEMINI_API_KEY),
        "GOOGLE_API_KEY": bool(GOOGLE_API_KEY),
        "GOOGLE_CSE_ID": bool(GOOGLE_CSE_ID),
        "Web Search Available": bool(GOOGLE_API_KEY and GOOGLE_CSE_ID),
        "Search Tool Initialized": bool(web_search_tool)
    }
    
    return jsonify(api_status)

if __name__ == '__main__':
    app.run(debug=True) 