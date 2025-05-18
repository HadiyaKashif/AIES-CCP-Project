from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file, Response
from werkzeug.utils import secure_filename
import os
import json
import io
from datetime import datetime
from bs4 import BeautifulSoup
import time
import cv2
import mediapipe as mp
import math
import pygame
import numpy as np
import random
import threading
from study_guardian import StudyGuardian
from config import gemini_llm, GOOGLE_API_KEY, GOOGLE_CSE_ID, web_search_tool, PINECONE_API_KEY, GEMINI_API_KEY
from chat_export import export_chat_history, generate_chat_pdf
from document_processing import process_documents, scrape_website, process_text_data
from vector_store import initialize_pinecone, clear_index, get_vector_store
from rag_techniques import reciprocal_rank_fusion, generate_query_variations, generate_reasoning_steps, generate_sub_questions
from web_scrapping import search_google
from components.flashcards import generate_flashcards_from_text

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB max upload size
app.config['SESSION_COOKIE_SAMESITE'] = 'Strict'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_PERMANENT'] = False

# Set session type to filesystem to prevent browser persistence
from flask_session import Session
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

# Create upload folder if it doesn't exist
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Initialize Pinecone index
initialize_pinecone()

# Initialize StudyGuardian
study_guardian = StudyGuardian("static/alert.wav")

# Initialize MediaPipe and other study buddy variables
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Constants
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14]

# Add these constants after the MediaPipe constants and before the global variables

# Task types and their specific motivational quotes
TASK_TYPES = {
    "reading": "Reading",
    "writing": "Writing",
    "problem_solving": "Problem Solving",
    "memorization": "Memorization",
    "research": "Research"
}

TASK_SPECIFIC_QUOTES = {
    "reading": [
        "Take it one page at a time! 📚",
        "Every paragraph brings new knowledge! 🎯",
        "Reading is to the mind what exercise is to the body! 💪"
    ],
    "writing": [
        "Let your ideas flow freely! ✍️",
        "Your words have power! 🌟",
        "Write first, edit later! 📝"
    ],
    "problem_solving": [
        "Break it down, solve it up! 🧩",
        "Every problem is a new opportunity! 🎯",
        "Think outside the box! 💡"
    ],
    "memorization": [
        "Repetition is the mother of learning! 🔄",
        "Your brain is a sponge! 🧠",
        "Connect the dots! 🎯"
    ],
    "research": [
        "Explore the unknown! 🔍",
        "Every search leads to discovery! 🌟",
        "Connect the pieces of knowledge! 🧩"
    ]
}

# General motivational quotes
GENERAL_QUOTES = [
    "You've got this! Just a little more focus! 💪",
    "Take a deep breath, stay strong! 🌟",
    "Remember why you started! 🎯",
    "Your future self will thank you! 🚀",
    "Small steps lead to big achievements! 🎓",
    "Stay focused, stay amazing! ✨",
    "You're doing great! Keep pushing! 🌈",
    "Success is built one study session at a time! 📚",
    "Believe in yourself! You can do this! ⭐",
    "Every minute of focus counts! 🎯"
]

# Global variables for study buddy
camera = None
is_monitoring = False
eye_closed_time = 0
current_quote = ""
last_quote_time = 0
tiredness_counter = 0
drowsiness_patterns = []
tasks = {}
current_task = None
session_start_time = None
break_time = None

# Initialize pygame for alert sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("static/alert.wav")

# Add these helper functions after the global variables and before the routes

def distance(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

def get_ear(eye):
    v1 = distance(eye[1], eye[5])
    v2 = distance(eye[2], eye[4])
    h1 = distance(eye[0], eye[3])
    return (v1 + v2) / (2.0 * h1)

def play_alert():
    if not pygame.mixer.get_busy():
        alert_sound.play()

def get_smart_break_suggestion():
    if not drowsiness_patterns:
        return None
        
    current_time = time.time()
    time_since_last_break = current_time - drowsiness_patterns[-1]
    
    recent_events = [t for t in drowsiness_patterns if current_time - t <= 300]
    
    if len(recent_events) >= 3:
        return "High drowsiness detected. Consider taking a break now!"
    elif len(recent_events) >= 2:
        return "You're showing signs of tiredness. A break might be helpful soon."
    elif time_since_last_break > 1500:  # 25 minutes
        return "You've been studying for a while. Consider a short break soon."
    
    return None

def get_task_specific_quote(task_type=None):
    if task_type and task_type in TASK_SPECIFIC_QUOTES:
        quotes = TASK_SPECIFIC_QUOTES[task_type]
    else:
        quotes = GENERAL_QUOTES
        
    return random.choice(quotes)

def generate_frames():
    global camera, is_monitoring, eye_closed_time, tiredness_counter, current_quote, last_quote_time, drowsiness_patterns
    
    if camera is None:
        camera = cv2.VideoCapture(0)
    
    while is_monitoring:
        success, frame = camera.read()
        if not success:
            break
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            face = results.multi_face_landmarks[0]
            landmarks = face.landmark

            def get_coords(points):
                return [(int(landmarks[p].x * w), int(landmarks[p].y * h)) for p in points]

            left_eye = get_coords(LEFT_EYE)
            right_eye = get_coords(RIGHT_EYE)
            mouth = get_coords(MOUTH)

            left_ear = get_ear(left_eye)
            right_ear = get_ear(right_eye)
            ear = (left_ear + right_ear) / 2.0
            mar = distance(mouth[0], mouth[1]) / w

            current_time = time.time()
            if ear < 0.25:
                eye_closed_time += 1
                tiredness_counter += 1
                if eye_closed_time > 30:
                    cv2.putText(frame, "EYES CLOSED!", (30, 100),
                              cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    threading.Thread(target=play_alert).start()
                    drowsiness_patterns.append(current_time)
            else:
                eye_closed_time = 0

            if mar > 0.03:
                tiredness_counter += 1
                cv2.putText(frame, "YAWNING...", (30, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                drowsiness_patterns.append(current_time)

            if tiredness_counter > 50 and current_time - last_quote_time > 10:
                active_task_type = None
                for task, data in tasks.items():
                    if not data["completed"]:
                        active_task_type = data.get("type")
                        break
                
                current_quote = get_task_specific_quote(active_task_type)
                last_quote_time = current_time
                tiredness_counter = 0

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.before_request
def initialize_session():
    # Clear session on new browser session
    if 'initialized' not in session:
        session.clear()
        session['initialized'] = True
        session['messages'] = []
        session['processed'] = False
        session['rich_notes'] = ""
        session['show_notes'] = False
        session['flashcards'] = []
        session['flash_index'] = 0
        session['wrong_flashcards'] = []
        session['score'] = 0
        session['challenge_mode'] = False
        session['show_answer'] = False
    # else:
    #     if 'messages' not in session:
    #         session['messages'] = []
    #     if 'processed' not in session:
    #         session['processed'] = False
    #     if 'rich_notes' not in session:
    #         session['rich_notes'] = ""
    #     if 'show_notes' not in session:
    #         session['show_notes'] = False
    #     if 'flashcards' not in session:
    #         session['flashcards'] = []
    #     if 'flash_index' not in session:
    #         session['flash_index'] = 0
    #     if 'wrong_flashcards' not in session:
    #         session['wrong_flashcards'] = []
    #     if 'score' not in session:
    #         session['score'] = 0
    #     if 'challenge_mode' not in session:
    #         session['challenge_mode'] = False
    #     if 'show_answer' not in session:
    #         session['show_answer'] = False

@app.route('/clear-session', methods=['POST'])
def clear_session():
    """Force-clear the session for new browser sessions"""
    session.clear()
    session['initialized'] = True  # Reinitialize fresh session
    session['messages'] = []
    session['processed'] = False
    session['rich_notes'] = ""
    return jsonify({'success': True})

@app.route('/')
def index():
    return render_template('index.html', GOOGLE_API_KEY=GOOGLE_API_KEY, GOOGLE_CSE_ID=GOOGLE_CSE_ID)

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
    
    # Check if no relevant data was found
    no_relevant_data = False
    if "I couldn't find this information in your documents" in response or "Couldn't find relevant information" in response:
        no_relevant_data = True
    
    return jsonify({
        'response': response,
        'messages': messages,
        'no_relevant_data': no_relevant_data
    })

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

@app.route('/flashcards')
def flashcards():
    return render_template('flashcards.html')

@app.route('/generate-flashcards', methods=['POST'])
def generate_flashcards():
    if not session.get('processed', False):
        flash('Please process some documents first', 'error')
        return redirect(url_for('flashcards'))
    
    try:
        vector_store = get_vector_store()
        docs = vector_store.similarity_search("summary", k=20)
        full_text = " ".join([doc.page_content for doc in docs])
        
        flashcards = generate_flashcards_from_text(full_text)
        session['flashcards'] = flashcards
        session['flash_index'] = 0
        session['wrong_flashcards'] = []
        session['score'] = 0
        session['show_answer'] = False
        
        flash('Flashcards generated successfully!', 'success')
    except Exception as e:
        flash(f'Error generating flashcards: {str(e)}', 'error')
    
    return redirect(url_for('flashcards'))

@app.route('/next-flashcard', methods=['POST'])
def next_flashcard():
    action = request.form.get('action')  # 'knew' or 'didnt_know'
    
    if action == 'knew':
        session['score'] = session.get('score', 0) + 1
    elif action == 'didnt_know':
        current_card = session['flashcards'][session['flash_index']]
        session['wrong_flashcards'] = session.get('wrong_flashcards', []) + [current_card]
    
    session['flash_index'] = session.get('flash_index', 0) + 1
    session['show_answer'] = False
    
    # If done with all cards
    if session['flash_index'] >= len(session['flashcards']):
        if session.get('wrong_flashcards'):
            session['flashcards'] = session['wrong_flashcards']
            session['wrong_flashcards'] = []
            session['flash_index'] = 0
            flash('Repeating the cards you missed!', 'info')
        else:
            flash(f'Game complete! Final score: {session["score"]}', 'success')
            session['flashcards'] = []
            session['flash_index'] = 0
    
    return redirect(url_for('flashcards'))

@app.route('/toggle-answer', methods=['POST'])
def toggle_answer():
    session['show_answer'] = not session.get('show_answer', False)
    return redirect(url_for('flashcards'))

@app.route('/toggle-challenge-mode', methods=['POST'])
def toggle_challenge_mode():
    session['challenge_mode'] = not session.get('challenge_mode', False)
    return redirect(url_for('flashcards'))

# Add study buddy routes
@app.route('/study-monitor')
def study_monitor():
    return render_template('study_monitor.html', task_types=TASK_TYPES)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    global is_monitoring, eye_closed_time, session_start_time
    is_monitoring = True
    eye_closed_time = 0
    session_start_time = time.time()
    return jsonify({"status": "success"})

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global is_monitoring, camera, eye_closed_time, session_start_time
    is_monitoring = False
    eye_closed_time = 0
    session_start_time = None
    if camera is not None:
        camera.release()
        camera = None
    return jsonify({"status": "success"})

@app.route('/tasks', methods=['GET', 'POST'])
def manage_tasks():
    global tasks, current_task
    if request.method == 'POST':
        data = request.get_json()
        if 'task' in data and 'type' in data:
            task_name = data['task']
            tasks[task_name] = {
                "completed": False,
                "completed_in_session": None,
                "type": data['type'],
                "drowsiness_events": []
            }
            current_task = task_name
            return jsonify({"status": "success"})
    return jsonify(tasks)

@app.route('/update_task', methods=['POST'])
def update_task():
    global tasks
    data = request.get_json()
    if 'task' in data and 'completed' in data and 'session' in data:
        task = data['task']
        if task in tasks:
            tasks[task]["completed"] = data['completed']
            if data['completed']:
                tasks[task]["completed_in_session"] = data['session']
            else:
                tasks[task]["completed_in_session"] = None
            return jsonify({"status": "success"})
    return jsonify({"error": "Invalid task data"}), 400

@app.route('/get_quote')
def get_quote():
    active_task_type = None
    for task, data in tasks.items():
        if not data["completed"]:
            active_task_type = data.get("type")
            break
    
    quote = get_task_specific_quote(active_task_type)
    return jsonify({"quote": quote})

@app.route('/get_break_suggestion')
def get_break_suggestion():
    suggestion = get_smart_break_suggestion()
    return jsonify({"suggestion": suggestion})

@app.route('/task-history')
def get_task_history():
    task_history = []
    for task_name, task_data in tasks.items():
        history = {
            "task": task_name,
            "type": task_data["type"],
            "completed": task_data["completed"],
            "completed_in_session": task_data["completed_in_session"],
            "drowsiness_events": len(task_data["drowsiness_events"]),
            "completion_time": None  # We'll add this feature later
        }
        task_history.append(history)
    
    return jsonify({
        "tasks": task_history,
        "current_session": session_start_time is not None,
        "total_sessions": max([task["completed_in_session"] for task in tasks.values() if task["completed_in_session"] is not None] + [0])
    })

if __name__ == '__main__':
    app.run(debug=True) 