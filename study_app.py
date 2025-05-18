from flask import Flask, render_template, request, jsonify, Response, session
from study_guardian import StudyGuardian, TASK_TYPES
import os
import cv2
import mediapipe as mp
import math
import threading
import time
import pygame
import numpy as np
import random

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'default-secret-key')
app.config['SESSION_TYPE'] = 'filesystem'

# Initialize StudyGuardian
study_guardian = StudyGuardian("static/alert.wav")

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize pygame for alert sound
pygame.mixer.init()
alert_sound = pygame.mixer.Sound("static/alert.wav")

# Constants for facial landmarks
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14]

# Task types
TASK_TYPES = {
    "reading": "Reading",
    "writing": "Writing",
    "problem_solving": "Problem Solving",
    "memorization": "Memorization",
    "research": "Research"
}

# Task-specific motivational quotes
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

# Global variables
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

@app.route('/')
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

@app.route('/study-stats')
def study_stats():
    stats = study_guardian.get_session_stats()
    if stats:
        return jsonify(stats)
    return jsonify({"error": "No active study session"})

@app.route('/start-break', methods=['POST'])
def start_break():
    study_guardian.start_break()
    return jsonify({"status": "success"})

@app.route('/end-break', methods=['POST'])
def end_break():
    study_guardian.end_break()
    return jsonify({"status": "success"})

@app.route('/set-study-task', methods=['POST'])
def set_study_task():
    data = request.json
    task = data.get('task')
    task_type = data.get('task_type')
    
    if task:
        study_guardian.set_current_task(task, task_type)
        return jsonify({"status": "success"})
    return jsonify({"error": "No task provided"}), 400

@app.route('/complete-task', methods=['POST'])
def complete_task():
    data = request.json
    task = data.get('task')
    completed = data.get('completed', False)
    session = data.get('session')
    
    if task:
        study_guardian.update_task(task, completed, session)
        return jsonify({"status": "success"})
    return jsonify({"error": "No task provided"}), 400

if __name__ == '__main__':
    app.run(debug=True) 