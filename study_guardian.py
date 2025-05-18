import cv2
import mediapipe as mp
import math
import pygame
import time
import threading
from datetime import datetime
import random

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

# Initialize sound
pygame.mixer.init()

# Constants
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [13, 14]

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

class StudyGuardian:
    def __init__(self, alert_sound_path):
        self.alert_sound = pygame.mixer.Sound(alert_sound_path)
        self.camera = None
        self.is_monitoring = False
        self.eye_closed_time = 0
        self.tiredness_counter = 0
        self.drowsiness_patterns = []
        self.current_task = None
        self.current_task_type = None
        self.session_start_time = None
        self.break_time = None
        self.frame_generator = None
        self.current_quote = ""
        self.last_quote_time = 0
        self.tasks = {}  # Format: {task: {"completed": bool, "completed_in_session": int or None, "type": str, "drowsiness_events": list}}

    def distance(self, p1, p2):
        return math.hypot(p2[0] - p1[0], p2[1] - p1[1])

    def get_ear(self, eye):
        v1 = self.distance(eye[1], eye[5])
        v2 = self.distance(eye[2], eye[4])
        h1 = self.distance(eye[0], eye[3])
        return (v1 + v2) / (2.0 * h1)

    def play_alert(self):
        if not pygame.mixer.get_busy():
            self.alert_sound.play()

    def get_task_specific_quote(self, task_type=None):
        if task_type and task_type in TASK_SPECIFIC_QUOTES:
            quotes = TASK_SPECIFIC_QUOTES[task_type]
        else:
            quotes = GENERAL_QUOTES
        return random.choice(quotes)

    def get_break_suggestion(self):
        if not self.drowsiness_patterns:
            return None
            
        current_time = time.time()
        time_since_last_break = current_time - self.drowsiness_patterns[-1]
        
        recent_events = [t for t in self.drowsiness_patterns if current_time - t <= 300]
        
        if len(recent_events) >= 3:
            return "High drowsiness detected. Consider taking a break now!"
        elif len(recent_events) >= 2:
            return "You're showing signs of tiredness. A break might be helpful soon."
        elif time_since_last_break > 1500:  # 25 minutes
            return "You've been studying for a while. Consider a short break soon."
        
        return None

    def start_monitoring(self):
        self.is_monitoring = True
        self.eye_closed_time = 0
        self.session_start_time = time.time()
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
        self.frame_generator = self.generate_frames()

    def stop_monitoring(self):
        self.is_monitoring = False
        self.eye_closed_time = 0
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.frame_generator = None

    def generate_frames(self):
        while self.is_monitoring:
            success, frame = self.camera.read()
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

                left_ear = self.get_ear(left_eye)
                right_ear = self.get_ear(right_eye)
                ear = (left_ear + right_ear) / 2.0
                mar = self.distance(mouth[0], mouth[1]) / w

                current_time = time.time()
                if ear < 0.25:
                    self.eye_closed_time += 1
                    self.tiredness_counter += 1
                    if self.eye_closed_time > 30:
                        cv2.putText(frame, "EYES CLOSED!", (30, 100),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                        threading.Thread(target=self.play_alert).start()
                        self.drowsiness_patterns.append(current_time)
                else:
                    self.eye_closed_time = 0

                if mar > 0.03:
                    self.tiredness_counter += 1
                    cv2.putText(frame, "YAWNING...", (30, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                    self.drowsiness_patterns.append(current_time)

                # Update quote if tired
                if self.tiredness_counter > 50 and current_time - self.last_quote_time > 10:
                    self.current_quote = self.get_task_specific_quote(self.current_task_type)
                    self.last_quote_time = current_time
                    self.tiredness_counter = 0

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def get_session_stats(self):
        if not self.session_start_time:
            return None
            
        current_time = time.time()
        session_duration = current_time - self.session_start_time
        drowsiness_count = len(self.drowsiness_patterns)
        
        return {
            "duration": session_duration,
            "drowsiness_events": drowsiness_count,
            "current_task": self.current_task,
            "current_task_type": self.current_task_type,
            "break_suggestion": self.get_break_suggestion(),
            "current_quote": self.current_quote,
            "tasks": self.tasks
        }

    def set_current_task(self, task, task_type=None):
        self.current_task = task
        self.current_task_type = task_type
        if task not in self.tasks:
            self.tasks[task] = {
                "completed": False,
                "completed_in_session": None,
                "type": task_type,
                "drowsiness_events": []
            }

    def update_task(self, task, completed, session=None):
        if task in self.tasks:
            self.tasks[task]["completed"] = completed
            if completed:
                self.tasks[task]["completed_in_session"] = session
            else:
                self.tasks[task]["completed_in_session"] = None

    def start_break(self):
        self.break_time = time.time()

    def end_break(self):
        self.break_time = None
        self.drowsiness_patterns = []  # Reset drowsiness patterns after break 