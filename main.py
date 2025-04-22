import cv2
import dlib
import math
import os
import time
import threading
import pygame
from dotenv import load_dotenv
import sounddevice as sd
from scipy.io.wavfile import write
from openai_MODEL import *

# === Initialize OpenAI Model ===
model = OpenAi()
model.initialize()

# === Setup Dlib for facial landmarks ===
cap = cv2.VideoCapture(0)
window_name = "Drowsiness Detection"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# === Global Flags ===
last_alert_time = 0
cooldown_secs = 5
is_recording = False
instructive = False
count = 0

# === Speak audio using pygame ===
def speak_the_audio(filepath="output.mp3"):
    pygame.mixer.init()
    pygame.mixer.music.load(filepath)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

# === Record 5 seconds of audio ===
def record_speak(filename):
    global is_recording
    sample_rate = 44100
    duration = 7
    is_recording = True
    print("🎤 Recording...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    write(filename, sample_rate, recording)
    print(f"✅ Audio saved as {filename}")
    is_recording = False

# === Full voice interaction flow ===
def handle_yawn_interaction():
    global instructive
    instructive = True
    record_speak("output.wav")
    threading.Thread(target = speak_the_audio, args = ("output2.mp3",)).start()
    model.response_to_require("output.wav")

    # Play the GPT response
    pygame.mixer.init()
    pygame.mixer.music.load("output1.mp3")
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        continue

    instructive = False  # Resume detection

# === Eye closure detection ===
def close_eye(landmarks):
    return all([
        math.dist((landmarks.part(37).x, landmarks.part(37).y), (landmarks.part(41).x, landmarks.part(41).y)) <= 8,
        math.dist((landmarks.part(38).x, landmarks.part(38).y), (landmarks.part(40).x, landmarks.part(40).y)) <= 8,
        math.dist((landmarks.part(43).x, landmarks.part(43).y), (landmarks.part(47).x, landmarks.part(47).y)) <= 8,
        math.dist((landmarks.part(44).x, landmarks.part(44).y), (landmarks.part(46).x, landmarks.part(46).y)) <= 8
    ])

# === Yawn detection ===
def yapping(landmarks):
    return not all([
        math.dist((landmarks.part(61).x, landmarks.part(61).y), (landmarks.part(67).x, landmarks.part(67).y)) <= 10,
        math.dist((landmarks.part(62).x, landmarks.part(62).y), (landmarks.part(66).x, landmarks.part(66).y)) <= 10,
        math.dist((landmarks.part(63).x, landmarks.part(63).y), (landmarks.part(65).x, landmarks.part(65).y)) <= 10
    ])

# === Main loop ===
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = detector(rgb_frame)

    for face in faces:
        landmarks = predictor(rgb_frame, face)

        # Draw facial landmarks
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        current_time = time.time()

        # === Yawning trigger ===
        if yapping(landmarks) and not instructive:
            cv2.putText(frame, "Yawn Detected!", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            print("😮 Yawn Detected!")
            if current_time - last_alert_time > cooldown_secs:
                last_alert_time = current_time
                threading.Thread(target=speak_the_audio, args=("output.mp3",)).start()
                threading.Thread(target=handle_yawn_interaction).start()

        # === Eye closure trigger ===
        elif close_eye(landmarks) and not instructive:
            count += 1
            cv2.putText(frame, "Eyes Closed", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 0, 255), 2)
            print("👁️ Eyes Closed")
            if count >= 10 and (current_time - last_alert_time > cooldown_secs):
                print("⚠️ Drowsiness Detected!")
                last_alert_time = current_time
                threading.Thread(target=speak_the_audio, args=("output.mp3",)).start()
                threading.Thread(target=handle_yawn_interaction).start()
        elif not instructive:
            count = 0
            print
            cv2.putText(frame, "Eyes Open", (10, 100), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

    # === Overlay system status ===
    if is_recording and not instructive:
        cv2.putText(frame, "Recording...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 165, 255), 2)
    elif instructive:
        cv2.putText(frame, "Instructive", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 215, 0), 2)

    # Show the frame
    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
