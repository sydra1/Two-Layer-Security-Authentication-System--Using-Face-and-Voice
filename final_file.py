# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 19:36:44 2024

@author: sidra
"""

import os
import cv2
import numpy as np
import face_recognition
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import librosa
import pickle
import pyaudio
import wave

# Paths to datasets
FACE_DATASET_PATH = r'C:\\Users\\sidra\\OneDrive\\Desktop\\dataSet'
VOICE_DATASET_PATH = r'C:\\Users\\sidra\\OneDrive\\Desktop\\AUDIO_VOICE\\Testing_Audio\\Sidra-002'

# Face Recognition System
class FaceRecognition:
    def __init__(self):
        self.encodings = []
        self.names = []

    def load_dataset(self, dataset_path):
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith(('jpg', 'jpeg', 'png')):
                    image_path = os.path.join(root, file)
                    image = face_recognition.load_image_file(image_path)
                    encoding = face_recognition.face_encodings(image)
                    if encoding:
                        self.encodings.append(encoding[0])
                        self.names.append(os.path.basename(root))

    def train_model(self):
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.names)
        self.clf = SVC(C=1.0, kernel='linear', probability=True)
        self.clf.fit(self.encodings, self.labels)
        with open('face_recognition_model.pkl', 'wb') as f:
            pickle.dump((self.le, self.clf), f)

    def detect_and_recognize(self):
        with open('face_recognition_model.pkl', 'rb') as f:
            self.le, self.clf = pickle.load(f)

        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb_frame = frame[:, :, ::-1]
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = self.clf.predict_proba([face_encoding])
                best_match_index = np.argmax(matches)
                name = self.le.inverse_transform([best_match_index])[0]

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Face Recognition', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

# Voice Recognition System
class VoiceRecognition:
    def __init__(self):
        self.gmm = None

    def extract_features(self, audio_path):
        y, sr = librosa.load(audio_path, sr=16000)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)

    def train_model(self, dataset_path):
        features = []
        labels = []
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.wav'):
                    audio_path = os.path.join(root, file)
                    label = os.path.basename(root)
                    features.append(self.extract_features(audio_path))
                    labels.append(label)

        self.le = LabelEncoder()
        y = self.le.fit_transform(labels)
        X = np.array(features)

        self.clf = SVC(C=1.0, kernel='linear', probability=True)
        self.clf.fit(X, y)
        with open('voice_recognition_model.pkl', 'wb') as f:
            pickle.dump((self.le, self.clf), f)

    def recognize_speaker(self, audio_path):
        with open('voice_recognition_model.pkl', 'rb') as f:
            self.le, self.clf = pickle.load(f)

        features = self.extract_features(audio_path)
        proba = self.clf.predict_proba([features])
        best_match_index = np.argmax(proba)
        speaker = self.le.inverse_transform([best_match_index])[0]
        return speaker

    def record_audio(self, filename):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = filename

        p = pyaudio.PyAudio()

        stream = p.open(format=FORMAT,
                        channels=CHANNELS,
                        rate=RATE,
                        input=True,
                        frames_per_buffer=CHUNK)

        print("* recording")

        frames = []

        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)

        print("* done recording")

        stream.stop_stream()
        stream.close()
        p.terminate()

        wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()

# Combine Systems
class TwoLayerAuthenticationSystem:
    def __init__(self):
        self.face_recognition = FaceRecognition()
        self.voice_recognition = VoiceRecognition()

    def train_models(self):
        print("Training face recognition model...")
        self.face_recognition.load_dataset(FACE_DATASET_PATH)
        self.face_recognition.train_model()

        print("Training voice recognition model...")
        self.voice_recognition.train_model(VOICE_DATASET_PATH)

    def authenticate(self):
        print("Starting face recognition...")
        self.face_recognition.detect_and_recognize()

        print("Starting voice recognition...")
        self.voice_recognition.record_audio('test.wav')
        speaker = self.voice_recognition.recognize_speaker('test.wav')
        print(f"Speaker: {speaker}")

if __name__ == "__main__":
    system = TwoLayerAuthenticationSystem()
    system.train_models()
    system.authenticate()
