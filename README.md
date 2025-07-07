Two-Layer Contactless Physical Authentication System (Face & Voice Recognition)
Overview
This project combines Facial Recognition and Voice Recognition into a dual-layer security authentication system to ensure enhanced, contactless, and reliable physical access control. It uses real-time face detection along with voice verification, providing a stronger alternative to traditional single-biometric authentication methods.

🚀 Key Features
Face Recognition using LBPH (Local Binary Pattern Histogram)

Voice Recognition using MFCC feature extraction and Cosine Similarity

Real-time Processing with OpenCV and Pyannote

Contactless Two-Layer Authentication

Backup Password Entry (Tkinter GUI)

Hardware Integration Ready

🛠 Technologies Used
Python

OpenCV

Pyannote.audio

Scikit-learn

Tkinter (for GUI)

NumPy, SciPy

Real-time Webcam & Microphone Inputs

🔄 System Workflow
1️⃣ Facial Recognition Process:
Capture multiple images of the user's face.

Convert images to grayscale.

Train the LBPH Face Recognizer model with images and user IDs.

During authentication:

Webcam captures live image.

If the face matches a known ID → proceed.

If not recognized → access denied.

2️⃣ Voice Recognition Process:
Record audio sample after face is recognized.

Preprocess audio: Resample to consistent format (WAV), extract MFCC features.

Generate voice embeddings.

Compare voice embeddings using Cosine Similarity against stored templates.

If similarity exceeds threshold → voice verified.

If voice fails → prompt for manual password (Tkinter GUI).

3️⃣ Integration Process:
If both Face and Voice are verified → access granted.

If either fails → system denies access or asks for password fallback.

Successful verification can trigger server communication or hardware signals (for locks, devices, etc.).

🖥 Hardware Setup (Optional)
Webcam for face detection.

Microphone for voice input.

Computer or embedded system (e.g., Raspberry Pi) to run the software.

Output device for access control (servo lock, relay, etc.).

📦 Installation Steps
Clone the repository:

bash
Copy
Edit
git clone https://github.com/YourUsername/two-layer-authentication.git
cd two-layer-authentication
Install required libraries:

bash
Copy
Edit
pip install opencv-python pyannote.audio scipy scikit-learn tkinter
Run face training script:

bash
Copy
Edit
python train_face.py
Run the main authentication system:

bash
Copy
Edit
python main_authentication.py
✅ Future Improvements
Use advanced deep learning models for face recognition (e.g., FaceNet).

Implement voice authentication using pre-trained speaker recognition models.

Enhance hardware integration for IoT-based physical security.

Add multi-user management system.

📌 Applications
Corporate and Office Security

Medical and Industrial Access Control

Smart Homes and IoT Devices

👨‍💻 Authors
Sidra Zulfiqar


