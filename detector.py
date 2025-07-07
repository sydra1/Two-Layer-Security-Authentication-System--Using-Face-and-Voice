import cv2
import numpy as np
import tensorflow as tf

# Load the trained CNN model
model = tf.keras.models.load_model(r'C:\Users\sidra\OneDrive\Desktop\d-final-face\face_recognition_model.h5')

# Load the face cascade
faceCascade = cv2.CascadeClassifier(r'C:\Users\sidra\OneDrive\Desktop\d-final-face\haarcascade_frontalface_default.xml')

# Mapping from class indices to actual labels
label_map = {}
with open("Mapping.txt", 'r') as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 2:
            idx = int(parts[0])
            name = ' '.join(parts[1:])
            label_map[idx] = name

# Open the video capture
cam = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
fontcolor = (255, 255, 255)

while True:
    ret, im = cam.read()
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(im, (x-50, y-50), (x+w+50, y+h+50), (255, 0, 0), 2)
        
        # Preprocess the face region
        face = im[y:y+h, x:x+w]
        face = cv2.resize(face, (150, 150))
        face = face.astype("float") / 255.0
        face = np.expand_dims(face, axis=0)
        
        # Predict the class of the face
        preds = model.predict(face)
        print("Predictions:", preds)
        Id = np.argmax(preds[0])
        conf = preds[0][Id]

        # Determine the label of the face
        if conf > 0.5:  # Confidence threshold
            Id = label_map.get(Id, "Unknown")
            print(Id)
        else:
            Id = "Unknown"
            print(Id)
        
        cv2.putText(im, str(Id), (x, y+h), fontface, fontscale, fontcolor) 
    
    cv2.imshow('im', im)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
