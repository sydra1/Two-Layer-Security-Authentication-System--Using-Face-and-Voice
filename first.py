import cv2
import os

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Id = input("Enter your Id: ")
Name = input("Enter your name: ")

s = ''
flag = False
with open("Mapping.txt", 'r') as r:
    for line in r:
        parts = line.split()
        if parts[0] == str(Id):
            s = parts[1]
            flag = True
            break

if not flag:
    with open("Mapping.txt", 'a') as w:
        w.write(f"{Id} {Name}\n")

sampleNum = 0
count = 0
dataset_path = 'dataSet'

while sampleNum < 50:  # Change loop condition to capture 1000 samples
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, scaleFactor=1.7, minNeighbors=5, minSize=(100, 100), flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        sampleNum += 1
        count += 1
        cv2.imwrite(os.path.join(dataset_path, f"{Name}.{Id}.{sampleNum}.jpg"), gray[y:y + h, x:x + w])
        cv2.imshow('frame', img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
