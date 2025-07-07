import os
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import img_to_array, ImageDataGenerator

# Parameters
img_width, img_height = 100, 100
batch_size = 32
epochs = 20
dataset_path = r'C:\Users\sidra\OneDrive\Desktop\d-final-face\dataSet\Ali'

# Load dataset
data = []
labels = []

# Loop over the dataset
for person_name in os.listdir(dataset_path):
    person_path = os.path.join(dataset_path, person_name)
    if os.path.isdir(person_path):
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                image = cv2.resize(image, (img_width, img_height))
                image = img_to_array(image)
                data.append(image)
                labels.append(person_name)
            else:
                print(f"Warning: Unable to load image {image_path}")

# Convert data to numpy arrays and normalize
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# Debugging: Print the shapes of data and labels
print(f"Data shape: {data.shape}")
print(f"Labels shape: {labels.shape}")

# Ensure there is data
if data.size == 0 or labels.size == 0:
    raise ValueError("Dataset contains no images or labels. Please check the dataset directory.")

# Encode labels
label_dict = {label: i for i, label in enumerate(np.unique(labels))}
labels = np.array([label_dict[label] for label in labels])
labels = to_categorical(labels)

# Split data into training and testing sets
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# Data augmentation
aug = ImageDataGenerator(rotation_range=10, zoom_range=0.1,
                         width_shift_range=0.1, height_shift_range=0.1, shear_range=0.1,
                         horizontal_flip=True, fill_mode="nearest")

# Calculate steps per epoch
steps_per_epoch = max(1, len(trainX) // batch_size)
validation_steps = max(1, len(testX) // batch_size)

# Build the model
model = Sequential()
model.add(Conv2D(32, (3, 3), padding="same", input_shape=(img_width, img_height, 1), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(64, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(128, (3, 3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Flatten())
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(label_dict), activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model
history = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
                    validation_data=(testX, testY),
                    steps_per_epoch=steps_per_epoch,
                    validation_steps=validation_steps,
                    epochs=epochs, verbose=1)

# Evaluate the model
scores = model.evaluate(testX, testY, verbose=1)
print(f"Accuracy: {scores[1] * 100}%")

# Save the model in the recommended Keras format
model.save("face_recognition_model.keras")
