import numpy as np
import cv2
import os
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization, Input
from mtcnn import MTCNN

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


emotion_model = tf.keras.Sequential([
    
    Input(shape = (48, 48, 1)),

    Conv2D(32, (3, 3), padding = 'same', activation = 'relu'),
    MaxPool2D(2),

    Conv2D(64, (3, 3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    MaxPool2D(2),
    
    Conv2D(128, (3,3), padding = 'same', activation = 'relu'),
    MaxPool2D(2),

    Conv2D(256, (3,3), padding = 'same', activation = 'relu'),
    BatchNormalization(),
    MaxPool2D(2),

    Dropout(0.3),

    Flatten(),

    Dense(512, activation = 'relu'),
    Dense(7, activation = 'softmax')
])

emotion_dir = 'data/emotion.h5'
haar_dir = 'data/haarcascade_frontalface_alt.xml'

emotion_model.load_weights(emotion_dir)

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

# dictionary which assigns each label an emotion (alphabetical order)
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# start the webcam feed

cap = cv2.VideoCapture(0)

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()

    if not ret:
        break
    
    facecasc = cv2.CascadeClassifier(haar_dir)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(
        gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(
            cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(
        frame, (800, 460), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
