import numpy as np
import cv2
import os
import tensorflow as tf
from keras.layers import Conv2D, MaxPool2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, Input

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

inputs = Input(shape=(48,48,1))
conv1 = Conv2D(32, kernel_size=(3, 3),activation='relu')(inputs)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, kernel_size=(3, 3),activation='relu')(pool1)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(128, kernel_size=(3, 3),activation='relu')(pool2)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

x = Dropout(0.2)(pool3)

flat = Flatten()(x)

dropout = Dropout(0.5)
age_model = Dense(128, activation='relu')(flat)
age_model = dropout(age_model)
age_model = Dense(64, activation='relu')(age_model)
age_model = dropout(age_model)
age_model = Dense(32, activation='relu')(age_model)
age_model = dropout(age_model)
age_model = Dense(1, activation='relu')(age_model)

dropout = Dropout(0.5)
gender_model = Dense(128, activation='relu')(flat)
gender_model = dropout(gender_model)
gender_model = Dense(64, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(32, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(16, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(8, activation='relu')(gender_model)
gender_model = dropout(gender_model)
gender_model = Dense(1, activation='sigmoid')(gender_model)
model = tf.keras.Model(inputs=inputs, outputs=[age_model, gender_model])

def get_age(distr):
    distr = distr*5
    if distr >= 0.65 and distr <= 1.4:return "0-18"
    if distr >= 1.65 and distr <= 2.4:return "19-30"
    if distr >= 2.65 and distr <= 3.4:return "31-80"
    if distr >= 3.65 and distr <= 4.4:return "80 +"
    return "Unknown"

def get_gender(prob):
    if prob < 0.5:return "Male"
    else: return "Female"


age_dir = 'data/agegender.h5'
emotion_dir = 'data/emotion.h5'
haar_dir = 'data/haarcascade_frontalface_alt.xml'

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

model.load_weights(age_dir)
emotion_model.load_weights(emotion_dir)

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

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

        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (64, 64)), -1), 0)
        emo_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48,48)), -1), 0)

        prediction = model.predict(cropped_img)
        em_prediction = emotion_model.predict(emo_img)
        maxindex = int(np.argmax(em_prediction))

        cv2.putText(frame, 'Age: {}'.format(get_age(prediction[0])), (x+5, y-20),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Gender: {}'.format(get_gender(prediction[1])), (x+5, y-40),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'Emotion: {}'.format(emotion_dict[maxindex]), (x+5, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(
        frame, (800, 460), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
