import numpy as np
import cv2
import os
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity

inputs = Input(shape=(48,48,1))
X = Conv2D(64, (3, 3), activation='relu', kernel_initializer = glorot_uniform(seed=0))(inputs)
X = BatchNormalization(axis = 3)(X)
X = MaxPooling2D((3, 3))(X)

X = Conv2D(128, (3, 3), activation='relu')(X)
X = MaxPooling2D((2, 2), strides=(2, 2))(X)

X = Conv2D(256, (3, 3), activation='relu')(X)
X = MaxPooling2D((2, 2))(X)

X = Flatten()(X)

dense_1 = Dense(256, activation='relu')(X)
dense_2 = Dense(256, activation='relu' )(X)
dense_3 = Dense(128, activation='relu' )(dense_2)
dropout_1 = Dropout(0.4)(dense_1)
dropout_2 = Dropout(0.4)(dense_3)
gender_model = Dense(1,activation='sigmoid', name='gender_output')(dropout_1)
age_model = Dense(1, activation='relu', name='age_output')(dropout_2)
model = tf.keras.Model(inputs=inputs, outputs=[age_model, gender_model])

def get_age(distr):
    return np.round(distr)

def get_gender(prob):
    if prob < 0.5:return "Male"
    else: return "Female"

age_dir = 'data/agegender.h5'
haar_dir = 'data/haarcascade_frontalface_alt.xml'

model.load_weights(age_dir)

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
        gray, scaleFactor=1.3, minNeighbors=3)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)

        prediction = model.predict(cropped_img)
        
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, 'Age: {}'.format((get_age(prediction[0]))), (x+5, y-30),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, 'Gender: {}'.format((get_gender(prediction[1]))), (x+5, y-60),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(
        frame, (800, 460), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
