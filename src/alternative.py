import os
import sys
import cv2
import numpy as np
import math
from keras.models import load_model

frame_skip = 5
age_dir = 'model/agegender.h5'
emotion_dir = 'model/emotion.h5'
emotion_model = load_model(emotion_dir)
age_model = load_model(age_dir)
emotion_dict = {0: "Angry", 1: "Neutral", 2: "Happy",
                3: "Fearful", 4: "Sad", 5: "Surprised"}


def get_age(distr):
    if distr >= 1 and distr <= 10:
        return "9-18"
    if distr >= 11 and distr <= 30:
        return "19-25"
    if distr >= 31 and distr <= 35:
        return "26-37"
    if distr >= 36 and distr <= 40:
        return "38-49"
    if distr >= 60:
        return "60 +"
    return "Unknown"


def get_gender(prob):
    if prob < 0.5:
        return "Male"
    else:
        return "Female"


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)

    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val + ((1.0 - linear_val) *
                 math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'


def process_current_frame(cnt):
    if (cnt % frame_skip == 0):
        return True
    else:
        return False


class FaceRecognition:
    face_locations = []
    face_encodings = []
    face_names = []
    known_face_encodings = []
    known_face_names = []
    count = 0

    def __init__(self):
        self.encode_faces()

    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = cv2.load_image_file(f"faces/{image}")
            face_encoding = cv2.face_encodings(face_image)[0]

            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
        print(self.known_face_names)

    def run_recognition(self):
        video_capture = cv2.VideoCapture(0)

        if not video_capture.isOpened():
            sys.exit('Video source not found...')

        while True:
            ret, frame = video_capture.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if process_current_frame(self.count):

                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

                self.face_locations = face_recognition.face_locations(
                    small_frame)
                self.face_encodings = face_recognition.face_encodings(
                    small_frame, self.face_locations)

                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding)
                    name = "Unknown"
                    confidence = '???'

                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding)

                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(
                            face_distances[best_match_index])

                    self.face_names.append(f'{name} ({confidence})')

                self.count = 0

            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                emo_img = np.expand_dims(np.expand_dims(
                    cv2.resize(gray, (48, 48)), -1), 0)
                emo_prediction = emotion_model.predict(emo_img)
                maxindex = int(np.argmax(emo_prediction))

                age_prediction = age_model.predict(emo_img)

                cv2.rectangle(frame, (left - 10, top - 10),
                              (right + 10, bottom + 10), (0, 0, 255), 2)
                cv2.putText(frame, name, (left + 6, top - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                cv2.putText(frame, "Emosi: {}".format(emotion_dict[maxindex]), (right + 15, top + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                cv2.putText(frame, "Gender: {}".format(get_gender(age_prediction[1])), (right + 15, top + 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)
                cv2.putText(frame, "Usia: {}".format(get_age(age_prediction[0])), (right + 15, top + 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (36, 255, 12), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) == ord('q'):
                break

            self.count += 1

        video_capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()
