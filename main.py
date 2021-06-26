import cv2
import datetime
import time
import face_recognition as fr
from fr_helper import KnownFaceEncodings
import numpy as np


screen_time = False  # is looking towards the screen? 

screen_up_time = datetime.timedelta(0, 0, 0)

webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

known_faces = KnownFaceEncodings(True)

while True:
    try:
        success, frame = webcam.read()

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting the image to grey for better face recognition
        # faces = face_detector.detectMultiScale(frame_gray, 1.3, 5)  # Detect for faces in the current frame
        faces = fr.face_locations(frame)
        num_faces = len(faces)
        # encoded_face = fr.face_encodings(frame)

        print('Counting' if num_faces > 0 else 'Ideal 😴   ', end='\r')

        # matches = fr.compare_faces(known_faces.encodings, encoded_face)
        # distance = fr.face_distance(known_faces.encodings, encoded_face)
        # idx = np.argmin(distance)

        # if matches[idx]:
        #     name = known_faces.names[idx]

        if num_faces >= 1 and not screen_time:
            screen_time = True
            start_time = datetime.datetime.now()

        elif num_faces <= 0 and screen_time:
            screen_time = False
            screen_up_time += datetime.datetime.now() - start_time
            print(screen_up_time)

        del frame
        time.sleep(1)

    except KeyboardInterrupt:
        print('*-'*20)
        print(F"{screen_up_time} is your total up time")
        print('Thank you for Using Face Screen Time')
        print('*-'*20)
        webcam.release()
        break

    except Exception as e:
        print('some unknown error occured')
        print(e)
        break
