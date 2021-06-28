import cv2
import datetime
import time
import face_recognition as fr
from fr_helper import KnownFaceEncodings
import numpy as np
from typing import List


screen_time = False  # is looking towards the screen? 
screen_up_time = datetime.timedelta(0, 0, 0)
webcam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
known_faces = KnownFaceEncodings(True)
init_time = datetime.datetime.now()

users = {None:{'screentime':datetime.timedelta(0)}}


# create object for all the people avaialable
def add_user(name):
    users[name] = {'screentime': datetime.timedelta(0)}

for name in known_faces.names:
    add_user(name)


def get_name(frame:np.array, face_loc:List[int]):
    global known_faces

    encoded_face = fr.face_encodings(frame, [face_loc])[0]
    matches = fr.compare_faces(known_faces.encodings, encoded_face)
    distance = fr.face_distance(known_faces.encodings, encoded_face)
    idx = np.argmin(distance)

    if matches[idx]:
        return known_faces.names[idx]

    return None


last_iter_people = set()  # to avoid NameError

while True:
    try:
        success, frame = webcam.read()

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Converting the image to grey for better face recognition
        # faces = face_detector.detectMultiScale(frame_gray, 1.3, 5)  # Detect for faces in the current frame
        faces = fr.face_locations(frame)
        num_faces = len(faces)

        print('Counting' if num_faces > 0 else 'Ideal 😴   ', end='\r')

        # encoded_face = fr.face_encodings(frame, faces)[0]
        # matches = fr.compare_faces(known_faces.encodings, encoded_face)
        # distance = fr.face_distance(known_faces.encodings, encoded_face)
        # idx = np.argmin(distance)

        # if matches[idx]:
        #     name = known_faces.names[idx]

        # if num_faces >= 1 and not screen_time:
        people = set()
        for face_loc in faces:
            name = get_name(frame, face_loc)
            people.add(name)
            if not users[name].get('screen_time'):
                users[name]['screen_time'] = True
                users[name]['start_time'] = datetime.datetime.now()
        # screen_time = True
        # start_time = datetime.datetime.now()

        for p in list(last_iter_people - people):
            # print(p)
            users[p]['screen_time'] = False
            users[p]['screentime'] += datetime.datetime.now() - users[p]['start_time']
        
        last_iter_people = people.copy()

        # elif num_faces <= 0 and screen_time:

        #     for face_loc in faces:
        #         name = get_name(frame, face_loc)
        #         users[name]['screen_time'] = False
        #         users[name]['screen_up_time'] = datetime.datetime.now() - users[name]['start_time']
            
        #     # screen_time = False
            # screen_up_time += datetime.datetime.now() - start_time
            # print(screen_up_time)

        del frame
        time.sleep(1)

    except KeyboardInterrupt:
        print('*-'*20)
        # print(F"{screen_up_time} is your total up time")
        print(users)
        print('Thank you for Using Face Screen Time')
        print('*-'*20)
        webcam.release()
        print(datetime.datetime.now() - init_time)
        break

    # except Exception as e:
    #     print('some unknown error occured')
    #     print(e)
    #     break

webcam.release()
