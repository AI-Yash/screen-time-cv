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

# add users in the users dictionary
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


def display_screen_time():
    for user in users.keys():
        if user is None: continue

        print(f'{user}: {users[user]["screentime"]}')

def save_data():
    with open('./data/log.csv', 'w') as datafile:
        datafile.write('Name, Time (in sec)\n')
        for name in users.keys():
            if name is None: continue
            
            datafile.write(f"{name},{users[name]['screentime'].seconds}\n")

def get_data():
    with open('./data/log.csv') as datafile:
        data = datafile.readlines()
    
    for row in data:
        row = row.replace('\n', ' ')
        d = row.split(',')
        
        if users.get(d[0]) is None: continue

        users[d[0]]['screentime'] = datetime.timedelta(seconds=int(d[1]))
    
get_data()

last_iter_people = set()  # to avoid NameError

while True:
    try:
        success, frame = webcam.read()

        faces = fr.face_locations(frame)
        num_faces = len(faces)

        print('Counting' if num_faces > 0 else 'Ideal 😴   ', end='\r')

        # check for faces and start the timeer
        people = set()  # to uniquely identify people
        for face_loc in faces:
            name = get_name(frame, face_loc)
            people.add(name)

            if not users[name].get('screen_time'):
                users[name]['screen_time'] = True
                users[name]['start_time'] = datetime.datetime.now()
        
        # check if some users left screen, if left then count the screentime
        for p in list(last_iter_people - people):
            # print(p)
            if users[p]['screen_time']:
                users[p]['screen_time'] = False
                users[p]['screentime'] += datetime.datetime.now() - users[p]['start_time']
        
        last_iter_people = people.copy()  # helper to make the logic work
        # all the people detected are stored in people:set
        # the names of people is copied to last_iter_people:set
        # in the next iteration we can calclate the missing people

        del frame
        time.sleep(1)

    except KeyboardInterrupt:
        # just a check before closing the programme
        for p in list(people):
            if users[p]['screen_time']:
                users[p]['screen_time'] = False
                users[p]['screentime'] += datetime.datetime.now() - users[p]['start_time']
        
        print('*-' * 20 + '*')
        
        for name in users.keys():
            if name is None: continue
            print(f"{name}: {users[name]['screentime'].seconds}\n")

        save_data()
        print('Thank you for Using Face Screen Time')
        print('*-' * 20 + '*')

        webcam.release()
       
        break

    except Exception as e:
        print('some unknown error occured')
        print(e)
        break

webcam.release()
