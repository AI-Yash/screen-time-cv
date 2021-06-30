# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition as fr
import imutils
import pickle
import time
import cv2
import numpy as np
import datetime


print("[INFO] loading encodings + face detector...")
data = pickle.loads(open('encodings.pickle', "rb").read())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
# start the FPS counter
fps = FPS().start()

screen_time = False  # is looking towards the screen?
screen_up_time = datetime.timedelta(0, 0, 0)

init_time = datetime.datetime.now()


users = {None: {'screentime': datetime.timedelta(0)}}


def add_user(name):
    users[name] = {'screentime': datetime.timedelta(0)}


for nms in data["names"]:
    add_user(nms)


def display_screen_time():
    for user in users.keys():
        if user is None:
            continue

        print(f'{user}: {users[user]["screentime"]}')


def save_data():
    with open('./data/log.csv', 'w') as datafile:
        for name in users.keys():
            # if name is None: continue

            datafile.write(f"{name},{users[name]['screentime'].seconds}\n")


def get_data():
    with open('./data/log.csv') as datafile:
        data = datafile.readlines()

    for row in data:
        row = row.replace('\n', ' ')
        d = row.split(',')
        users[d[0]]['screentime'] = datetime.timedelta(seconds=int(d[1]))
        
last_iter_people=set()

while True:
    try:
        frame = vs.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = imutils.resize(frame, width=750)
        r = frame.shape[1] / float(rgb.shape[1])
        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input frame, then compute
        # the facial embeddings for each face
        boxes = fr.face_locations(rgb, model='hog')
        num_faces=len(boxes)
        
        print('Counting' if num_faces > 0 else 'Ideal 😴   ', end='\r')
        
        encodings = fr.face_encodings(rgb, boxes)
        names = set()
        people = set() 
        
        for encoding in encodings:
            # attempt to match each face in the input image to our known
            # encodings

            matches = fr.compare_faces(data["encodings"], encoding)
            name = "Unknown"
            # check to see if we have found a match
            if True in matches:
                # find the indexes of all matched faces then initialize a
                # dictionary to count the total number of times each face
                # was matched
                matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                counts = {}
                # loop over the matched indexes and maintain a count for
                # each recognized face face
                for i in matchedIdxs:
                    name = data["names"][i]
                    counts[name] = counts.get(name, 0) + 1
                # determine the recognized face with the largest number
                # of votes (note: in the event of an unlikely tie Python
                # will select first entry in the dictionary)
                dname = max(counts, key=counts.get)

            # update the list of names
            names.add(dname)
            
            
    
            
            if not users[name].get('screen_time'):
                users[name]['screen_time'] = True
                users[name]['start_time'] = datetime.datetime.now()
                
        for p in list(last_iter_people - names):
            print(p)
            if users[p]['screen_time']:
                users[p]['screen_time'] = False
                users[p]['screentime'] += datetime.datetime.now() - users[p]['start_time']
        
        last_iter_people = names.copy()  # helper to make the logic work
        
        # all the people detected are stored in people:set
        # the names of people is copied to last_iter_people:set
        # in the next iteration we can calclate the missing people

        del frame
        time.sleep(1)
        
            
        
                    
    except KeyboardInterrupt:
        # just a check before closing the programme
        for p in list(names):
            if users[p]['screen_time']:
                users[p]['screen_time'] = False
                users[p]['screentime'] += datetime.datetime.now() - users[p]['start_time']
        
        print('*-'*20)
        print(users)
        save_data()
        print('Thank you for Using Face Screen Time')
        print('*-'*20)

        vs.stop()
       
        break

    except Exception as e:
        print('some unknown error occured')
        print(e)
        break  

    # do a bit of cleanup

vs.stop()
