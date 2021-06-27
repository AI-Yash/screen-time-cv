# import the necessary packages
from imutils.video import VideoStream
from imutils.video import FPS
import face_recognition as fr

import imutils
import pickle
import time
import cv2
import numpy as np

prototxtPath = "./files/deploy.prototxt"
weightsPath = "./files/res10_300x300_ssd_iter_140000.caffemodel"
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxtPath, weightsPath)

print("[INFO] loading encodings + face detector...")
data = pickle.loads(open('encodings.pickle', "rb").read())

# initialize the video stream and allow the camera sensor to warm up
print("[INFO] starting video stream...")
vs = VideoStream(src=0).start()
# vs = VideoStream(usePiCamera=True).start()
time.sleep(2.0)
# start the FPS counter
fps = FPS().start()

while True:
    frame = vs.read()
    frame2 = imutils.resize(frame, width=500)
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(
        frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.5:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

        (startX, startY, endX, endY) = box.astype("int")

    encodings = fr.face_encodings(rgb, box)
    names = []

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
            name = max(counts, key=counts.get)

        # update the list of names
        names.append(name)

    for ((top, right, bottom, left), name) in zip(box, names):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom),
                      (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)
        # display the image to our screen
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    # update the FPS counter
    fps.update()

fps.stop()
print(f"[INFO] elapsed time: {fps.elapsed:.2f}")
print(f"[INFO] approx. FPS: {fps.fps:.2f}")
# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()
