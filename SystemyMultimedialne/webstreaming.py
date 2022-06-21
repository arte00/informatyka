# import the necessary packages
from logging import shutdown

from pkg_resources import working_set
from pyimagesearch.motiondetection.singlemtiondetector import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
from flask import request
import numpy as np

import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful when multiple browsers/tabs
# are viewing the stream)
outputFrame = None
lock = threading.Lock()
# initialize a flask object
app = Flask(__name__)
# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/", methods=['GET', 'POST'])
def index():
    global working, t, target
    if request.method == 'POST':
        if  request.form.get('action') == 'Stop':
            working=False
            t.join()
            shutdown_func = request.environ.get('werkzeug.server.shutdown')
            if shutdown_func is None:
                raise RuntimeError('Not running werkzeug')
            shutdown_func()
        elif request.form.get('action') == 'Detect':
            working = False
            target = "detect"
        elif request.form.get('action') == 'Normal':
            working = False
            target = "normal"
        elif request.form.get('action') == 'Face':
            working = False
            target = "face"
        elif request.form.get('action') == 'Motion':
            working = False
            target = "motion"
        elif request.form.get('action') == 'Flow':
            working = False
            target = "flow"
    elif request.method == 'GET':
        return render_template('index.html',form=request.form)
    
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(generate(),
        mimetype = "multipart/x-mixed-replace; boundary=frame")

def action():

    global working, vs, outputFrame, lock, target
    while True:

        working = True

        if target == 'normal':
            normal()
        elif target == 'detect':
            detect_motion(32)
        elif target == 'face':
            face_recognition()
        elif target == 'motion':
            motion()
        elif target == 'flow':
            flow()

def flow():
    global working, vs, outputFrame, lock

    feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

    lk_params = dict( winSize  = (15, 15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    color = np.random.randint(0, 255, (100, 3))
    # Take first frame and find corners in it
    old_frame = vs.read()
    old_frame = imutils.resize(old_frame, width=400)
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    mask = np.zeros_like(old_frame)

    while working:

        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        # Select good points
        if p1 is not None:
            good_new = p1[st==1]
            good_old = p0[st==1]
        # draw the tracks
        for i, (new, old) in enumerate(zip(good_new, good_old)):
            a, b = new.ravel()
            c, d = old.ravel()
            mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
            frame = cv2.circle(frame, (int(a), int(b)), 5, color[i].tolist(), -1)
        img = cv2.add(frame, mask)

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)


        with lock:
            outputFrame = img.copy()

def motion():

    global working, vs, outputFrame, lock

    backSub = cv2.createBackgroundSubtractorMOG2()

    while working:

        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        fgMask = backSub.apply(frame)

        # mask = np.dstack([fgMask, fgMask, fgMask]).astype(np.uint8)

        # out = np.concatenate

        with lock:
            outputFrame = fgMask.copy()



def face_recognition():

    global working, vs, outputFrame, lock

    while working:

        face_cascade = cv2.CascadeClassifier()
        eyes_cascade = cv2.CascadeClassifier()

        face_cascade.load(cv2.samples.findFile('C:\\Users\\huber\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python38\\site-packages\\cv2\\data\\haarcascade_frontalface_alt.xml'))
        eyes_cascade.load(cv2.samples.findFile('C:\\Users\\huber\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.8_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python38\\site-packages\\cv2\\data\\haarcascade_eye_tree_eyeglasses.xml'))

        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_gray = cv2.equalizeHist(frame_gray)

        faces = face_cascade.detectMultiScale(frame_gray)
        for (x,y,w,h) in faces:
            center = (x + w//2, y + h//2)
            frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4)
            faceROI = frame_gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            eyes = eyes_cascade.detectMultiScale(faceROI)
            for (x2,y2,w2,h2) in eyes:
                eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                radius = int(round((w2 + h2)*0.25))
                frame = cv2.circle(frame, eye_center, radius, (255, 0, 0 ), 4)

        # grab the current timestamp and draw it on the frame
        # timestamp = datetime.datetime.now()
        # cv2.putText(frame, timestamp.strftime(
        #     "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        with lock:
            outputFrame = frame.copy()

def normal():

    global working, vs, outputFrame, lock

    while working:

        frame = vs.read()
        frame = imutils.resize(frame, width=400)

        # grab the current timestamp and draw it on the frame
        # timestamp = datetime.datetime.now()
        # cv2.putText(frame, timestamp.strftime(
        #     "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
        #     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
        with lock:
            outputFrame = frame.copy()


def detect_motion(frameCount):
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, lock, working
    # initialize the motion detector and the total number of frames
    # read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0

    # loop over frames from the video stream
    while working:

        # read the next frame from the video stream, resize it,
        # convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=400)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        # grab the current timestamp and draw it on the frame
        timestamp = datetime.datetime.now()
        cv2.putText(frame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

        # if the total number of frames has reached a sufficient
        # number to construct a reasonable background model, then
        # continue to process the frame
        if total > frameCount:
            # detect motion in the image
            motion = md.detect(gray)
            # check to see if motion was found in the frame
            if motion is not None:
                # unpack the tuple and draw the box surrounding the
                # "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),
                    (0, 0, 255), 2)
        
        # update the background model and increment the total number
        # of frames read thus far
        md.update(gray)
        total += 1
        # acquire the lock, set the output frame, and release the
        # lock
        with lock:
            outputFrame = frame.copy()

def generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock, target
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            timestamp = datetime.datetime.now()
            cv2.putText(outputFrame, timestamp.strftime(
            "%A %d %B %Y %I:%M:%S%p"), (10, outputFrame.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(encodedImage) + b'\r\n')

if __name__ == '__main__':
    args = {"ip" : "127.0.0.1", "port": 8080, "frame_count": 32}

    global working, t, target

    target = 'normal'
    
    # start a thread that will perform motion detection
    t = threading.Thread(target=action)
    t.daemon = True
    t.start()
    # start the flask app
    app.run(host="127.0.0.1", port="8080", debug=True,threaded=True, use_reloader=False)

    #####
    working=False
# release the video stream pointer
vs.stop()
