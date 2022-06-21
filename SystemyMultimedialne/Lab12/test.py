from __future__ import print_function
import cv2 as cv
import argparse
from imutils.video import VideoStream
import numpy as np
import imutils
# parser = argparse.ArgumentParser(description='This program shows how to use background subtraction methods provided by \
#                                               OpenCV. You can process both videos and images.')
# parser.add_argument('--input', type=str, help='Path to a video or a sequence of image.', default='vtest.avi')
# parser.add_argument('--algo', type=str, help='Background subtraction method (KNN, MOG2).', default='MOG2')
# args = parser.parse_args()
# if args.algo == 'MOG2':
#     backSub = cv.createBackgroundSubtractorMOG2()
# else:
#     backSub = cv.createBackgroundSubtractorKNN()
backSub = cv.createBackgroundSubtractorMOG2()
# capture = cv.VideoCapture(cv.samples.findFileOrKeep(args.input))
vs = VideoStream(0).start()
# if not capture.isOpened():
#     print('Unable to open: ' + args.input)
#     exit(0)
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    if frame is None:
        break
    
    fgMask = backSub.apply(frame)

    mask = np.dstack([fgMask, fgMask, fgMask]).astype(np.uint8)

    print(mask)

    if mask is None:
        continue
    # encode the frame in JPEG format
    (flag, encodedImage) = cv.imencode(".jpg", mask)
    # ensure the frame was successfully encoded
    if not flag:
        continue
    
    
    cv.rectangle(mask, (10, 2), (100,20), (255,255,255), -1)
    cv.putText(mask, "XD", (15, 15),
               cv.FONT_HERSHEY_SIMPLEX, 0.5 , (0,0,0))
    
    cv.imshow('FG Mask', mask)
    
    keyboard = cv.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break