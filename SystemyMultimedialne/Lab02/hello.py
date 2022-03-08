import cv2

cap = cv2.VideoCapture("clip_1.mp4")

fourcc = cv2.VideoWriter_fourcc(*'X264')
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640,  480))
out.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
out.set(cv2.CAP_PROP_FRAME_HEIGHT,720)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # out.write(frame)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = cv2.flip(frame, 0)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break
# When everything done, release the capture
out.release()
cap.release()
cv2.destroyAllWindows()



