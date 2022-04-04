import cv2

cap = cv2.VideoCapture("clip_1.mp4")

if not cap.isOpened():
    print("cannot open camera")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('output1.avi', fourcc, 20.0, (1280,  720))
out2 = cv2.VideoWriter('output2.avi', fourcc, 20.0, (1280,  720))

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        print("can't receive frame")
        break

    out1.write(frame)
    frame = cv2.flip(frame, 0)
    out2.write(frame)

    # cv2.CAP_PROP_FRAME_WIDTH = 1280
    # cv2.CAP_PROP_FRAME_HEIGHT = 720

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
out1.release()
out2.release()
cv2.destroyAllWindows()



