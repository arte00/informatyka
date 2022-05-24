    YCrCb=cv2.cvtColor(img,cv2.COLOR_RGB2YCrCb).astype(int)

    Y, Cr, Cb = cv2.split(YCrCb)