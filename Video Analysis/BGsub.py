import cv2

backSub1 = cv2.createBackgroundSubtractorMOG2()
backSub2 = cv2.createBackgroundSubtractorKNN()

capture = cv2.VideoCapture(0)

while True:
    ret, frame = capture.read()
    if frame is None:
        break

    fgMask1 = backSub1.apply(frame)
    fgMask2 = backSub1.apply(frame)

    cv2.imshow('Frame',frame)
    cv2.imshow('MOG2 FG Mask', fgMask1)
    cv2.imshow('KNN FG Mask', fgMask2)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break
