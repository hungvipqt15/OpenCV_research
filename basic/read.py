import cv2 as cv

# img = cv.imread('Photos/3.jpg')
# cv.imshow('Cat', img)

capture = cv.VideoCapture('Videos/videoplayback.mp4')

while True:
    isTrue, frame = capture.read()
    cv.imshow('Video', frame)

    if cv.waitKey(20) & 0xFF==ord('d'):
        break
capture.release()
cv.destroyAllWindows()
