import cv2
import numpy as np

img = cv2.imread('Photos/4.jpg')

cv2.imshow('Cat', img)

tiny_img = img[300:600,400:600]
cv2.imshow('Tiny', tiny_img)

height, width, channels = tiny_img.shape



res = cv2.matchTemplate(img,tiny_img, cv2.TM_CCOEFF_NORMED)
threshold = 0.96
loc = np.where(res>=threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img, pt, (pt[0]+width, pt[1]+height), (0,0,255),1)

cv2.imshow('After', img)
cv2.waitKey(0)