import cv2 as cv
import numpy as np

img  = cv.imread('Photos/2.jpg')
cv.imshow('Image', img)

blank = np.zeros(img.shape[:2], dtype='uint8')
cv.imshow('Blank Image',blank)

circle = cv.circle(blank.copy(), (img.shape[1]//2, img.shape[0]//2), 100, 255, -1)
# cv.imshow('Mask', mask)

rectangle = cv.rectangle(blank.copy(), (img.shape[1]//2+100,img.shape[0]//2), (img.shape[1]//2,img.shape[0]//2-100), 255, -1)
weird_shape = cv.bitwise_and(circle, rectangle)
cv.imshow('Weird Shape', weird_shape)

masked = cv.bitwise_and (img,img,mask=weird_shape)
cv.imshow('Masked',masked)




cv.waitKey(0)