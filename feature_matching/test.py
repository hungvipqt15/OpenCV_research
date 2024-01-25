import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

MIN_MATCH_COUNT=10

img2 = cv.imread('Photos\davinci1.jpg',cv.IMREAD_GRAYSCALE)
img1 = cv.imread('Photos\davinci2.jpg',cv.IMREAD_GRAYSCALE)


# Khởi tạo detector FAST và descriptor
fast = cv.FastFeatureDetector_create()
brief = cv.xfeatures2d.BriefDescriptorExtractor_create()

# Tìm key points và descriptors cho ảnh gốc và ảnh so sánh
kp1 = fast.detect(img1, None)
kp2 = fast.detect(img2, None)
kp1, des1 = brief.compute(img1, kp1)
kp2, des2 = brief.compute(img2, kp2)

# Khởi tạo FLANN matcher
bf = cv.BFMatcher()
matches = bf.knnMatch(des1,des2,k=2)

# Chọn những điểm tốt nhất
good = []
for m, n in matches:
    if m.distance < 0.7 * n.distance:
        good.append(m)

if len(good)>MIN_MATCH_COUNT:
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    h,w = img1.shape
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv.perspectiveTransform(pts,M)
    img2 = cv.polylines(img2,[np.int32(dst)],True,255,3, cv.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None

draw_params = dict(matchColor = (0,255,0), # draw matches in green color
    singlePointColor = None,
    matchesMask = matchesMask, # draw only inliers
    flags = 2)
img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
plt.imshow(img3, 'gray'),plt.show()

cv.waitKey(0)