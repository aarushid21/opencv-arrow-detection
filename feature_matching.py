import math
import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


#im = cv.imread('Video_1_frame0.jpg')
#im = cv.imread('Video_6_frame111.jpg')
im = cv.imread('Video_10_frame1.jpg')
#im = cv.imread('Video_19_frame125.jpg') #this example does not work
#im = cv.imread('Video_21_frame124.jpg') 

im = cv.resize(im, (math.ceil(im.shape[1]/2), math.ceil(im.shape[0]/2)))
arrow = cv.imread('arrow.jpg')

sift = cv.SIFT_create()
kp1, des1 = sift.detectAndCompute(im,None)
kp2, des2 = sift.detectAndCompute(arrow,None)
bf = cv.BFMatcher()
matches = bf.match(des1,des2)
sorted_matches = sorted(matches, key=lambda val: val.distance)

"""

out = cv.drawMatches(im, kp1, arrow, kp2, sorted_matches[:21], None, flags=2)
plt.imshow(out), plt.show()

"""

list_kp1 = []
list_kp1 = np.array([kp1[mat.queryIdx].pt for mat in sorted_matches[:21]], np.uint)


kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(list_kp1)
ls = list_kp1[np.where(kmeans.labels_ == 0)[0]]

img_copy = np.zeros((math.ceil(im.shape[0]), math.ceil(im.shape[1]), 3), np.uint8)

for i in ls: 
    cv.circle(img_copy, i, 75, (255, 255, 255), -1) 
bit = cv.bitwise_and(img_copy, im)

im_hsv = cv.cvtColor(bit, cv.COLOR_BGR2HSV)
h, s, v = cv.split(im_hsv)
thresh_v = cv.inRange(v, 70, 120)
kernel = np.ones((3, 3), np.uint8) 
dilated = cv.dilate(thresh_v, kernel, iterations = 1)


contours, hierarchy = cv.findContours(image= dilated, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_SIMPLE)                               
image_copy = np.zeros(im.shape)
for c in contours: 
    approx = cv.approxPolyDP(c, 0.01*cv.arcLength(c, True), True)
    if len(approx) >= 7 and len(approx) <= 10:
        cv.drawContours(image_copy, [approx], -1, (0,255,0), 2)
        





cv.imshow("Image", image_copy)
cv.waitKey(0)




"""
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
matchesMask = mask.ravel().tolist()
h,w = im.shape[:2]
pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
dst = cv.perspectiveTransform(pts,M)
dst += (w, 0)  


draw_params = dict(matchColor = (0,255,0), singlePointColor = None, matchesMask = matchesMask, flags = 2)
img3 = cv.drawMatches(im, kp1, arrow, kp2, good_matches, None, **draw_params)
img3 = cv.polylines(img3, [np.int32(dst)], True, (0,0,255), 3, cv.LINE_AA)
cv.imshow("result", img3)
cv.waitKey()
# or another option for display output
#plt.imshow(img3, 'result'), plt.show()
"""

"""
out = cv.drawMatches(im, kp1, arrow, kp2, matches[:50], None, flags=2)
plt.imshow(out), plt.show()
"""