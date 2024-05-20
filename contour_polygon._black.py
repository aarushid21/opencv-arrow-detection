import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt


im = cv.imread('Video_1_frame0.jpg')
#im = cv.imread('Video_10_frame1.jpg')
#im = cv.imread('Video_19_frame125.jpg')
#im = cv.imread('Video_21_frame124.jpg') //this example does not work


#convert to hsv
im_hsv = cv.cvtColor(im, cv.COLOR_BGR2HSV)
h, s, v = cv.split(im_hsv)
#thresh_s = cv.inRange(s, 50, 120)
#thresh_v = cv.inRange(v, 80, 100)


#apply binary threshold on hsv
thresh_s = cv.inRange(s, 50, 130)
thresh_v = cv.inRange(v, 70, 120)
thresh_combined = cv.bitwise_and(thresh_s, thresh_v)
kernel = np.ones((3, 3), np.uint8) 
eroded = cv.erode(thresh_combined, kernel)
dilated = cv.dilate(eroded, kernel, iterations = 5)


#apply contouring and polygon approximation 
"""
areas = []
contours, hierarchy = cv.findContours(image=dilated, mode=cv.RETR_TREE, method= cv.CHAIN_APPROX_NONE)                               
image_copy = np.zeros(im.shape)
for c in contours: 
    approx = cv.approxPolyDP(c, 0.01*cv.arcLength(c, True), True)
    (x,y) = c[0,0]
    if len(approx) >= 7 and len(approx) <= 10:
    #if len(approx) >= 7: 
      cv.drawContours(image_copy, [approx], -1, (0,255,0), 3)
      areas.append(cv.contourArea(c))
"""

#displaying polygon with max area only 
max_area = 0
max_contour = None
contours, hierarchy = cv.findContours(image=dilated, mode = cv.RETR_TREE, method = cv.CHAIN_APPROX_SIMPLE)                               
image_copy = np.zeros(im.shape)
for c in contours: 
    approx = cv.approxPolyDP(c, 0.01*cv.arcLength(c, True), True)
    (x,y) = c[0,0]
    if len(approx) > 7 and len(approx) <= 10:
        cv.drawContours(image_copy, [approx], -1, (0,255,0), 3)
        if cv.contourArea(c) > max_area: 
            max_contour = c
            max_area = cv.contourArea(c)

#Draw border around detected arrow
approx =  cv.approxPolyDP(max_contour, 0.01*cv.arcLength(max_contour, True), True)
cv.drawContours(image_copy, [approx], -1, (0,0, 255), 5)

#Use moment of intertia to determine bounding box
#x,y,w,h = cv.boundingRect(max_contour)
#cv.rectangle(image_copy,(x,y),(x+w,y+h),(0,0, 255),2)
m = cv.moments(max_contour)
cx = int(m['m10']/m['m00'])
print(cx)




#display final result
while(True):
    cv.imshow("Result", image_copy)
    c = cv.waitKey()
    if c == ord('q'): 
        break

"""
cv.imshow("Result", image_copy)
cv.waitKey(2000)              

"""

