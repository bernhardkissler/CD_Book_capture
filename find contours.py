import cv2
import numpy as np

img = cv2.imread("opnecvcdtest01.jpg", 0)
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, 1, 2)

cnt = contours[0]
M = cv2.moments(cnt)
print(M)

x, y, w, h = cv2.boundingRect(cnt)
img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

# rect = cv2.minAreaRect(cnt)
# box = cv2.boxPoints(rect)
# box = np.int0(box)
# img = cv2.drawContours(img, [box], 0, (0, 0, 255), 2)

cv2.imshow("img", img)
cv2.waitKey(0)