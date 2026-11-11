import cv2
import numpy as np
from cv2 import boundingRect
img = cv2.imread("3.jpg")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

tem = cv2.GaussianBlur(img_gray, (5, 5), 0)

tem = cv2.threshold(tem, 191, 255, cv2.THRESH_BINARY)

kernel = np.ones((9, 9), np.uint8)
th = cv2.morphologyEx(tem[1], cv2.MORPH_CLOSE, kernel)
rv = tem[0]

min_c = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
top = min_c[0]

valid = [c for c in top if 2000 < cv2.contourArea(c) < 120000]
bot = max(valid, key=cv2.contourArea) if len(valid) > 0 else max(top, key=cv2.contourArea)
cv2_box = cv2.boundingRect(bot)

print(f"topLeft, bottomRight {cv2_box[:2]} {(cv2_box[0]+cv2_box[2], cv2_box[1]+cv2_box[3])}")
img = cv2.imread("3.jpg")
cv2.rectangle(img, cv2_box[:2], (cv2_box[0]+cv2_box[2], cv2_box[1]+cv2_box[3]), (255, 255, 255), 4)
cv2.namedWindow("img", cv2.WINDOW_NORMAL); cv2.resizeWindow("img", 1200, 800)
cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
