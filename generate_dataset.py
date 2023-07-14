import cv2
import numpy as np
import os

cap = cv2.VideoCapture(0)

hsv_lower_limit = np.array([0,5,100])
hsv_upper_limit = np.array([55,255,255])
COUNTER = 1

DATASET_DIR = "dataset/"
number = {'1':'yes', '2':'no', '3':'nothing'}

while True:
    _,  frame = cap.read()

    shape = frame.shape

    x1 = 10
    y1 = 10
    x2 = int(0.5*shape[0])
    y2 = int(0.5*shape[0])

    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0))

    roi = frame[y1:y2, x1:x2]

    # Converting ROI to HSV to extract brown component of image
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Extracting brown component of image into a mask
    mask = cv2.inRange(hsv, hsv_lower_limit, hsv_upper_limit)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    hsv = hsv & mask
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    _, test_image = cv2.threshold(hsv, 65, 255, cv2.THRESH_BINARY)
    hsv = cv2.bitwise_and(hsv, test_image)

    # Show HSV after applying threshold
    cv2.imshow("hsv2", hsv)

    key = cv2.waitKey(1)

    if key != -1 and key != ord('q') and COUNTER <= 100:
        cv2.imwrite(DATASET_DIR + number[str(key-48)] + '/' + str(COUNTER) + ".jpg", hsv)
        COUNTER += 1

    cv2.imshow("video", frame)

    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()