import cv2
import numpy as np
import os
from keras.models import load_model
from pynput.keyboard import Key, Controller

cap = cv2.VideoCapture(0)

hsv_lower_limit = np.array([0,5,100])
hsv_upper_limit = np.array([55,255,255])
COUNTER = 1
IMG_SIZE = 100
FLAG = 0
jump = {'1':'yes', '2':'no', '3':'nothing'}

model = load_model("hand_model.h5")

keyboard = Controller()

while True:
    _,  frame = cap.read()

    shape = frame.shape

    x1 = 10
    y1 = 10
    x2 = int(0.5*shape[0])
    y2 = int(0.5*shape[0])

    pred_x = int(0.10*shape[1])
    pred_y = int(0.55*shape[0])

    cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0))

    roi = frame[y1:y2, x1:x2]

    # Converting ROI to HSV to extract brown component of image
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Extracting brown component of image into a mask
    mask = cv2.inRange(hsv, hsv_lower_limit, hsv_upper_limit)
    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    hsv = hsv & mask
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2GRAY)

    # Show HSV before applying threshold
    # cv2.imshow("hsv1", hsv)

    _, test_image = cv2.threshold(hsv, 75, 255, cv2.THRESH_BINARY)
    hsv = cv2.bitwise_and(hsv, test_image)

    # Show HSV after applying threshold
    cv2.imshow("hsv2", hsv)

    # Prediction
    img = cv2.resize(hsv, (IMG_SIZE,IMG_SIZE))
    pred = model.predict(img.reshape(1,IMG_SIZE,IMG_SIZE,1))

    font = cv2.FONT_HERSHEY_COMPLEX

    if(max(pred[0]) > 0.9):
        pred = np.argmax(pred)+1
        print(pred)
        cv2.putText(frame, jump[str(pred)], (pred_x, pred_y), font, 1, (0,0,0))

        if(pred==1 and FLAG==0):
            FLAG = 1
            keyboard.press(Key.space)
            keyboard.release(Key.space)
        
        elif pred==2:
            FLAG = 0

    # cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()