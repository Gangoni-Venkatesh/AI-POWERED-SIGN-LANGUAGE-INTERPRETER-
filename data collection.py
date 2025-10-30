import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Camera failed to open.")
    exit()

detector = HandDetector(maxHands=1)
offset = 20
imgSize = 300
counter = 0

folder ="(C:/MINI PROJECT/SIGN-LANGUAGE DETECTION/Data/Hello)"
while True:
    success, img = cap.read()
    if not success:
        print("❌ Failed to read from camera.")
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Fixing out-of-bounds error
        imgCrop = img[max(0, y - offset): min(y + h + offset, img.shape[0]),
                      max(0, x - offset): min(x + w + offset, img.shape[1])]
        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize)) 
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap: min(wGap + wCal, imgSize)] = imgResize[:, 0:min(wCal, imgSize - wGap)]
     
        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap: min(hGap + hCal, imgSize), :] = imgResize[0:min(hCal, imgSize - hGap), :]


        cv2.imshow('ImageCrop', imgCrop)
        cv2.imshow('ImageWhite', imgWhite)

    cv2.imshow('Image', img)
    key = cv2.waitKey(1)
    if key == ord("s"):
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(counter)
