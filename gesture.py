import cv2
import math
import time
import numpy as np
from PIL import Image
from black import draw_black
from color import draw_color

video_capture = cv2.VideoCapture(0)
while(video_capture.isOpened()):
    ret, frame = video_capture.read()
    _, orig = video_capture.read()
    cv2.rectangle(frame, (300, 300), (100, 100), (0, 255, 0), 0)
    crop_frame = frame[100:300, 100: 300]
    gray = cv2.cvtColor(crop_frame, cv2.COLOR_BGR2GRAY)
    value = (35, 35)
    blurrd = cv2.GaussianBlur(gray, value, 0)
    _, thresh1 = (cv2.threshold(blurrd, 127, 255, cv2.THRESH_BINARY_INV +
                                cv2.THRESH_OTSU))
    cv2.imshow('Thresholed', thresh1)

    frame, contours, hierarchy = (cv2.findContours(thresh1.copy(),
                                  cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE))

    cnt = max(contours, key=lambda x: cv2.contourArea(x))

    x, y, w, h = cv2.boundingRect(cnt)
    cv2.rectangle(crop_frame, (x, y), (x + w, y + h), (0, 0, 255), 0)
    hull = cv2.convexHull(cnt)
    draw = np.zeros(crop_frame.shape, np.uint8)
    cv2.drawContours(draw, [cnt], 0, (0, 255, 0), 0)
    cv2.drawContours(draw, [hull], 0, (0, 0, 255), 0)
    hull = cv2.convexHull(cnt, returnPoints=False)
    defects = cv2.convexityDefects(cnt, hull)
    count_defects = 0
    cv2.drawContours(thresh1, contours, -1, (0, 255, 0), 3)
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        start = tuple(cnt[s][0])
        end = tuple(cnt[e][0])
        far = tuple(cnt[f][0])
        a = math.sqrt((end[0] - start[0])**2 + (end[1] - start[1])**2)
        b = math.sqrt((far[0] - start[0])**2 + (far[1] - start[1])**2)
        c = math.sqrt((end[0] - far[0])**2 + (end[1] - far[1])**2)
        angle = math.acos((b**2 + c**2 - a**2) / (2*b*c)) * 57
        if angle <= 90:
            count_defects += 1
            cv2.circle(crop_frame, far, 1, [0, 0, 255], -1)
        cv2.line(crop_frame, start, end, [0, 255, 0], 2)

    if count_defects == 4:
        time.sleep(1)
        _, cv_img = video_capture.read()
        cv_img = cv2.blur(cv_img, (4, 4))
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_imr = Image.fromarray(cv_img)
        img_black = draw_black(pil_imr, gammaS=1, gammaI=1)
        img_black = img_black.convert('RGB')
        img_black.save('for_test_black.jpg')
        img_black.show()
        img_color = draw_color(pil_imr, gammaS=1, gammaI=1)
        img_color = img_color.convert('RGB')
        img_color.save('for_test_color.jpg')
        img_color.show()

    cv2.imshow('origin', orig)
    cv2.imshow('Gesture', frame)
    all_img = np.hstack((draw, crop_frame))
    cv2.imshow('Contours', all_img)
    cv2.waitKey(10)
